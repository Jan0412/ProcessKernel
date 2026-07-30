import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    X, W, Out,
    N, CO, C, H, W,
    H_OUT, W_OUT,
    stride_h, stride_w,
    pad_h, pad_w,
    KH, KW,
    BLOCK_CO: tl.constexpr
):
    # Map program ID to output spatial location and batch index
    pid = tl.program_id(0)

    # Total number of (n, h_out, w_out) combinations
    total_spatial = N * H_OUT * W_OUT

    # Calculate n, h_out, w_out from pid
    n = pid // (H_OUT * W_OUT)
    rem = pid % (H_OUT * W_OUT)
    h_out = rem // W_OUT
    w_out = rem % W_OUT

    # Define offsets for output channels
    co_offsets = tl.arange(0, BLOCK_CO)
    co_mask = co_offsets < CO

    # Initialize output accumulator
    out_vals = tl.zeros([BLOCK_CO], dtype=tl.float32)

    # Loop over kernel spatial dimensions
    for kh in range(KH):
        for kw in range(KW):
            # Calculate input coordinates
            h_in = h_out * stride_h + kh - pad_h
            w_in = w_out * stride_w + kw - pad_w

            # Check bounds for input coordinates
            # If out of bounds, the value is 0 (due to padding)
            h_valid = (h_in >= 0) & (h_in < H)
            w_valid = (w_in >= 0) & (w_in < W)
            valid = h_valid & w_valid

            if not valid:
                continue

            # Loop over input channels
            for c in range(C):
                # Calculate input pointer offset
                # X is (N, C, H, W)
                # offset = n * C * H * W + c * H * W + h_in * W + w_in
                x_ptr = X + n * C * H * W + c * H * W + h_in * W + w_in

                # Load input value (broadcasted to all threads in block for this channel)
                # We use tl.load with a scalar offset if needed, or just load a single value
                # Since it's a single value for all CO, we load it once
                x_val = tl.load(x_ptr)

                # Weight is (CO, C, KH, KW)
                # offset = co * C * KH * KW + c * KH * KW + kh * KW + kw
                # We need to load a vector of weights for all CO in this block
                w_ptr_base = W + c * KH * KW + kh * KW + kw
                # The stride for CO in weight is C * KH * KW
                w_stride = C * KH * KW

                # Load weights for all co in block
                w_vals = tl.load(w_ptr_base + co_offsets * w_stride, mask=co_mask)

                # Accumulate
                out_vals += x_val * w_vals

    # Store output
    # Out is (N, CO, H_OUT, W_OUT)
    # offset = n * CO * H_OUT * W_OUT + co * H_OUT * W_OUT + h_out * W_OUT + w_out
    out_ptr_base = Out + n * CO * H_OUT * W_OUT + h_out * W_OUT + w_out
    out_stride = H_OUT * W_OUT

    tl.store(out_ptr_base + co_offsets * out_stride, out_vals, mask=co_mask)

def triton_conv2d(x, weight):
    # x: (N, C, H, W)
    # weight: (CO, C, KH, KW)

    N, C, H, W = x.shape
    CO, C_w, KH, KW = weight.shape

    assert C == C_w

    # Calculate output dimensions
    H_OUT = (H + 2 * 2 - KH) // 4 + 1 # padding=2, stride=4 fixed in problem
    W_OUT = (W + 2 * 2 - KW) // 4 + 1

    # Padding and Stride values
    pad_h, pad_w = 2, 2
    stride_h, stride_w = 4, 4

    # Create output tensor
    out = torch.empty((N, CO, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    BLOCK_CO = 64
    grid = (N * H_OUT * W_OUT,)

    conv2d_kernel[grid](
        x, weight, out,
        N, CO, C, H, W,
        H_OUT, W_OUT,
        stride_h, stride_w,
        pad_h, pad_w,
        KH, KW,
        BLOCK_CO=BLOCK_CO
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        # Replace F.conv2d with our custom kernel
        x = triton_conv2d(x, self.conv1.weight)
        return x