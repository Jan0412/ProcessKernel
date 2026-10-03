import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        # Keep the original layer to access parameters or replicate logic
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        # Store hyperparameters for the kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def forward(self, x):
        return triton_conv2d(x, self.conv2d.weight, self.conv2d.bias, 
                             self.in_channels, self.out_channels, 
                             self.kernel_size, self.stride, self.padding, 
                             self.dilation, self.groups, self.bias)

@triton.jit
def conv2d_kernel(
    X, W, B, Out,
    N, C, H, W,
    C_OUT, K_H, K_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    DILATION_H, DILATION_W,
    GROUPS,
    H_OUT, W_OUT,
    BLOCK_K: tl.constexpr
):
    # Grid dimensions: (C_OUT * N, H_OUT * W_OUT) 
    # Actually let's map program_id(0) to (C_OUT, N) and program_id(1) to (H_OUT, W_OUT)
    # To keep grid small and manageable, let's map:
    # pid_x = C_OUT, pid_y = N, pid_z = H_OUT, pid_w = W_OUT? 
    # Triton usually supports 3D grid. Let's use 3D grid: (C_OUT, N, H_OUT * W_OUT)

    pid_c_out = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_spatial = tl.program_id(2)

    # Decode spatial
    w_out = pid_spatial % W_OUT
    h_out = pid_spatial // W_OUT

    # Output index
    # Out shape: [N, C_OUT, H_OUT, W_OUT]
    out_ptr_base = pid_n * (C_OUT * H_OUT * W_OUT) + pid_c_out * (H_OUT * W_OUT) + pid_spatial
    out_ptr = Out + out_ptr_base

    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)

    # Loop over K = C * K_H * K_W
    # We need to iterate over groups, channels_in_group, kh, kw
    # Total iterations = C * K_H * K_W

    # We will iterate with a block size BLOCK_K
    for k_start in range(0, C * K_H * K_W, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < (C * K_H * K_W)

        # Decode k_offsets into (c_in, h_k, w_k)
        k_flat = k_offsets
        w_k = k_flat % K_W
        rem = k_flat // K_W
        h_k = rem % K_H
        c_in_flat = rem // K_H

        c_in = c_in_flat % (C // GROUPS)
        g = c_in_flat // (C // GROUPS)

        # Only process if group matches the output channel's group
        g_target = pid_c_out // (C_OUT // GROUPS)
        group_match = g == g_target

        # Calculate input coordinates
        # h_in = h_out * stride_h + h_k * dilation_h - pad_h
        h_in = h_out * STRIDE_H + h_k * DILATION_H - PAD_H
        w_in = w_out * STRIDE_W + w_k * DILATION_W - PAD_W

        # Check bounds
        in_valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & group_match

        # Load Input
        # X shape: [N, C, H, W]
        # Index: n * C * H * W + c_in * H * W + h_in * W + w_in
        x_idx = pid_n * (C * H * W) + c_in * (H * W) + h_in * W + w_in
        x_val = tl.load(X + x_idx, mask=in_valid, other=0.0)

        # Load Weight
        # W shape: [C_OUT, C, K_H, K_W]
        # Index: c_out * C * K_H * K_W + c_in * K_H * K_W + h_k * K_W + w_k
        w_idx = pid_c_out * (C * K_H * K_W) + c_in * (K_H * K_W) + h_k * K_W + w_k
        w_val = tl.load(W + w_idx, mask=in_valid, other=0.0)

        acc = acc + tl.sum(x_val * w_val)

    # Add Bias
    if B is not None:
        bias_val = tl.load(B + pid_c_out)
        acc = acc + bias_val

    tl.store(out_ptr, acc)

def triton_conv2d(x, weight, bias, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, has_bias):
    N, C, H, W = x.shape
    K_H, K_W = kernel_size
    S_H, S_W = stride
    P_H, P_W = padding
    D_H, D_W = dilation

    H_OUT = (H + 2 * P_H - D_H * (K_H - 1) - 1) // S_H + 1
    W_OUT = (W + 2 * P_W - D_W * (K_W - 1) - 1) // S_W + 1

    # Output tensor
    out = torch.empty((N, out_channels, H_OUT, W_OUT), dtype=x.dtype, device=x.device)

    # Grid
    # We use 3D grid: (C_OUT, N, H_OUT * W_OUT)
    # However, Triton grid usually takes a tuple of ints. 
    # If H_OUT * W_OUT is large, we are fine.

    grid = (out_channels, N, H_OUT * W_OUT)

    # Handle bias
    b_ptr = bias if has_bias else None

    # Launch
    BLOCK_K = 8 # Tunable
    conv2d_kernel[grid](
        x, weight, b_ptr, out,
        N, C, H, W,
        out_channels, K_H, K_W,
        S_H, S_W,
        P_H, P_W,
        D_H, D_W,
        groups,
        H_OUT, W_OUT,
        BLOCK_K=BLOCK_K
    )
    return out