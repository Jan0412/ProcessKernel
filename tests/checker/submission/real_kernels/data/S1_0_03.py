import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_h_x, stride_w_x, stride_c_x, stride_b_x,
    stride_h_w, stride_w_w, stride_c_w, stride_c_w,
    stride_h_o, stride_w_o, stride_c_o, stride_b_o,
    in_h, in_w, out_h, out_w, in_c, out_c,
    K, pad, stride,
    BLOCK_B: tl.constexpr, BLOCK_C_OUT: tl.constexpr, BLOCK_C_IN: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Program coordinates
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Base output coordinates
    ho_base = pid_h * BLOCK_H
    wo_base = pid_w * BLOCK_W

    # Offsets within the block
    off_h = tl.arange(0, BLOCK_H)
    off_w = tl.arange(0, BLOCK_W)
    off_c_out = tl.arange(0, BLOCK_C_OUT)
    off_c_in = tl.arange(0, BLOCK_C_IN)
    off_b = tl.arange(0, BLOCK_B)

    # Output shape for masks
    mask_h = (ho_base + off_h) < out_h
    mask_w = (wo_base + off_w) < out_w
    mask_c_out = (pid_c * BLOCK_C_OUT + off_c_out) < out_c
    mask_b = (pid_b * BLOCK_B + off_b) < in_c

    # We will accumulate results in registers
    # Shape: [BLOCK_B, BLOCK_C_OUT, BLOCK_H, BLOCK_W]
    acc = tl.zeros((BLOCK_B, BLOCK_C_OUT, BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Iterate over input channels
    for c in range(0, in_c, BLOCK_C_IN):
        c_idx = pid_c * BLOCK_C_IN + c + off_c_in
        mask_c_in = c_idx < in_c

        # Iterate over kernel spatial dimensions
        for ki in range(K):
            for kj in range(K):
                # Compute input coordinates for this kernel offset
                # Formula: h_in = h_out - ki + pad - (stride - 1) * h_out ... wait, standard formula:
                # For transposed conv: output[ho, wo] += input[hi, wi] * weight[ho - hi, wo - wi]
                # hi = ho - ki + pad
                hi = ho_base + off_h - ki + pad
                wi = wo_base + off_w - kj + pad

                # Masks for input bounds
                mask_hi = (hi >= 0) & (hi < in_h)
                mask_wi = (wi >= 0) & (wi < in_w)

                # Load weights: W[out_c, in_c, K, K]
                # We need W[co, c, ki, kj]
                w_offsets = (off_c_out[:, None, None, None] * stride_c_w) + \
                            (c_idx[None, :, None, None] * stride_c_w) + \
                            (ki * stride_h_w + kj * stride_w_w)
                w_mask = mask_c_out[:, None, None, None] & mask_c_in[None, :, None, None]
                w = tl.load(W_ptr + w_offsets, mask=w_mask, other=0.0)

                # Load inputs: X[b, c, h, w]
                # We need X[b, c, hi, wi]
                x_offsets = (off_b[:, None, None, None] * stride_b_x) + \
                            (c_idx[None, :, None, None] * stride_c_x) + \
                            (hi[None, None, :, None] * stride_h_x) + \
                            (wi[None, None, None, :] * stride_w_x)
                x_mask = mask_b[:, None, None, None] & mask_c_in[None, :, None, None] & \
                         mask_hi[None, None, :, None] & mask_wi[None, None, None, :]
                x = tl.load(X_ptr + x_offsets, mask=x_mask, other=0.0)

                # Accumulate
                acc += w * x

    # Store output
    # Output coordinates
    o_h = ho_base + off_h
    o_w = wo_base + off_w
    o_c = pid_c * BLOCK_C_OUT + off_c_out
    o_b = pid_b * BLOCK_B + off_b

    o_offsets = (o_b[:, None, None, None] * stride_b_o) + \
                (o_c[None, :, None, None] * stride_c_o) + \
                (o_h[None, None, :, None] * stride_h_o) + \
                (o_w[None, None, None, :] * stride_w_o)

    o_mask = mask_b[:, None, None, None] & mask_c_out[None, :, None, None] & \
             mask_h[None, None, :, None] & mask_w[None, None, None, :]

    tl.store(Out_ptr + o_offsets, acc, mask=o_mask)


def triton_transpose_2d(x, weight, stride=1, padding=0, kernel_size=3):
    """
    Wrapper for the Triton transposed 2D convolution kernel.
    """
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()

    B, C_in, H_in, W_in = x.shape
    C_out, _, K, _ = weight.shape

    # Calculate output dimensions
    H_out = (H_in - 1) * stride - 2 * padding + K
    W_out = (W_in - 1) * stride - 2 * padding + K

    out = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Block sizes
    BLOCK_B = 1
    BLOCK_C_OUT = 64
    BLOCK_C_IN = 64
    BLOCK_H = 128
    BLOCK_W = 128

    # Grid configuration
    n_blocks_b = (B + BLOCK_B - 1) // BLOCK_B
    n_blocks_c = (C_out + BLOCK_C_OUT - 1) // BLOCK_C_OUT
    n_blocks_h = (H_out + BLOCK_H - 1) // BLOCK_H
    n_blocks_w = (W_out + BLOCK_W - 1) // BLOCK_W

    grid = (n_blocks_b, n_blocks_c, n_blocks_h, n_blocks_w)

    # Strides
    stride_h_x = W_in
    stride_w_x = 1
    stride_c_x = H_in * W_in
    stride_b_x = C_in * H_in * W_in

    stride_h_w = K
    stride_w_w = 1
    stride_c_w = K * K
    stride_c_w = C_in * K * K

    stride_h_o = W_out
    stride_w_o = 1
    stride_c_o = H_out * W_out
    stride_b_o = C_out * H_out * W_out

    conv_transpose_2d_kernel[grid](
        x, weight, out,
        stride_h_x, stride_w_x, stride_c_x, stride_b_x,
        stride_h_w, stride_w_w, stride_c_w, stride_c_w,
        stride_h_o, stride_w_o, stride_c_o, stride_b_o,
        H_in, W_in, H_out, W_out, C_in, C_out,
        K, padding, stride,
        BLOCK_B, BLOCK_C_OUT, BLOCK_C_IN, BLOCK_H, BLOCK_W
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_transpose_2d(x, self.conv_transpose2d.weight, self.conv_transpose2d.stride[0], self.conv_transpose2d.padding[0], self.conv_transpose2d.kernel_size[0])


def get_inputs():
    batch_size = 8
    in_channels = 64
    out_channels = 64
    kernel_size = 3
    height = 1024
    width = 1024
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [64, 64, 3]