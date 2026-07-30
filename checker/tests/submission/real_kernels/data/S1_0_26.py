import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def triton_conv_transpose_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    # Dimensions
    N, C_in, C_out, H_in, W_in, H_out, W_out,
    K_h, W_out, W_out,
    # Strides
    stride_h, stride_w,
    # Padding
    pad_h, pad_w,
    # Output padding
    out_pad_h, out_pad_w,
    # Pointers
    BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    # Weights
    w_ptr,
    # Bias
    b_ptr,
    # Output
    out_ptr,
    # Dimensions
    N, C_in, C_out, H_in, W_in, H_out, W_out,
    # Strides
    stride_h, stride_w,
    # Padding
    pad_h, pad_w,
    # Output padding
    out_pad_h, out_pad_w,
    # Block sizes
    BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # 1. Calculate global indices for this block
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # 2. Create offsets for N, H, W
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    off_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # 3. Load input and weights
    # Input shape: (N, C_in, H_in, W_in)
    # Weight shape: (C_in, C_out // groups, K_h, K_w)
    # Bias shape: (C_out,)
    # Output shape: (N, C_out, H_out, W_out)

    # We will iterate over C_out in the kernel, but for simplicity, let's assume C_out is small or we can loop over it.
    # Actually, C_out=64. Let's loop over C_out in the kernel.
    # But wait, Triton is better at parallelizing over C_out. Let's make grid over (N, H_out, W_out) and loop over C_out in the kernel.
    # Or grid over (N * C_out // BLOCK_NC, H_out // BLOCK_H, W_out // BLOCK_W). Let's do grid over (N, C_out, H_out, W_out) is too many blocks.
    # Let's do grid over (N, C_out, H_out, W_out) -> too many blocks.
    # Let's do grid over (N * C_out // BLOCK_NC, H_out // BLOCK_H, W_out // BLOCK_W) -> 128 * 64 / 8 = 1024 blocks in N*C_out dim. 256/32 = 8 blocks in H. 256/32 = 8 blocks in W. Total 1024 * 8 * 8 = 65536 blocks. Good.
    pass

    # Wait, let's restructure the kernel to loop over C_in and K_h, K_w inside.
    # For each (N, C_out, H_out, W_out), we compute:
    # out[N, C_out, H_out, W_out] = sum_{C_in, K_h, K_w} x[N, C_in, H_in, W_in] * w[C_in, C_out, K_h, K_w] + b[C_out]
    # where H_in = (H_out - 1) // stride_h + 1 - pad_h + ... -> Actually, H_in is given.
    # The mapping from (H_out, W_out) to (H_in, W_in) is:
    # H_in = (H_out - 1) // stride_h + 1 - pad_h + ... -> No, ConvTranspose2d maps (H_in, W_in) to (H_out, W_out).
    # H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h + out_pad_h.
    # So for a given (H_out, W_out), the corresponding (H_in, W_in) are:
    # H_in = (H_out - out_pad_h - K_h + 1) // stride_h + 1 + pad_h = (H_out - K_h + 1) // stride_h + 1 + pad_h.
    # Let's verify: H_in = 128, K_h = 3, stride_h = 2, pad_h = 1, out_pad_h = 1.
    # H_out = 256.
    # H_in = (256 - 3 + 1) // 2 + 1 + 1 = 254 // 2 + 2 = 127 + 2 = 129. Wait, 129 != 128.
    # Let's use the exact formula from PyTorch:
    # H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h + out_pad_h.
    # => H_in = (H_out - K_h - out_pad_h + 2 * pad_h) / stride_h + 1.
    # For H_out = 256: H_in = (256 - 3 - 1 + 2) / 2 + 1 = 254 / 2 + 1 = 127 + 1 = 128. Correct.
    # So H_in = (H_out - K_h - out_pad_h + 2 * pad_h) // stride_h + 1.
    # But we need to iterate over K_h, K_w to find all (H_in, W_in) that contribute to (H_out, W_out).
    # The contribution from (H_in, W_in) to (H_out, W_out) is:
    # H_out = H_in * stride_h + K_h - 1 - 2 * pad_h + out_pad_h? No.
    # Let's use the standard im2col approach for ConvTranspose2d.
    # For each (N, C_out, H_out, W_out), the valid (H_in, W_in) are those where:
    # H_in * stride_h + k_h - pad_h <= H_out < (H_in + 1) * stride_h + k_h - pad_h + out_pad_h?
    # Actually, the condition is:
    # H_out - k_h + pad_h <= H_in * stride_h <= H_out + out_pad_h - 1.
    # => H_in = floor((H_out - k_h + pad_h) / stride_h).
    # Let's just loop over k_h, k_w and compute H_in, W_in. If H_in, W_in are within bounds, add to sum.
    # H_in = (H_out - k_h + pad_h) // stride_h.
    # W_in = (W_out - k_w + pad_w) // stride_w.
    # Check if H_in >= 0 and H_in < H_in and W_in >= 0 and W_in < W_in.
    # If so, acc += x[N, C_in, H_in, W_in] * w[C_in, C_out, k_h, k_w].
    # This is simple and correct.

    # Let's implement this.
    # Grid: (N * C_out // BLOCK_NC, H_out // BLOCK_H, W_out // BLOCK_W)
    # BLOCK_NC = 8, BLOCK_H = 32, BLOCK_W = 32.
    # Total blocks: 128 * 64 / 8 = 1024. 256 / 32 = 8. 256 / 32 = 8. Total 65536 blocks. Good.
    # Each block processes 8 * 32 * 32 = 8192 elements.
    # Total elements: 128 * 64 * 256 * 256 = 536870912.
    # 536870912 / 8192 = 65536 blocks. Perfect.

    # Offsets
    off_nc = pid_nc * BLOCK_NC + tl.arange(0, BLOCK_NC)
    off_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    off_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # Masks
    mask_nc = off_nc < N * C_out
    mask_h = off_h < H_out
    mask_w = off_w < W_out

    # Extract N, C_out from off_nc
    n_idx = off_nc // C_out
    c_out_idx = off_nc % C_out

    # Initialize accumulator
    acc = tl.zeros([BLOCK_NC, BLOCK_H, BLOCK_W], dtype=tl.float32)

    # Loop over C_in, K_h, K_w
    for c_in_idx in range(C_in):
        for k_h in range(K_h):
            for k_w in range(K_w):
                # Compute H_in, W_in
                h_in = (off_h - k_h + pad_h) // stride_h
                w_in = (off_w - k_w + pad_w) // stride_w

                # Check bounds
                mask_h_in = (h_in >= 0) & (h_in < H_in)
                mask_w_in = (w_in >= 0) & (w_in < W_in)
                mask_valid = mask_nc & mask_h & mask_w & mask_h_in & mask_w_in

                # Load x
                x_idx = n_idx * C_in * H_in * W_in + c_in_idx * H_in * W_in + h_in * W_in + w_in
                x_val = tl.load(x_ptr + x_idx, mask=mask_valid, other=0.0)

                # Load w
                w_idx = c_in_idx * C_out * K_h * K_w + c_out_idx * K_h * K_w + k_h * K_w + k_w
                w_val = tl.load(w_ptr + w_idx, mask=mask_nc, other=0.0)

                # Accumulate
                acc += x_val * w_val

    # Add bias
    b_idx = c_out_idx
    b_val = tl.load(b_ptr + b_idx, mask=mask_nc, other=0.0)
    acc += b_val

    # Store output
    out_idx = off_nc * H_out * W_out + off_h * W_out + off_w
    tl.store(out_ptr + out_idx, acc, mask=mask_nc & mask_h & mask_w)


def triton_conv_transpose(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, 
                          stride_h: int, stride_w: int, pad_h: int, pad_w: int, 
                          out_pad_h: int, out_pad_w: int):
    """
    Custom Triton kernel for 2D transposed convolution.
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_out, _, K_h, K_w = w.shape
    H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h + out_pad_h
    W_out = (W_in - 1) * stride_w - 2 * pad_w + K_w + out_pad_w

    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)

    BLOCK_NC = 8
    BLOCK_H = 32
    BLOCK_W = 32

    grid = lambda meta: (
        (N * C_out + meta["BLOCK_NC"] - 1) // meta["BLOCK_NC"],
        (H_out + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (W_out + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    triton_conv_transpose_kernel[grid](
        x, w, b, out,
        N, C_in, C_out, H_in, W_in, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        BLOCK_NC=BLOCK_NC, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )

    return out


@triton.jit
def triton_mish_add_hardtanh_scale_fused_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    add_value: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused element-wise kernel for Mish, Add, Hardtanh, and Scale.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Mish: x * tanh(softplus(x))
    # Stable softplus: ln(1 + exp(x))
    # For x > 20, softplus(x) ~ x
    # For x < -20, softplus(x) ~ 0
    # For mid range, use ln(1 + exp(x))
    is_large_pos = x > 20.0
    is_large_neg = x < -20.0

    # Compute softplus
    exp_x = tl.exp(x)
    softplus_x = tl.where(
        x >= 0.0,
        x + tl.log1p(tl.exp(-x)),
        tl.log1p(exp_x)
    )

    tanh_sp = tl.tanh(softplus_x)
    mish_x = x * tanh_sp

    # Add
    added_x = mish_x + add_value

    # Hardtanh: clamp between -1 and 1
    hardtanh_x = tl.minimum(tl.maximum(added_x, -1.0), 1.0)

    # Scale
    out = hardtanh_x * scale

    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_mish_add_hardtanh_scale_fused(x: torch.Tensor, add_value: float, scale: float):
    """
    Fuses Mish, Add, Hardtanh, and Scale operations into a single Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 512
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    triton_mish_add_hardtanh_scale_fused_kernel[grid](
        x,
        out,
        n_elements,
        add_value=add_value,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then fuses 
    Mish activation, addition, Hardtanh activation, and scaling into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose2d as a weight holder
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        # 1. Transposed Convolution (Triton)
        x = triton_conv_transpose(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            stride_h=self.conv_transpose.stride[0],
            stride_w=self.conv_transpose.stride[1],
            pad_h=self.conv_transpose.padding[0],
            pad_w=self.conv_transpose.padding[1],
            out_pad_h=self.conv_transpose.output_padding[0],
            out_pad_w=self.conv_transpose.output_padding[1],
        )

        # 2. Fused Post-Conv Operations (Triton)
        x = triton_mish_add_hardtanh_scale_fused(x, self.add_value, self.scale)

        return x