import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_transpose2d_kernel(
    X, W, Y, Bias,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc_out, stride_wc_in, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    stride_bc,
    H_out, W_out,
    H, W, K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    C_in, C_out,
    BLOCK_C_OUT: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    # Program ID
    pid_n = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(2)

    # Base pointers for this batch
    X_base = X + pid_n * stride_xn
    Y_base = Y + pid_n * stride_yn

    # Output offsets
    y_offsets = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
    x_offsets = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)

    # Masks for spatial bounds
    y_mask = y_offsets < H_out
    x_mask = x_offsets < W_out

    # Initialize output accumulator in shared memory
    # Shape: [BLOCK_C_OUT, BLOCK_H, BLOCK_W]
    acc = tl.zeros((BLOCK_C_OUT, BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Iterate over kernel spatial dimensions
    for ky in range(K_h):
        for kx in range(K_w):
            # Calculate input spatial coordinates
            # y_in = y_out * stride_h - ky + pad_h
            # x_in = x_out * stride_w - kx + pad_w
            y_in_offsets = y_offsets * stride_h - ky + pad_h
            x_in_offsets = x_offsets * stride_w - kx + pad_w
            
            # Masks for input bounds
            y_in_mask = (y_in_offsets >= 0) & (y_in_offsets < H)
            x_in_mask = (x_in_offsets >= 0) & (x_in_offsets < W)
            combined_mask = y_mask & x_mask & y_in_mask & x_in_mask
            
            # Load input tile
            # Shape: [BLOCK_H, BLOCK_W]
            # Input layout: [N, C_in, H, W]
            # We need to load [C_in, BLOCK_H, BLOCK_W] but we loop C_in inside
            # Actually, let's load the whole spatial slice for all C_in? 
            # No, C_in is 64. Let's loop C_in inside to keep register pressure low.
            
            # We will accumulate across C_in inside this loop
            c_in_offsets = tl.arange(0, C_in)
            
            # We want to do: acc[c_out, y, x] += sum_c_in (x[n, c_in, y_in, x_in] * w[c_out, c_in, ky, kx])
            
            # To optimize, we can load the whole X slice for this (ky, kx) if C_in is small enough
            # C_in=64, B=8, B=8 -> 4096 floats. This fits in registers/shared memory comfortably.
            # Let's load X as [C_in, BLOCK_H, BLOCK_W]
            
            # Construct pointer for X: [C_in, BLOCK_H, BLOCK_W]
            # X_base + c_in[:, None, None] * stride_xc + y_in[None, :, None] * stride_xh + x_in[None, None, :] * stride_xw
            x_ptr = X_base + c_in_offsets[:, None, None] * stride_xc + y_in_offsets[None, :, None] * stride_xh + x_in_offsets[None, None, :] * stride_xw
            x_mask_3d = combined_mask[None, :, :] # Broadcast C_in
            
            x_tile = tl.load(x_ptr, mask=x_mask_3d, other=0.0) # Shape [C_in, BLOCK_H, BLOCK_W]
            
            # Load weight tile for this (ky, kx)
            # W layout: [C_out, C_in, K_h, K_w]
            # We want [BLOCK_C_OUT, C_in]
            c_out_offsets = tl.arange(0, BLOCK_C_OUT)
            c_out_mask = c_out_offsets < C_out
            
            w_ptr = W + c_out_offsets[:, None] * stride_wc_out + c_in_offsets[None, :] * stride_wc_in + ky * stride_wkh + kx * stride_wkw
            w_tile = tl.load(w_ptr, mask=c_out_mask[:, None] & (c_in_offsets[None, :] < C_in), other=0.0) # Shape [BLOCK_C_OUT, C_in]
            
            # Accumulate using manual loop or dot product
            # Triton tl.dot is 2D. We need 3D.
            # acc[B, H, W] += sum_C (w[B, C] * x[C, H, W])
            # We can loop over C_in
            for c_in_idx in range(C_in):
                w_col = tl.load(w_ptr + c_in_idx * stride_wc_in) # Shape [BLOCK_C_OUT]
                x_row = tl.load(x_ptr + c_in_idx * stride_xc) # Shape [BLOCK_H, BLOCK_W]
                
                # Update accumulator
                acc += w_col[:, None, None] * x_row[None, :, :]

    # Add bias if present
    if HAS_BIAS:
        c_out_offsets = tl.arange(0, BLOCK_C_OUT)
        bias_vals = tl.load(Bias + c_out_offsets, mask=c_out_offsets < C_out, other=0.0)
        acc += bias_vals[:, None, None]

    # Store accumulator to global memory
    c_out_offsets = tl.arange(0, BLOCK_C_OUT)
    y_out_offsets = y_offsets
    x_out_offsets = x_offsets
    
    out_ptr = Y_base + c_out_offsets[:, None, None] * stride_yc + y_out_offsets[None, :, None] * stride_yh + x_out_offsets[None, None, :] * stride_yw
    
    tl.store(
        out_ptr,
        acc,
        mask=combined_mask[None, :, :]
    )


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1
):
    """
    Custom Triton kernel for ConvTranspose2d.
    Assumes:
    - Input x: [N, C_in, H, W]
    - Weight W: [C_out, C_in, K_h, K_w] (PyTorch layout)
    - Strides are 1D (stride_h == stride_w)
    - Padding is 1D (pad_h == pad_w)
    - Groups = 1
    """
    assert x.is_cuda and weight.is_cuda
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert groups == 1

    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    stride_h = stride
    stride_w = stride
    pad_h = padding
    pad_w = padding

    # Calculate output shape
    H_out = (H - 1) * stride_h - 2 * pad_h + K_h + output_padding
    W_out = (W - 1) * stride_w - 2 * pad_w + K_w + output_padding

    # Allocate output
    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Block sizes
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_C_OUT = 64

    # Grid
    grid = (
        N,
        triton.cdiv(H_out, BLOCK_H),
        triton.cdiv(W_out, BLOCK_W)
    )

    # Launch kernel
    _conv_transpose2d_kernel[grid](
        x, weight, y, bias,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        bias.stride(0),
        H_out, W_out,
        H, W, K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        C_in, C_out,
        BLOCK_C_OUT, BLOCK_H, BLOCK_W,
        bias is not None
    )

    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get weight tensor
        weight = self.conv_transpose2d.weight
        
        # Get bias tensor if present
        bias_tensor = self.conv_transpose2d.bias if self.bias else None
        
        # Call custom Triton kernel
        out = triton_conv_transpose2d(
            x,
            weight,
            bias_tensor,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )
        
        return out