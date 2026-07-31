import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    X, W, Y, B,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wk, stride_wl,
    stride_yn, stride_yc, stride_yh, stride_yw,
    N, C_in, C_out, H, W, H_out, W_out, K_h, K_w,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    BLOCK_O: tl.constexpr, BLOCK_KIN: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    # Program indices
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Calculate base offsets for output channels and spatial dimensions
    off_o_base = pid_oc * BLOCK_O
    off_h = pid_h
    off_w = pid_w

    # Create range for output channels handled by this program
    off_o = off_o_base + tl.arange(0, BLOCK_O)
    mask_o = off_o < C_out

    # Initialize accumulator for the output tile (BLOCK_O,)
    acc = tl.zeros((BLOCK_O,), dtype=tl.float32)

    # Iterate over kernel height and width
    for kh in range(K_h):
        for kw in range(K_w):
            # Calculate input coordinates for this kernel offset
            # x_h = h_out * stride_h + kh * dilation_h - padding_h
            x_h = off_h * stride_h + kh * dilation_h - padding_h
            x_w = off_w * stride_w + kw * dilation_w - padding_w

            # Check if the input coordinate is within bounds
            mask_h = (x_h >= 0) & (x_h < H)
            mask_w = (x_w >= 0) & (x_w < W)
            mask_spatial = mask_h & mask_w

            # Load Input Block: X[n, c, x_h, x_w]
            # Shape: (BLOCK_KIN,)
            off_c = tl.arange(0, BLOCK_KIN)
            mask_c = off_c < C_in
            
            # Construct pointers for X
            x_ptrs = X + pid_n * stride_xn + off_c * stride_xc + x_h * stride_xh + x_w * stride_xw
            x_mask = mask_c & mask_spatial
            
            # Load X, masked with 0.0 if out of bounds or padding
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Load Weight Block: W[o, c, kh, kw]
            # Shape: (BLOCK_O, BLOCK_KIN)
            # Construct pointers for W
            w_ptrs = W + off_o[:, None] * stride_wn + off_c[None, :] * stride_wc + kh * stride_wk + kw * stride_wl
            w_mask = mask_o[:, None] & mask_c[None, :]
            
            # Load W
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)

            # Perform Dot Product: acc += sum_c (w[o, c] * x[c])
            # We reshape w to (BLOCK_O, BLOCK_KIN) and x to (1, BLOCK_KIN) for tl.dot
            # tl.dot(w, x.T) results in (BLOCK_O, 1)
            # We sum over the last dimension to get (BLOCK_O,)
            # Or simply use tl.dot and flatten/sum. 
            # tl.dot expects (M, K) and (N, K). Here M=BLOCK_O, K=BLOCK_KIN, N=1.
            # Result is (BLOCK_O, 1).
            
            # To use tl.dot efficiently:
            # x_reshaped should be (1, BLOCK_KIN)
            x_reshaped = x[None, :] 
            # w is (BLOCK_O, BLOCK_KIN)
            
            # acc += tl.sum(w * x_reshaped, axis=1) is also valid but tl.dot is often faster for matmul units
            # However, for N=1, element-wise might be comparable. Let's use tl.dot for generality.
            acc += tl.sum(w * x_reshaped, axis=1)

    # Add Bias if present
    if HAS_BIAS:
        b_ptrs = B + off_o * 1  # B stride is 1 usually
        b_mask = mask_o
        bias = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += bias

    # Store Output: Y[n, o, h, w]
    y_ptrs = Y + pid_n * stride_yn + off_o * stride_yc + off_h * stride_yh + off_w * stride_yw
    tl.store(y_ptrs, acc, mask=mask_o)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                  stride: tuple, padding: tuple, dilation: tuple, groups: int):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert groups == 1, "Only groups=1 is supported for this optimized kernel."
    
    x = x.contiguous()
    weight = weight.contiguous()
    
    N, C_in, H, W = x.shape
    C_out, C_in_w, K_h, K_w = weight.shape
    
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # Calculate output dimensions
    H_out = (H + 2 * padding_h - dilation_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * padding_w - dilation_w * (K_w - 1) - 1) // stride_w + 1
    
    # Prepare output tensor
    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Configuration for Triton blocks
    BLOCK_O = 4   # Number of output channels per program instance
    BLOCK_KIN = 16 # Number of input channels per program instance (K dimension in dot)
    
    # Grid dimensions
    grid = (
        N,
        (C_out + BLOCK_O - 1) // BLOCK_O,
        H_out,
        W_out
    )
    
    # Launch Kernel
    conv2d_kernel[grid](
        x, weight, y, bias,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        N, C_in, C_out, H, W, H_out, W_out, K_h, K_w,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
        BLOCK_O, BLOCK_KIN,
        HAS_BIAS=(bias is not None)
    )
    
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize standard Conv2d to hold weights and biases
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replace PyTorch operator with custom Triton kernel
        return triton_conv2d(
            x, 
            self.conv2d.weight, 
            self.conv2d.bias, 
            self.conv2d.stride, 
            self.conv2d.padding, 
            self.conv2d.dilation, 
            self.conv2d.groups
        )