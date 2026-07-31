import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    Kh, Kw,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Grid coordinates
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Base offsets for the output tile
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W

    # Ranges for the block
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)

    # Create a 2D mesh of offsets for the output tile
    h_offsets = h_offsets[:, None] # BLOCK_H x 1
    w_offsets = w_offsets[None, :] # 1 x BLOCK_W
    
    # Output coordinates
    out_h = h_offsets
    out_w = w_offsets

    # Masks to handle boundaries
    mask_h = out_h < H_out
    mask_w = out_w < W_out
    mask = mask_h & mask_w

    # Calculate input spatial coordinates based on stride and padding
    # in_h = out_h * stride_h - pad_h
    # in_w = out_w * stride_w - pad_w
    in_h_base = out_h * stride_h - pad_h
    in_w_base = out_w * stride_w - pad_w

    # Load Kernel Weights
    # Weight layout: [C, 1, Kh, Kw]
    # We only need weights for the current channel pid_c
    # We can load the entire kernel into registers since it's small
    weight_offsets = tl.arange(0, Kh * Kw)
    weight_mask = weight_offsets < (Kh * Kw)
    
    # Weight ptr offset for current channel
    weight_base_offset = pid_c * (1 * Kh * Kw)
    
    # Load weights into a 2D array for easier indexing
    # Note: Triton tl.load returns a tensor. 
    weights = tl.load(weight_ptr + weight_base_offset + weight_offsets, mask=weight_mask, other=0.0)
    weights = weights.reshape(Kh, Kw)

    # Initialize accumulators for the output tile
    # Shape: BLOCK_H x BLOCK_W
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Iterate over kernel window
    for kh in range(Kh):
        for kw in range(Kw):
            # Input coordinates for this kernel element
            in_h = in_h_base + kh
            in_w = in_w_base + kw
            
            # Mask for input bounds (0 <= in_h < H_in and 0 <= in_w < W_in)
            mask_in_h = (in_h >= 0) & (in_h < H_in)
            mask_in_w = (in_w >= 0) & (in_w < W_in)
            mask_in = mask_in_h & mask_in_w
            
            # Calculate input pointer offset
            # Input layout: [N, C, H_in, W_in]
            # offset = pid_n * (C * H_in * W_in) + pid_c * (H_in * W_in) + in_h * W_in + in_w
            in_offset = pid_n * (C * H_in * W_in) + pid_c * (H_in * W_in) + in_h * W_in + in_w
            
            # Load input values
            x_val = tl.load(x_ptr + in_offset, mask=mask_in, other=0.0)
            
            # Multiply and accumulate
            # weights[kh, kw] is a scalar
            w_val = weights[kh, kw]
            acc = acc + x_val * w_val

    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_c)
        acc = acc + bias_val

    # Store output
    # Output layout: [N, C, H_out, W_out]
    out_offset = pid_n * (C * H_out * W_out) + pid_c * (H_out * W_out) + out_h * W_out + out_w
    tl.store(out_ptr + out_offset, acc, mask=mask)

def triton_depthwise_conv2d(x, weight, bias, stride, padding):
    # x: [N, C, H_in, W_in]
    # weight: [C, 1, Kh, Kw]
    # bias: [C]
    
    N, C, H_in, W_in = x.shape
    Kh, Kw = weight.shape[2], weight.shape[3]
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    H_out = (H_in + 2 * pad_h - Kh) // stride_h + 1
    W_out = (W_in + 2 * pad_w - Kw) // stride_w + 1
    
    out = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    BLOCK_H = 4
    BLOCK_W = 16
    
    grid = (
        N, 
        C, 
        triton.cdiv(H_out, BLOCK_H), 
        triton.cdiv(W_out, BLOCK_W)
    )
    
    depthwise_conv2d_kernel[grid](
        x,
        weight,
        bias,
        out,
        N, C, H_in, W_in, H_out, W_out,
        Kh, Kw,
        stride_h, stride_w,
        pad_h, pad_w,
        BLOCK_H, BLOCK_W
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(ModelNew, self).__init__()
        # Initialize the convolution layer to get the weights initialized as per PyTorch defaults
        # We store the parameters directly to access them in forward
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = None
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.stride = (stride, stride)
        self.padding = (padding, padding)

    def forward(self, x):
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)