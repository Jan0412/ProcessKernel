import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triton_conv_transpose2d_kernel(
    x_ptr, wt_ptr, out_ptr,
    B, C_in, C_out, H_in, W_in, H_out, W_out,
    stride_y, stride_x, padding_y, padding_x, dilation_y, dilation_x,
    groups,
    BLOCK_OUT_CH: tl.constexpr, BLOCK_SPATIAL: tl.constexpr
):
    # Calculate grid coordinates
    block_idx_x = tl.program_id(0)
    block_idx_y = tl.program_id(1)
    
    # Determine batch index and output channel block index
    num_output_blocks = tl.cdiv(C_out, BLOCK_OUT_CH)
    batch_idx = block_idx_x // num_output_blocks
    out_ch_block = block_idx_x % num_output_blocks
    
    # Calculate output channel offset and input channel offset
    c_out_offset = out_ch_block * BLOCK_OUT_CH
    # For groups, each group has C_out/groups output channels and C_in/groups input channels
    # Mapping: c_in = (c_out // (C_out/groups)) * (C_in/groups) + (c_out % (C_in/groups))
    # Since BLOCK_OUT_CH = 8 and C_out/groups = 16, c_out % 16 < 8, so (c_out // 16) is constant for this block.
    c_in_offset = (c_out_offset // (C_out // groups)) * (C_in // groups)
    
    # Calculate spatial offset
    spatial_offset = block_idx_y * BLOCK_SPATIAL
    
    # Loop over the 8 output channels handled by this block
    for c_idx in range(BLOCK_OUT_CH):
        c_out = c_out_offset + c_idx
        c_in = c_in_offset + c_idx
        
        # Pointers for this output channel (to be stored in registers)
        # out_ptr is [B, C_out, H_out, W_out]
        out_base_ptr = out_ptr + batch_idx * C_out * H_out * W_out + c_out * H_out * W_out
        
        # wt_ptr is [C_out, C_in, H_k, W_k] -> [64, 32, 3, 5]
        # We need to load the kernel slice: wt[c_out, c_in, dy, dx]
        wt_base_ptr = wt_ptr + c_out * C_in * H_in * W_in + c_in * H_in * W_in
        
        # Loop over the 128 spatial positions handled by this block
        for s_idx in range(BLOCK_SPATIAL):
            linear_spatial = spatial_offset + s_idx
            if linear_spatial >= H_out * W_out:
                break
                
            y_out = linear_spatial // W_out
            x_out = linear_spatial % W_out
            
            acc = 0.0
            
            # Loop over kernel dimensions (3x5)
            for dy in range(H_in): # Wait, kernel size is 3x5, but H_in is 128! 
                                   # Correction: H_k = 3, W_k = 5. 
                                   # My bad, I used H_in instead of H_k in the loop. 
                                   # Let's use H_k and W_k passed as constexpr or just hardcode 3 and 5? 
                                   # Better to pass H_k, W_k as args.
                pass 

            # Actually, I'll rewrite the loop correctly below.
            pass

# Let's rewrite the kernel properly with correct kernel dimensions.
@triton.jit
def triton_conv_transpose2d_kernel_fixed(
    x_ptr, wt_ptr, out_ptr,
    B, C_in, C_out, H_in, W_in, H_out, W_out,
    stride_y, stride_x, padding_y, padding_x, dilation_y, dilation_x,
    groups, H_k, W_k,
    BLOCK_OUT_CH: tl.constexpr, BLOCK_SPATIAL: tl.constexpr
):
    block_idx_x = tl.program_id(0)
    block_idx_y = tl.program_id(1)
    
    num_output_blocks = tl.cdiv(C_out, BLOCK_OUT_CH)
    batch_idx = block_idx_x // num_output_blocks
    out_ch_block = block_idx_x % num_output_blocks
    
    c_out_offset = out_ch_block * BLOCK_OUT_CH
    c_in_offset = (c_out_offset // (C_out // groups)) * (C_in // groups)
    
    spatial_offset = block_idx_y * BLOCK_SPATIAL
    
    # Precompute strides for memory access
    x_stride_b = C_in * H_in * W_in
    x_stride_c = H_in * W_in
    x_stride_h = W_in
    # x_stride_w = 1
    
    wt_stride_c_out = C_in * H_k * W_k
    wt_stride_c_in = H_k * W_k
    wt_stride_h = W_k
    # wt_stride_w = 1
    
    out_stride_b = C_out * H_out * W_out
    out_stride_c = H_out * W_out
    out_stride_h = W_out
    # out_stride_w = 1
    
    for c_idx in range(BLOCK_OUT_CH):
        c_out = c_out_offset + c_idx
        c_in = c_in_offset + c_idx
        
        out_base = out_ptr + batch_idx * out_stride_b + c_out * out_stride_c
        wt_base = wt_ptr + c_out * wt_stride_c_out + c_in * wt_stride_c_in
        
        for s_idx in range(BLOCK_SPATIAL):
            linear_spatial = spatial_offset + s_idx
            if linear_spatial >= H_out * W_out:
                break
                
            y_out = linear_spatial // W_out
            x_out = linear_spatial % W_out
            
            acc = 0.0
            
            for dy in range(H_k):
                # Calculate input y coordinate
                y_in = y_out * stride_y - padding_y + dy * dilation_y
                
                # Check y bounds
                if y_in >= 0 and y_in < H_in:
                    for dx in range(W_k):
                        # Calculate input x coordinate
                        x_in = x_out * stride_x - padding_x + dx * dilation_x
                        
                        # Check x bounds
                        if x_in >= 0 and x_in < W_in:
                            # Load weight
                            w = tl.load(wt_base + dy * wt_stride_h + dx)
                            # Load input
                            x_val = tl.load(x_ptr + batch_idx * x_stride_b + c_in * x_stride_c + y_in * x_stride_h + x_in)
                            acc += x_val * w
            
            # Store output
            tl.store(out_base + y_out * out_stride_h + x_out, acc)


def triton_conv_transpose2d(x, weight, stride, padding, dilation, groups):
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    
    B, C_in, H_in, W_in = x.shape
    C_out, _, H_k, W_k = weight.shape
    
    stride_y, stride_x = stride
    padding_y, padding_x = padding
    dilation_y, dilation_x = dilation
    
    # Calculate output dimensions
    H_out = (H_in - 1) * stride_y + 2 * padding_y - 2 * (H_k - 1) * dilation_y + H_k
    W_out = (W_in - 1) * stride_x + 2 * padding_x - 2 * (W_k - 1) * dilation_x + W_k
    
    out = torch.zeros((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    BLOCK_OUT_CH = 8
    BLOCK_SPATIAL = 128
    
    num_output_blocks = (C_out + BLOCK_OUT_CH - 1) // BLOCK_OUT_CH
    num_spatial_blocks = (H_out * W_out + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL
    
    grid = (B * num_output_blocks, num_spatial_blocks)
    
    triton_conv_transpose2d_kernel_fixed[grid](
        x, weight, out,
        B, C_in, C_out, H_in, W_in, H_out, W_out,
        stride_y, stride_x, padding_y, padding_x, dilation_y, dilation_x,
        groups, H_k, W_k,
        BLOCK_OUT_CH, BLOCK_SPATIAL
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 2D transposed convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the original layer to initialize weights, but we will override forward.
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract weight and bias (if any)
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias
        
        # Call our custom Triton kernel
        out = triton_conv_transpose2d(x, weight, self.stride, self.padding, self.dilation, self.groups)
        
        # Add bias if present
        if bias is not None:
            out = out + bias.view(1, -1, 1, 1)
            
        return out