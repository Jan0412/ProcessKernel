import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    H_in, W_in, H_out, W_out, C, K, B,
    stride_h, pad_h, dil_h,
    BLOCK_C: tl.constexpr
):
    # Grid coordinates
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Output spatial coordinates
    h_out = pid_h
    w_out = pid_w
    
    # Iterate over Batch and Channels in blocks
    for bc_idx in range(0, B * C, BLOCK_C):
        # Determine batch and channel indices
        bc_offsets = bc_idx + tl.arange(0, BLOCK_C)
        batch_idx = bc_offsets // C
        channel_idx = bc_offsets % C
        
        # Mask for valid B*C
        mask_bc = bc_offsets < B * C
        
        # Calculate input height range
        # h_in_start = h_out * stride_h - pad_h
        # We need to load K elements: h_in_start + i*dil_h for i in 0..K-1
        
        # Base pointers for this B,C
        # x shape: (B, C, H, W)
        # w shape: (C, 1, K, 1)
        
        # Offset in x for (b, c, 0, w_out)
        x_base_offset = (batch_idx * C + channel_idx) * H_in * W_in + w_out
        
        # Offset in w for (c, 0, 0, 0)
        w_base_offset = channel_idx * K
        
        # Accumulator
        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        
        # Unroll over Kernel Height K
        # Since K is small (e.g., 3), a loop is fine. 
        for i in range(K):
            h_in = h_out * stride_h - pad_h + i * dil_h
            
            # Mask for spatial bounds
            mask_h = (h_in >= 0) & (h_in < H_in)
            
            # Load input
            # x_ptr + x_base_offset + h_in * W_in
            # Note: x_base_offset already has w_out. So we just add h_in * W_in
            x_val = tl.load(x_ptr + x_base_offset + h_in * W_in, mask=mask_h & mask_bc, other=0.0)
            
            # Load weight
            # w_ptr + w_base_offset + i
            w_val = tl.load(w_ptr + w_base_offset + i, mask=mask_bc, other=0.0)
            
            acc += x_val * w_val
            
        # Add bias if present
        if tl.static_ptr(b_ptr):
            bias_val = tl.load(b_ptr + channel_idx, mask=mask_bc, other=0.0)
            acc += bias_val
            
        # Store output
        # out shape: (B, C, H_out, W_out)
        out_offset = (batch_idx * C + channel_idx) * H_out * W_out + h_out * W_out + w_out
        tl.store(out_ptr + out_offset, acc, mask=mask_bc)

def triton_depthwise_conv2d(x, weight, bias, kernel_size, stride, padding, dilation):
    B, C, H_in, W_in = x.shape
    K = kernel_size
    stride_h = stride
    pad_h = padding
    dil_h = dilation
    
    H_out = (H_in + 2 * pad_h - dil_h * (K - 1) - 1) // stride_h + 1
    W_out = W_in # Kernel width is 1, stride 1, padding 0 effectively
    
    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)
    
    BLOCK_C = 64
    grid = (H_out, W_out)
    
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        H_in, W_in, H_out, W_out, C, K, B,
        stride_h, pad_h, dil_h,
        BLOCK_C=BLOCK_C
    )
    return out