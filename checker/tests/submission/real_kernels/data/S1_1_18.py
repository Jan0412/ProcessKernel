import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_pool_2d_kernel(
    input_ptr,
    output_ptr,
    H, W,
    H_out, W_out,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE: tl.constexpr,
    N, C
):
    # Program IDs
    pid_spatial = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_batch = tl.program_id(2)

    # Calculate offsets for spatial dimensions
    # Each block handles BLOCK_SIZE elements in the flattened spatial dimension (H_out * W_out)
    offsets = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (H_out * W_out)

    # Decompose flattened offset into h_out, w_out
    # h_out = offsets // W_out
    # w_out = offsets % W_out
    # Note: Division and modulo might be slightly expensive, but acceptable.
    # Alternatively, we can structure the grid differently, but this is fine.
    
    h_out = tl.div_rn(offsets, W_out) # or just //
    w_out = tl.remainder(offsets, W_out) # or just %
    
    # Actually, standard // and % work in Triton.
    h_out = offsets // W_out
    w_out = offsets % W_out

    # Calculate starting position in input
    h_start = h_out * stride - padding
    w_start = w_out * stride - padding

    # Initialize max value
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)

    # Iterate over kernel window
    # We can unroll this or just loop. 4x4 is small.
    for k in range(kernel_size):
        for l in range(kernel_size):
            # Calculate input coordinates
            h_in = h_start + k * dilation
            w_in = w_start + l * dilation

            # Check bounds
            # We need to check if h_in and w_in are within [0, H) and [0, W)
            # Since h_in and w_in are vectors (size BLOCK_SIZE), we need vectorized checks.
            valid_h = (h_in >= 0) & (h_in < H)
            valid_w = (w_in >= 0) & (w_in < W)
            valid = valid_h & valid_w & mask

            # Calculate input index
            # Input shape: (N, C, H, W)
            # idx = pid_batch * (C * H * W) + pid_channel * (H * W) + h_in * W + w_in
            # However, h_in and w_in might be out of bounds, so we should be careful with indexing?
            # Triton loads with mask, so we can compute index freely as long as we mask the load.
            # But computing index with out-of-bounds values might wrap around or be invalid if not masked?
            # Actually, loading with mask=0 prevents access. But we must provide a valid pointer address.
            # If h_in is huge negative, pointer might be invalid?
            # Better to clamp or ensure address is valid?
            # Actually, if mask is false, Triton shouldn't access memory.
            # But the address calculation `input_ptr + offsets` must be valid?
            # No, the pointer arithmetic is done, but the load is conditional.
            # However, if h_in is very negative, the pointer might point to random memory or crash?
            # Usually safe to compute index, but let's clamp h_in and w_in to [0, max_dim-1] for address calculation if needed,
            # or just rely on mask.
            # To be safe, let's just compute index. If mask is false, load is skipped.
            
            # Wait, if mask is false, we don't care about the value loaded.
            # But we must ensure the pointer doesn't segfault.
            # With h_in potentially negative, `h_in * W` is negative.
            # `input_ptr` is a pointer to the start of the batch/channel data.
            # `input_ptr + negative_offset` might be valid memory (previous batch) or invalid.
            # Since we iterate batch/channel via pid, `input_ptr` is the start of the specific (n, c) slice?
            # No, `input_ptr` is the start of the whole tensor.
            # So `pid_batch * ...` adds a large positive offset.
            # If `h_in` is -1, we access previous row. That's fine, it's valid memory (just wrong data, but masked).
            # If `h_in` is -512, we might access previous channel?
            # It's safer to just use the mask.
            
            # Let's refine the base pointer logic.
            # It's better to pass `input_ptr` pointing to the start of the tensor.
            
            base_offset = pid_batch * (C * H * W) + pid_channel * (H * W)
            idx = base_offset + h_in * W + w_in
            
            # Load
            # We use mask=valid.
            val = tl.load(input_ptr + idx, mask=valid, other=-float('inf'))
            
            # Update max
            max_val = tl.maximum(max_val, val)

    # Calculate output index
    # Output shape: (N, C, H_out, W_out)
    out_base_offset = pid_batch * (C * H_out * W_out) + pid_channel * (H_out * W_out)
    out_idx = out_base_offset + offsets
    
    # Store
    tl.store(output_ptr + out_idx, max_val, mask=mask)

def triton_max_pool_2d(x, kernel_size, stride, padding, dilation):
    # x shape: (N, C, H, W)
    N, C, H, W = x.shape
    
    # Calculate output dimensions
    # Formula: floor((W + 2*padding - dilation*(kernel_size - 1) - 1) / stride) + 1
    # Wait, standard PyTorch formula for output size:
    # out = floor((in + 2*padding - dilation*(kernel_size - 1) - 1) / stride) + 1
    # Let's verify with dilation=1.
    # out = floor((in + 2*padding - kernel_size) / stride) + 1
    
    H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Ensure H_out, W_out are at least 1? 
    # PyTorch handles this, but usually inputs are large enough.
    if H_out <= 0 or W_out <= 0:
        return torch.empty((N, C, 0, 0), dtype=x.dtype, device=x.device)

    output = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Grid configuration
    # We want to cover H_out * W_out spatial elements.
    # BLOCK_SIZE = 128
    BLOCK_SIZE = 128
    num_spatial_elements = H_out * W_out
    num_spatial_blocks = (num_spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Grid: (spatial_blocks, channels, batches)
    grid = (num_spatial_blocks, C, N)
    
    max_pool_2d_kernel[grid](
        x,
        output,
        H, W,
        H_out, W_out,
        kernel_size,
        stride,
        padding,
        dilation,
        BLOCK_SIZE,
        N, C
    )
    return output