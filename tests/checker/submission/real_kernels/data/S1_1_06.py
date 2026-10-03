import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    x_stride_outer,
    x_stride_reduction,
    x_stride_inner,
    reduction_dim_size,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate row and col indices for this program
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Check bounds
    if pid_row >= n_rows or pid_col >= n_cols:
        return

    # Calculate base offset for the current reduction sequence
    # offset = row * stride_outer + col * stride_inner
    base_offset = pid_row * x_stride_outer + pid_col * x_stride_inner

    # Initialize min value and its index
    min_val = tl.full([1], float('inf'), dtype=tl.float32)
    min_idx = tl.full([1], -1, dtype=tl.int64)

    # Loop over the reduction dimension in blocks
    # We need to load elements with stride x_stride_reduction
    for start_idx in range(0, reduction_dim_size, BLOCK_SIZE):
        # Create offsets for the current block
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        
        # Check if we are out of bounds for the reduction dimension
        mask = offsets < reduction_dim_size
        
        # Calculate absolute offsets in memory
        # base_offset + offsets * x_stride_reduction
        ptrs = base_offset + offsets * x_stride_reduction
        
        # Load values
        vals = tl.load(x_ptr + ptrs, mask=mask, other=float('inf'))
        
        # Find local min and its offset within the block
        local_min_val = tl.min(vals)
        local_min_offset = tl.argmin(vals)
        
        # Check if this local min is better than the global min found so far
        # We only care about the first element in the block if there are ties? 
        # PyTorch argmin returns the first occurrence.
        # tl.argmin returns the first occurrence in the block.
        # We need to check if local_min_val < min_val.
        
        # To avoid redundant checks, we can just update if local_min_val is smaller
        # or if it's equal and we haven't found a min yet (min_idx == -1).
        # Actually, simpler:
        # if local_min_val < min_val:
        #    min_val = local_min_val
        #    min_idx = start_idx + local_min_offset
        
        # Triton control flow for scalars
        if local_min_val < min_val:
            min_val = local_min_val
            min_idx = start_idx + local_min_offset
            
    # Store the result
    # Output is contiguous for the post-reduction dims? 
    # Output shape is (n_rows, 1, n_cols).
    # The output stride for col is 1.
    out_offset = pid_row * n_cols + pid_col
    tl.store(out_ptr + out_offset, min_idx)

def triton_argmin(x: torch.Tensor, dim: int):
    # Ensure contiguous
    x = x.contiguous()
    
    # Get shapes
    input_shape = x.shape
    reduction_dim_size = input_shape[dim]
    
    # Calculate n_rows and n_cols
    # n_rows: product of dims before dim
    # n_cols: product of dims after dim
    n_rows = 1
    for i in range(dim):
        n_rows *= input_shape[i]
        
    n_cols = 1
    for i in range(dim + 1, len(input_shape)):
        n_cols *= input_shape[i]
        
    # Calculate strides for input
    # Since x is contiguous, we can compute strides easily
    # But let's just use x.stride() for safety and clarity
    x_strides = x.stride()
    
    x_stride_outer = 1
    for i in range(dim):
        x_stride_outer *= x_strides[i]
        
    x_stride_reduction = x_strides[dim]
    
    x_stride_inner = 1
    for i in range(dim + 1, len(input_shape)):
        x_stride_inner *= x_strides[i]
        
    # Prepare output tensor
    # Output shape is input_shape with dim size 1
    output_shape = list(input_shape)
    output_shape[dim] = 1
    out = torch.empty(output_shape, dtype=torch.int64, device=x.device)
    
    # Grid
    grid = (n_rows, n_cols)
    
    # BLOCK_SIZE
    BLOCK_SIZE = 512
    
    # Launch
    argmin_kernel[grid](
        x,
        out,
        x_stride_outer,
        x_stride_reduction,
        x_stride_inner,
        reduction_dim_size,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out