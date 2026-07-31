import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def intra_block_scan_kernel(
    x_ptr, out_ptr, block_sums_ptr,
    rows, cols, blocks_per_row,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_offset = row_idx * cols
    
    x = tl.load(x_ptr + row_offset + offsets)
    
    # Inclusive scan within the block
    out = tl.associative_scan(x, axis=0)
    tl.store(out_ptr + row_offset + offsets, out)
    
    # Store block sum
    block_sum = tl.sum(x, axis=0)
    tl.store(block_sums_ptr + row_idx * blocks_per_row + block_idx, block_sum)

@triton.jit
def block_sums_scan_kernel(
    block_sums_ptr, prefix_sums_ptr,
    rows, blocks_per_row,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_sums = tl.load(block_sums_ptr + row_idx * BLOCK_SIZE + offsets)
    
    # Inclusive scan
    inclusive_scan = tl.associative_scan(block_sums, axis=0)
    
    # Exclusive scan = inclusive - current
    exclusive_scan = inclusive_scan - block_sums
    
    tl.store(prefix_sums_ptr + row_idx * BLOCK_SIZE + offsets, exclusive_scan)

@triton.jit
def add_prefix_kernel(
    out_ptr, prefix_sums_ptr,
    rows, cols, blocks_per_row,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    prefix_val = tl.load(prefix_sums_ptr + row_idx * blocks_per_row + block_idx)
    
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_offset = row_idx * cols
    
    out = tl.load(out_ptr + row_offset + offsets)
    out = out + prefix_val
    tl.store(out_ptr + row_offset + offsets, out)

def triton_cumsum(x: torch.Tensor, dim: int):
    # Ensure contiguous
    x = x.contiguous()
    rows = x.shape[0]
    cols = x.shape[1]
    
    BLOCK_SIZE = 1024
    blocks_per_row = (cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    block_sums = torch.empty((rows, blocks_per_row), dtype=x.dtype, device=x.device)
    prefix_sums = torch.empty_like(block_sums)
    
    # Kernel 1
    grid1 = (rows, blocks_per_row)
    intra_block_scan_kernel[grid1](x, out, block_sums, rows, cols, blocks_per_row, BLOCK_SIZE)
    
    # Kernel 2
    grid2 = (rows, 1)
    block_sums_scan_kernel[grid2](block_sums, prefix_sums, rows, blocks_per_row, blocks_per_row)
    
    # Kernel 3
    grid3 = (rows, blocks_per_row)
    add_prefix_kernel[grid3](out, prefix_sums, rows, cols, blocks_per_row, BLOCK_SIZE)
    
    return out