import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmse_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared difference
    diff = x - y
    squared_diff = diff * diff
    
    # Initialize accumulator for reduction
    acc = tl.zeros((1,), dtype=tl.float32)
    acc += squared_diff
    
    # Reduction across the block
    for i in range(1, BLOCK_SIZE):
        acc += tl.where(offsets + i < n_elements, tl.load(x_ptr + offsets + i, mask=mask, other=0.0) - tl.load(y_ptr + offsets + i, mask=mask, other=0.0), 0.0)
    
    # Actually, let's do a proper reduction - we need to accumulate in the kernel
    # But the above approach is not correct. Let's use the standard approach:
    # Since we're doing per-block accumulation, we'll use tl.sum for the block
    
    # Compute for this block
    block_sum = tl.sum(squared_diff, axis=0)
    
    # Store partial sums
    tl.store(out_ptr + tl.program_id(0), block_sum)


@triton.jit
def final_reduction_kernel(
    partial_sums_ptr,
    total_ptr,
    n_partial_sums,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partial_sums
    
    # Load partial sums
    partial_sum = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    
    # Sum all partial sums
    total = tl.sum(partial_sum)
    
    # Store result
    tl.store(total_ptr, total)


def triton_rmse(x: torch.Tensor, y: torch.Tensor):
    """
    Compute RMSE using Triton kernels for better performance.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()
    
    # Check shapes match
    assert x.shape == y.shape, "Input tensors must have the same shape."
    
    n_elements = x.numel()
    
    # For simplicity, we'll use a two-phase approach:
    # Phase 1: Compute squared differences and partial sums
    BLOCK_SIZE = 256
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if num_blocks == 0:
        num_blocks = 1
    
    # Create tensor to hold partial sums
    partial_sums = torch.zeros(num_blocks, device=x.device, dtype=torch.float32)
    
    # Launch kernel to compute partial squared sums
    rmse_kernel[(num_blocks,)](
        x, y, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Reduce partial sums to final sum
    # This is simplified - in practice, we might need multiple reduction steps
    # For simplicity, if we have many blocks, we could do another kernel call
    # But for now, let's just use torch.sum for the final reduction
    total_sum = torch.sum(partial_sums[:num_blocks])
    
    # Compute mean and take sqrt
    mean_squared = total_sum / n_elements
    rmse = torch.sqrt(mean_squared)
    
    return rmse


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, y):
        return triton_rmse(x, y)


# Note: The above implementation has an issue - the first kernel doesn't properly
# accumulate per-block sums. Let me fix this with a better approach using Triton's
# reduce functionality.

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def squared_diff_sum_kernel(
    x_ptr,
    y_ptr,
    result_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    # Each program handles a block of input
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared difference and sum
    diff = x - y
    squared_diff = diff * diff
    block_sum = tl.sum(squared_diff, axis=0)
    
    # Store block sum
    tl.store(result_ptr + block_id, block_sum)


def triton_rmse_optimized(x: torch.Tensor, y: torch.Tensor):
    """
    Optimized RMSE using Triton kernels with proper reduction.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()
    assert x.shape == y.shape, "Input tensors must have the same shape."
    
    n_elements = x.numel()
    if n_elements == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    # Use a reasonable block size
    BLOCK_SIZE = 256
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor for partial sums
    # We'll use a 2D approach to handle the reduction more efficiently
    # But for simplicity, let's do a single kernel that computes the full sum
    # using a tree-reduction pattern in the kernel
    
    # For now, use a simpler approach: single block reduction if n_elements <= BLOCK_SIZE
    if n_elements <= BLOCK_SIZE:
        # Direct kernel call with single block
        result = torch.empty(1, device=x.device, dtype=torch.float32)
        squared_diff_sum_kernel[(1,)](
            x, y, result,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_BLOCKS=1
        )
        total_sum = result[0]
    else:
        # Multiple blocks - compute partial sums
        partial_sums = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
        squared_diff_sum_kernel[(num_blocks,)](
            x, y, partial_sums,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_BLOCKS=num_blocks
        )
        # Use PyTorch for final reduction (it's highly optimized)
        total_sum = torch.sum(partial_sums)
    
    # Compute mean and sqrt
    mean_squared = total_sum / n_elements
    rmse = torch.sqrt(mean_squared)
    
    return rmse


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, y):
        return triton_rmse_optimized(x, y)