import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_mean_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Initialize thread-local accumulator
    local_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    pred = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss: max(0, 1 - pred * target)
    hinge_loss = tl.maximum(0.0, 1.0 - pred * target)
    
    # Store in local accumulator
    local_sum = hinge_loss
    
    # Reduce within block using shared memory
    # Use a simple reduction approach for demonstration
    # In practice, you'd want more sophisticated reduction
    block_sum = tl.sum(local_sum, axis=0)
    
    # Store partial sum for this block
    tl.store(output_ptr + tl.program_id(0), block_sum, mask=tl.program_id(0) < (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)

@triton.jit
def reduce_and_mean_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    num_blocks,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    # Load partial sums
    partial_sum = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce all partial sums to single value
    if tl.program_id(0) == 0:
        total_sum = tl.sum(partial_sum, axis=0)
        mean_result = total_sum / n_elements
        tl.store(output_ptr, mean_result)

def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Number of elements in the tensor
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # First kernel: compute hinge loss and partial sums
    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device='cuda')
    
    # Launch the Triton kernel for hinge loss computation and partial sum
    grid_hinge = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hinge_loss_mean_kernel[grid_hinge](predictions, targets, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second kernel: reduce partial sums and compute mean
    final_output = torch.empty(1, dtype=torch.float32, device='cuda')
    grid_reduce = lambda meta: (1,)
    reduce_and_mean_kernel[grid_reduce](partial_sums, final_output, n_elements, num_blocks, BLOCK_SIZE=BLOCK_SIZE)
    
    return final_output[0]

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)