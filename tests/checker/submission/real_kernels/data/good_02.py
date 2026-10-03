import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(
    a_ptr,  # Pointer to predictions
    b_ptr,  # Pointer to targets
    out_ptr,  # Pointer to output (mean hinge loss)
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss: clamp(1 - a * b, min=0)
    ab = a * b
    hinge = 1.0 - ab
    hinge = tl.maximum(hinge, 0.0)
    
    # Store intermediate values for reduction
    tl.store(out_ptr + offsets, hinge, mask=mask)
    
    # Reduction to compute mean
    # Use shared memory for reduction
    shared = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    tl.store(shared + offsets - block_start, hinge, mask=mask)
    tl.sync()
    
    # Reduce within block
    if tl.program_id(0) == 0:
        # Only first block does the final reduction
        total = tl.sum(hinge, axis=0)
        mean = total / n_elements
        tl.store(out_ptr, mean)

def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Triton optimized hinge loss computation.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape."
    
    # Prepare output tensor for intermediate values
    out = torch.empty_like(predictions)
    
    # Number of elements in the tensor
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    hinge_loss_kernel[grid](predictions, targets, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # The kernel stores the mean in the first element of out
    return out[0]

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)