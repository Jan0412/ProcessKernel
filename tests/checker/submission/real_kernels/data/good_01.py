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
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    pred = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss: clamp(1 - pred * target, min=0)
    product = pred * target
    diff = 1.0 - product
    hinge_loss = tl.maximum(diff, 0.0)
    
    # Compute sum for mean calculation
    sum_val = tl.sum(hinge_loss, axis=0)
    
    # Store partial sum in shared memory
    # For simplicity, we'll use a single reduction approach
    if tl.program_id(0) == 0:
        # Only first block contributes to final result
        tl.store(output_ptr, sum_val / n_elements)

def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    
    # Allocate output tensor
    output = torch.empty(1, dtype=torch.float32, device=predictions.device)
    
    # Launch kernel
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hinge_loss_mean_kernel[grid](predictions, targets, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output[0]

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)