import torch
import triton
import triton.language as tl

@triton.jit
def kl_div_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    batch_size,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Load predictions and targets
    preds = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    tgts = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute KL divergence term: target * (log(target) - log(prediction))
    # Note: PyTorch kl_div(input, target) computes sum(target * (log(target) - input))
    # Here input is log(predictions)
    log_preds = tl.log(preds)
    log_tgts = tl.log(tgts)
    
    # Compute per-element loss
    loss = tgts * (log_tgts - log_preds)
    
    # Sum block and atomic add to global output
    block_sum = tl.sum(loss * mask)
    tl.atomic_add(output_ptr, block_sum)

def triton_kl_div(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size = predictions.shape[0]
    total_elements = predictions.numel()
    
    output = torch.zeros(1, device=predictions.device, dtype=torch.float32)
    
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    kl_div_kernel[(num_programs,)](
        predictions, targets, output, batch_size, total_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item() / batch_size

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)