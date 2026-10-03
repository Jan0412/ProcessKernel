import torch
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(x_ptr, out_ptr, dim, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Compute row offset
    row_offset = pid * dim
    
    # Phase 1: Compute sum of squares
    sum_sq = 0.0
    for i in range(0, dim, BLOCK_SIZE):
        offsets = row_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_offset + dim
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)
    
    # Compute inverse norm
    norm = tl.sqrt(sum_sq)
    inv_norm = tl.where(norm > 0, 1.0 / norm, 1.0)
    
    # Phase 2: Normalize and store
    for i in range(0, dim, BLOCK_SIZE):
        offsets = row_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_offset + dim
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x * inv_norm, mask=mask)

def triton_l2_norm(x: torch.Tensor):
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, dim = x.shape
    
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    l2_norm_kernel[grid](x, out, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)