import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute ReLU: max(0, x)
    # In Triton, tl.maximum or conditional logic works.
    # tl.where(x > 0, x, 0.0) is explicit.
    # However, standard math ops are often optimized. 
    # tl.maximum(x, 0.0) is concise.
    out = tl.maximum(x, 0.0)
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_relu(x: torch.Tensor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128 # Or 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return triton_relu(x)