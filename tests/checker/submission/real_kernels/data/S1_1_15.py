import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Using tl.exp is standard.
    # Note: tl.sigmoid might be available, but formula is explicit.
    # For numerical stability, usually sigmoid is implemented carefully, 
    # but for a general kernel, direct formula is fine.
    
    # Optimization: exp(-x) can be expensive. 
    # 1 / (1 + exp(-x)) is the standard definition.
    
    val = tl.exp(-x)
    out = 1.0 / (1.0 + val)
    
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_sigmoid(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out