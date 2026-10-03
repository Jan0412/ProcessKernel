import torch
import triton
import triton.language as tl

@triton.jit
def softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Stable softplus: log(1 + exp(x))
    # For large positive x, exp(x) overflows. Use x + log1p(exp(-x))
    # For large negative x, exp(x) underflows to 0, log(1) = 0.
    # We can use a threshold, e.g., 20.0
    exp_x = tl.exp(x)
    out = tl.log(1.0 + exp_x)
    
    # More stable:
    # out = tl.where(x > 20.0, x + tl.exp(-x), tl.log(1.0 + tl.exp(x)))
    # Actually, torch.nn.functional.softplus uses log1p(exp(x)) internally, which is stable.
    # Let's stick to tl.log(1.0 + tl.exp(x)) for simplicity, it's fine for FP32 up to ~88.
    # But to be safe:
    out = tl.where(x > 20.0, x + tl.exp(-x), tl.log(1.0 + tl.exp(x)))
    
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_softplus(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out