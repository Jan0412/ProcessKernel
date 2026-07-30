import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    x_ptr,
    out_ptr,
    stride_x,
    stride_out,
    dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    x_row = x_ptr + pid * stride_x
    out_row = out_ptr + pid * stride_out
    
    # Compute logsumexp
    # Initialize with -inf
    lse = -float('inf')
    
    num_blocks = tl.cdiv(dim, BLOCK_SIZE)
    
    for i in range(num_blocks):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(x_row + offsets, mask=mask, other=-float('inf'))
        
        # Improve numerical stability
        row_max = tl.max(vals)
        shifted = vals - row_max
        sum_exp = tl.sum(tl.exp(shifted))
        chunk_lse = row_max + tl.log(sum_exp)
        
        # Update global lse
        # logsumexp(a, b) = max(a, b) + log(1 + exp(-abs(a-b)))
        if lse == -float('inf'):
            lse = chunk_lse
        else:
            diff = lse - chunk_lse
            lse = tl.maximum(lse, chunk_lse) + tl.log(1.0 + tl.exp(-tl.abs(diff)))
            
    # Compute output
    for i in range(num_blocks):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(x_row + offsets, mask=mask, other=-float('inf'))
        out_vals = vals - lse
        tl.store(out_row + offsets, out_vals, mask=mask)

def triton_log_softmax(x: torch.Tensor, dim: int):
    assert x.is_cuda
    x = x.contiguous()
    batch_size, D = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    
    log_softmax_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out