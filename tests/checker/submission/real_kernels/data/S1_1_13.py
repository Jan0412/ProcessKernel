import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cumprod_kernel(
    x_ptr,
    out_ptr,
    row_size,
    stride,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row
    pid = tl.program_id(0)
    
    # Base pointers for this row
    row_start = pid * row_size
    x_row = x_ptr + row_start
    out_row = out_ptr + row_start
    
    acc = 1.0
    # Sequential accumulation
    for i in tl.range(row_size):
        val = tl.load(x_row + i)
        acc = acc * val
        tl.store(out_row + i, acc)

def triton_cumprod(x, dim):
    # x shape: (batch, length)
    batch_size, length = x.shape
    
    # Ensure contiguous
    x = x.contiguous()
    out = torch.empty_like(x)
    
    # Grid: one program per row
    grid = (batch_size,)
    
    cumprod_kernel[grid](
        x, out, length, x.stride(0), BLOCK_SIZE=length
    )
    return out