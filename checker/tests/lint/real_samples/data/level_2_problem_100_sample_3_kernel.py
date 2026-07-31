@triton.jit
def clamp_div_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    min_val,
    div_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Clamp
    x = tl.maximum(x, min_val)
    
    # Divide
    x = x / div_val
    
    tl.store(out_ptr + offsets, x, mask=mask)