@triton.jit
def leaky_relu_kernel(x_ptr, out_ptr, n_elements, negative_slope, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, x * negative_slope)
    tl.store(out_ptr + offsets, out, mask=mask)