"""Kernel sources and shapes shared by the Family-2 check tests."""

SHAPES = [((64, 64), "float32"), ((64, 64), "float32")]
NBYTES = 64 * 64 * 4

#: Big enough that memory time dominates the launch overhead (67 MB per input).
BIG_SHAPES = [((4096, 4096), "float32"), ((4096, 4096), "float32")]

TWO_ELEMENTWISE = '''
@triton.jit
def exp_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.exp(tl.load(x_ptr + offs, mask=offs < n)), mask=offs < n)

@triton.jit
def scale_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) * 2.0, mask=offs < n)
'''

REDUCE_KERNEL = '''
@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + tl.program_id(0), tl.sum(tl.load(x_ptr + offs, mask=offs < n), axis=0))
'''

#: Three launches -- one over the reporting threshold.
THREE_LAUNCHES = """
class ModelNew(nn.Module):
    def forward(self, x):
        n = x.numel()
        a = torch.empty_like(x)
        exp_kernel[(1,)](x, a, n, BLOCK=128)
        b = torch.empty_like(x)
        scale_kernel[(1,)](a, b, n, BLOCK=128)
        out = torch.empty_like(x)
        exp_kernel[(1,)](b, out, n, BLOCK=128)
        return out
"""
