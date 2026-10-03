"""Kernels shared by the F3 tests, shaped like the real generations they came from."""

#: One output element per program: scalar accumulator, scalar loads, no tl.arange.
#: Transcribed from DeepSeek-V4-Flash level 1 problem 69 sample 0.
SCALAR_CONV = '''
@triton.jit
def conv_kernel(x_ptr, w_ptr, out_ptr, C_in, KH, KW,
                x_sn, x_sc, x_sh, x_sw, o_sn, o_sc, o_sh):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    acc = 0.0
    for ci in range(C_in):
        for kh in range(KH):
            for kw in range(KW):
                x_val = tl.load(x_ptr + pid_n * x_sn + ci * x_sc + kh * x_sh + kw * x_sw)
                w_val = tl.load(w_ptr + ci * x_sc + kh * x_sh + kw * x_sw)
                acc += x_val * w_val
    tl.store(out_ptr + pid_n * o_sn + pid_c * o_sc + pid_hw * o_sh, acc)
'''

LAUNCH_CONV = '''
class ModelNew(nn.Module):
    def forward(self, x, w):
        out = torch.empty_like(x)
        conv_kernel[(4, 4, 4)](x, w, out, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1)
        return out
'''

#: The shape of the slower kernel of the paper's problem-46 pair: a blocked kernel whose window
#: loop bound arrives as a runtime argument, so the trip count is unknown at compile time.
RUNTIME_WINDOW = '''
@triton.jit
def pool_kernel(x_ptr, out_ptr, n, K, S, P, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for kd in range(K):
        acc += tl.load(x_ptr + offs * S + kd - P, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        pool_kernel[(4,)](x, out, x.numel(), 3, 2, 1, BLOCK=128)
        return out
'''
