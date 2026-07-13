"""Kernel-body analysis: argument-role recovery."""


from triton_lint import build_model

ROLES = '''
import torch
import triton
import triton.language as tl


@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, v * 2.0, mask=mask)
'''


def test_basic_param_roles():
    model = build_model(ROLES, "test.py")
    kernel = model.kernels["k"]
    assert kernel.params["x_ptr"].loaded and not kernel.params["x_ptr"].stored
    assert kernel.params["out_ptr"].stored and not kernel.params["out_ptr"].loaded
    assert kernel.outputs() == ["out_ptr"]


SCALAR_ON_SPINE = '''
import torch
import triton
import triton.language as tl


@triton.jit
def two_halves_kernel(x_ptr, out_ptr, B, H, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    h = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + b * H + h, mask=h < H)
    # store into the second half of a (B, 2*H) buffer: H is a dim, not a pointer
    tl.store(out_ptr + b * (2 * H) + H + h, v, mask=h < H)
'''


def test_scalar_dim_on_additive_spine_is_not_a_pointer():
    model = build_model(SCALAR_ON_SPINE, "test.py")
    kernel = model.kernels["two_halves_kernel"]
    assert kernel.outputs() == ["out_ptr"]
