"""F1.6 passthrough_kernel."""

from __future__ import annotations

from conftest import ELEMENTWISE_KERNEL, src
from helpers import lint_raw


class TestF16PassthroughKernel:
    def test_fires_on_memcpy_kernel(self, check):
        found = check(
            "F1.6",
            src(
                """
@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n)
    tl.store(out_ptr + offs, v, mask=offs < n)

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        copy_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
        )
        assert len(found) == 1

    def test_silent_when_kernel_computes(self, fired):
        assert not fired(
            "F1.6",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out
"""
            ),
        )


# ---------------------------------------------------------------------------
# Regression tests for former linter bugs, now fixed (history in tests/BUGS.md). The 2026-07-13 audit found 3/3 sampled F1.6
# findings false.
# ---------------------------------------------------------------------------

HEADER = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl
'''

CONSTEXPR_DISPATCH = HEADER + '''
@triton.jit
def elementwise_kernel(x_ptr, out_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if OP == 0:
        out = tl.tanh(x)
    elif OP == 1:
        out = 1.0 / (1.0 + tl.exp(-x))
    else:
        out = x  # fallback branch, never selected
    tl.store(out_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        elementwise_kernel[(1,)](x, out, x.numel(), OP=0, BLOCK=1024)
        return out
'''


def test_constexpr_dispatch_kernel_is_not_a_copy():
    assert lint_raw(CONSTEXPR_DISPATCH, "F1.6") == []


GATHER_KERNEL = HEADER + '''
@triton.jit
def downsample_kernel(x_ptr, out_ptr, n_out, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    v = tl.load(x_ptr + offs * 2, mask=mask)
    tl.store(out_ptr + offs, v, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty(x.numel() // 2, device=x.device, dtype=x.dtype)
        downsample_kernel[(1,)](x, out, x.numel() // 2, BLOCK=1024)
        return out
'''

CAT_KERNEL = HEADER + '''
@triton.jit
def cat_kernel(x1_ptr, x2_ptr, out_ptr, n1, n_total, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_total
    src = tl.where(offs < n1, x1_ptr + offs, x2_ptr + (offs - n1))
    v = tl.load(src, mask=mask)
    tl.store(out_ptr + offs, v, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty(x.numel() + y.numel(), device=x.device, dtype=x.dtype)
        cat_kernel[(1,)](x, y, out, x.numel(), x.numel() + y.numel(), BLOCK=1024)
        return out
'''


def test_gather_kernel_is_not_a_decoy():
    assert lint_raw(GATHER_KERNEL, "F1.6") == []


def test_concat_kernel_is_not_a_decoy():
    assert lint_raw(CAT_KERNEL, "F1.6") == []
