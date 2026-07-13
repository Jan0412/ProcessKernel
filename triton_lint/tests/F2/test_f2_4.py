"""F2.4 zeroed_overwritten_buffer."""

from __future__ import annotations

from conftest import src
from helpers import lint, lint_raw

from ._fixtures import NBYTES, SHAPES, TWO_ELEMENTWISE


class TestF24ZeroedOverwrittenBuffer:
    def test_fires_when_fully_overwritten(self, check):
        found = check(
            "F2.4",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.zeros_like(x)
        exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert found[0].data["bytes"] == NBYTES
        assert "empty_like" in found[0].message

    def test_silent_when_kernel_accumulates_with_atomics(self, fired):
        """zeros IS required for an atomic accumulator -- 'fixing' it would be a bug."""
        assert not fired(
            "F2.4",
            src(
                """
@triton.jit
def hist_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.atomic_add(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n))

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.zeros_like(x)
        hist_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )

    def test_silent_for_empty_like(self, fired):
        assert not fired(
            "F2.4",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )


# ---------------------------------------------------------------------------
# Regression tests for former linter bugs, now fixed (history in tests/BUGS.md).
# ---------------------------------------------------------------------------

DIAGONAL_STORE = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    v = tl.load(x_ptr + i, mask=mask)
    # writes ONLY the diagonal of the (n, n) output
    tl.store(out_ptr + i * n + i, v, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x):
        n = x.numel()
        out = torch.zeros((n, n), device=x.device, dtype=x.dtype)
        diag_kernel[(1,)](x, out, n, BLOCK=1024)
        return out
'''


def test_partial_store_keeps_its_zero_init():
    assert lint_raw(DIAGONAL_STORE, "F2.4") == []


BRANCHED_ACCUMULATOR = '''
class ModelNew(nn.Module):
    def forward(self, x):
        if x.dim() == 3:
            out = torch.empty_like(x)
            work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
            return out
        out = torch.zeros_like(x)
        for _ in range(3):
            out += x
        return out / 3.0
'''


def test_host_accumulated_zeros_in_other_branch_is_kept():
    assert lint(BRANCHED_ACCUMULATOR, "F2.4") == []


#: One kernel with two outputs: `hist` is a real atomic accumulator (its zero-init is
#: required) and `out` is unconditionally overwritten by a plain tl.store (its
#: zero-init is wasted). `out` and `hist` are independent parameters.
SCATTER_PLUS_DENSE = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scatter_dense_kernel(x_ptr, hist_ptr, out_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < n
    v = tl.load(x_ptr + o, mask=m)
    tl.atomic_add(hist_ptr + o, v, mask=m)   # hist: accumulator, zeros required
    tl.store(out_ptr + o, v * 2.0, mask=m)   # out: full overwrite, zeros wasted

class ModelNew(nn.Module):
    def forward(self, x):
        n = x.numel()
        hist = torch.zeros_like(x)
        out = torch.zeros_like(x)
        scatter_dense_kernel[(1,)](x, hist, out, n, BLOCK=128)
        return out, hist
'''


def test_atomic_accumulator_buffer_itself_is_not_flagged():
    """Control for BUG-20: `hist` really is an accumulator (the kernel loads it back
    via the atomic), so F2.4 must stay silent on it -- and does."""
    fs = lint_raw(SCATTER_PLUS_DENSE, "F2.4")
    assert all(f.data["buffer"] != "hist" for f in fs)


def test_wasted_zeros_still_flagged_when_a_sibling_param_is_atomic():
    fs = lint_raw(SCATTER_PLUS_DENSE, "F2.4")
    assert any(f.data["buffer"] == "out" for f in fs)
