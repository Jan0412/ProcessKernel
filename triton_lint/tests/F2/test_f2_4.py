"""F2.4 zeroed_overwritten_buffer."""

from __future__ import annotations

import pytest

from conftest import src
from helpers import lint, lint_raw

from triton_lint import build_model
from triton_lint.checks.family2 import f2_4_zeroed_overwritten_buffer

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


# ---------------------------------------------------------------------------
# BUG-32 -- open. See tests/BUGS.md.
# ---------------------------------------------------------------------------

_BUG32_REASON = (
    "BUG-32: F2.4 assumes `stores to it, no atomic, no load, no structural partial_store "
    "=> every element is overwritten`. But a scatter / pad / one-hot / upsample kernel "
    "grids over a SUBSET of the output (or over the input, writing into a larger padded "
    "output) and leaves the rest at its zero value, which is the semantically-required "
    "default. _is_partial_coverage (the BUG-8 guard) only catches a single index name in "
    ">=2 additive terms (diagonal) or a literal-int stride -- it does not catch a store "
    "whose index space is smaller than the buffer (a data-dependent scatter, or a "
    "padded/relocated write). F2.4 tells the model to use `torch.empty_like`, which "
    "leaves the untouched region uninitialised -- a correctness bug (garbage padding, a "
    "one-hot vector full of garbage). 25 of the 76 F2.4-flagged files have a "
    "pad/scatter/shift/upsample writer (clear FPs); ~46/76 write a relocated/subset "
    "output. Real samples: p1229_s4 (one_hot), p15104_s6 (pad_same), p17549_s4 (avg_pool_pad)."
)

# A one-hot kernel: one 1.0 per row at a data-dependent column, every other column
# stays at its zero value. The zeros ARE the one-hot encoding.
_ONE_HOT = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def one_hot_kernel(idx_ptr, out_ptr, num_classes, n_rows, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    mask = row < n_rows
    cls = tl.load(idx_ptr + row, mask=mask)
    tl.store(out_ptr + row * num_classes + cls, 1.0, mask=mask)

class ModelNew(nn.Module):
    def forward(self, idx):
        n = idx.shape[0]
        out = torch.zeros((n, 10), device=idx.device, dtype=torch.float32)
        one_hot_kernel[(n,)](idx, out, 10, n, BLOCK=1)
        return out
'''

# A padding kernel: grids over the INPUT, scatters each element into a LARGER output at
# an offset; the border rows/cols are never written and must stay zero.
_PAD_INTO_LARGER = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def pad_kernel(inp_ptr, out_ptr, W, out_W, pad, n_in, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_in
    x = tl.load(inp_ptr + offs, mask=mask)
    h = offs // W
    w = offs % W
    out_idx = (h + pad) * out_W + (w + pad)
    tl.store(out_ptr + out_idx, x, mask=mask)

class ModelNew(nn.Module):
    def forward(self, x):
        H, W = x.shape
        out = torch.zeros((H + 2, W + 2), device=x.device, dtype=x.dtype)
        pad_kernel[(1,)](x, out, W, W + 2, 1, x.numel(), BLOCK=128)
        return out
'''


def test_one_hot_scatter_needs_its_zero_init():
    assert lint_raw(_ONE_HOT, "F2.4") == []


def test_pad_into_larger_output_needs_its_zero_init():
    assert lint_raw(_PAD_INTO_LARGER, "F2.4") == []


def test_full_overwrite_zeros_are_still_flagged():
    # Passing control: the fix must not blanket-suppress every scatter -- a kernel that
    # writes every element of the zeroed buffer is a genuine wasted memset and must keep
    # firing. Pins the bug to the subset-coverage write, not to `torch.zeros` itself.
    source = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def full_kernel(inp_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(inp_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.zeros_like(x)
        full_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
'''
    assert [f.data["buffer"] for f in lint_raw(source, "F2.4")] == ["out"]


# A data-dependent scatter: `tl.store(out_ptr + idx, v)` with `idx` loaded from a tensor,
# grids over the input, and writes into a larger zeros output. The index SPACE (one write
# per input element) is smaller than the buffer, so the unwritten slots must keep their
# zeros -- but the offset `{idx}` has no repeated index and no literal stride, so
# _is_partial_coverage misses it.
_DATA_DEPENDENT_SCATTER = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scatter_kernel(idx_ptr, val_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask)
    v = tl.load(val_ptr + offs, mask=mask)
    tl.store(out_ptr + idx, v, mask=mask)

class ModelNew(nn.Module):
    def forward(self, idx, val):
        out = torch.zeros(100, device=val.device, dtype=val.dtype)
        scatter_kernel[(1,)](idx, val, out, idx.numel(), BLOCK=128)
        return out
'''


def test_data_dependent_scatter_needs_its_zero_init():
    assert lint_raw(_DATA_DEPENDENT_SCATTER, "F2.4") == []


def test_stride_2_scatter_is_already_silent():
    # Passing boundary control: a stride-2 upsample scatter (`out_ptr + offs * 2`) is a
    # SUBSET write too, but the literal-stride tell in _is_partial_coverage (the BUG-8
    # guard) already catches it, so F2.4 is correctly silent. This pins BUG-32 to the
    # subset writes that carry *no* structural stride/repeat tell -- the gap the fix must
    # close without disturbing the strided case already handled.
    source = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def upsample_kernel(inp_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(inp_ptr + offs, mask=mask)
    tl.store(out_ptr + offs * 2, v, mask=mask)

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.zeros(x.numel() * 2, device=x.device, dtype=x.dtype)
        upsample_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
'''
    assert lint_raw(source, "F2.4") == []


# ---------------------------------------------------------------------------
# BUG-36 -- open. See tests/BUGS.md.
# ---------------------------------------------------------------------------

_BUG36_REASON = (
    "BUG-36: a triangular matmul addresses the WHOLE (N, N) output "
    "(`C + off_m * N + off_n`, both indices spanning the full range) but carries a "
    "data-dependent store mask `off_m >= off_n` (or `col >= row`) that writes only one "
    "triangle -- the other triangle must stay at its `torch.zeros` value. This is a "
    "different gap from BUG-32: the store's index SPACE is not smaller than the buffer, "
    "so `_is_partial_coverage` (address-only) sees full coverage, and F2.4's docstring "
    "even asserts a mask never disqualifies a finding -- true for a bounds mask "
    "`offs < n`, false for a triangular/conditional one. F2.4 calls the `torch.zeros` a "
    "wasted memset and prescribes `torch.empty_like`, which leaves the un-written "
    "triangle uninitialised -- a correctness bug. 7 triangular-matmul FPs across the four "
    "Qwen3.6 runs (p14_s9/p15_s1/p15_s5/p15_s7/p15_s8 level1, p14_s7/p15_s8 level1_lintloop)."
)

# A lower-triangular matmul: every block runs and the store address covers the full
# matrix; only the triangular mask `off_m >= off_n` restricts writes, so the strictly
# upper triangle keeps its zero-init value (that IS the tril result).
_TRIANGULAR_MASKED_STORE = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tril_matmul_kernel(A, B, C, N, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    off_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, N, BLOCK):
        off_k = k + tl.arange(0, BLOCK)
        a = tl.load(A + off_m[:, None] * N + off_k[None, :])
        b = tl.load(B + off_k[:, None] * N + off_n[None, :])
        acc += tl.dot(a, b)
    mask = off_m[:, None] >= off_n[None, :]      # strictly-upper triangle stays zero
    tl.store(C + off_m[:, None] * N + off_n[None, :], acc, mask=mask)

class ModelNew(nn.Module):
    def forward(self, A, B):
        N = A.shape[0]
        C = torch.zeros((N, N), device=A.device, dtype=torch.float32)
        grid = (triton.cdiv(N, 16), triton.cdiv(N, 16))
        tril_matmul_kernel[grid](A, B, C, N, BLOCK=16)
        return C
'''


def test_triangular_masked_store_needs_its_zero_init():
    assert lint_raw(_TRIANGULAR_MASKED_STORE, "F2.4") == []


def test_unmasked_full_matmul_store_is_still_flagged():
    # Passing control: the SAME kernel shape with the triangular mask removed writes
    # every element, so the `torch.zeros` really is a wasted memset and F2.4 must keep
    # firing. Pins BUG-36 to the store mask, not to the matmul shape.
    unmasked = _TRIANGULAR_MASKED_STORE.replace(
        "    mask = off_m[:, None] >= off_n[None, :]      # strictly-upper triangle stays zero\n"
        "    tl.store(C + off_m[:, None] * N + off_n[None, :], acc, mask=mask)",
        "    tl.store(C + off_m[:, None] * N + off_n[None, :], acc)",
    )
    assert [f.data["buffer"] for f in lint_raw(unmasked, "F2.4")] == ["C"]


# The block-skip spelling: no store mask at all -- every launched block writes its full
# tile -- but strictly-upper blocks return early (`if pid_m < pid_n: return`), so those
# tiles keep their zeros. BUGS.md names this the "and/or an `if pid_m < pid_n: return`
# block skip" face. The control-flow skip is invisible to the address-only coverage
# analysis, so partial_store stays unset exactly as with the mask.
_TRIANGULAR_BLOCK_SKIP = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tril_matmul_kernel(A, B, C, N, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m < pid_n:          # strictly-upper blocks never write -> keep their zeros
        return
    off_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    off_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, N, BLOCK):
        off_k = k + tl.arange(0, BLOCK)
        a = tl.load(A + off_m[:, None] * N + off_k[None, :])
        b = tl.load(B + off_k[:, None] * N + off_n[None, :])
        acc += tl.dot(a, b)
    tl.store(C + off_m[:, None] * N + off_n[None, :], acc)

class ModelNew(nn.Module):
    def forward(self, A, B):
        N = A.shape[0]
        C = torch.zeros((N, N), device=A.device, dtype=torch.float32)
        grid = (triton.cdiv(N, 16), triton.cdiv(N, 16))
        tril_matmul_kernel[grid](A, B, C, N, BLOCK=16)
        return C
'''


def test_triangular_block_skip_store_needs_its_zero_init():
    assert lint_raw(_TRIANGULAR_BLOCK_SKIP, "F2.4") == []


def test_upper_triangular_masked_store_needs_its_zero_init():
    # The upper-triangular mirror (`off_m <= off_n`) leaves the strictly-lower triangle at
    # its zeros. Same full-address / conditional-mask shape, opposite triangle.
    upper = _TRIANGULAR_MASKED_STORE.replace(
        "    mask = off_m[:, None] >= off_n[None, :]      # strictly-upper triangle stays zero",
        "    mask = off_m[:, None] <= off_n[None, :]      # strictly-lower triangle stays zero",
    )
    assert lint_raw(upper, "F2.4") == []


# ---------------------------------------------------------------------------
# Only a reachable, resolvable, non-accumulating writer makes a zero-fill wasteful.
# ---------------------------------------------------------------------------


def test_zeroed_buffer_stored_only_by_unreachable_launch_is_skipped(check):
    """F2.4 scans every zero-alloc buffer, but only a *reachable* launch can be its
    writer. A zeros buffer written solely from dead code (a helper the entry never
    calls) has no reachable writer, so it is left alone -- exercising the `launch is
    None` branch of the writer loop.
    """
    found = check(
        "F2.4",
        src(
            TWO_ELEMENTWISE
            + """
def unused(x):
    out = torch.zeros_like(x)
    scale_kernel[(1,)](x, out, x.numel(), BLOCK=128)
    return out

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
""",
        ),
        SHAPES,
    )
    assert found == []


def test_writer_kernel_absent_from_the_model_is_skipped():
    """The writer launch's kernel is resolved through `model.kernels`; a reachable
    launch always has one after a normal build. The `kernel is None` guard covers a
    broken mapping -- exercise it directly.
    """
    model = build_model(
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
        "<t>",
        SHAPES,
    )
    assert len(f2_4_zeroed_overwritten_buffer.check(model)) == 1
    del model.kernels["exp_kernel"]
    assert f2_4_zeroed_overwritten_buffer.check(model) == []


def test_atomic_writer_role_suppresses_the_finding():
    """The zero-init of an atomic-accumulator output is *required*, so the check must
    stay silent when the bound parameter is atomic. A real atomic also marks the pointer
    `loaded`, which the earlier `loaded_by` guard already catches; forcing only the
    `atomic` role on an otherwise-plain writer isolates the `role.atomic` branch of the
    accumulation test.
    """
    model = build_model(
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
        "<t>",
        SHAPES,
    )
    assert len(f2_4_zeroed_overwritten_buffer.check(model)) == 1  # a wasted zero-fill...
    model.kernels["exp_kernel"].params["out_ptr"].atomic = True
    assert f2_4_zeroed_overwritten_buffer.check(model) == []  # ...until it accumulates


def test_computed_data_dependent_scatter_needs_its_zero_init():
    # BUG-32 class coverage: the scatter index is *computed* from a load (`pos = base + idx`,
    # idx loaded) rather than being the bare load -- the loaded-taint must survive arithmetic.
    source = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scatter_kernel(idx_ptr, val_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask)
    v = tl.load(val_ptr + offs, mask=mask)
    pos = idx * 2 + 1
    tl.store(out_ptr + pos, v, mask=mask)

class ModelNew(nn.Module):
    def forward(self, idx, val):
        out = torch.zeros(100, device=val.device, dtype=val.dtype)
        scatter_kernel[(1,)](idx, val, out, idx.numel(), BLOCK=128)
        return out
'''
    assert lint_raw(source, "F2.4") == []


def test_strict_lower_triangular_mask_needs_its_zero_init():
    # BUG-36 class coverage: the strict `off_m > off_n` mask (excluding the diagonal) is the
    # same conditional-mask shape as `>=`, opposite inclusivity.
    strict = _TRIANGULAR_MASKED_STORE.replace(
        "    mask = off_m[:, None] >= off_n[None, :]      # strictly-upper triangle stays zero",
        "    mask = off_m[:, None] > off_n[None, :]       # diagonal + upper stay zero",
    )
    assert lint_raw(strict, "F2.4") == []
