"""F1.6 passthrough_kernel."""

from __future__ import annotations

import pytest

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


# ---------------------------------------------------------------------------
# BUG-22 -- open. See tests/BUGS.md.
# ---------------------------------------------------------------------------

_BUG22_REASON = (
    "BUG-22: _offset_signature strips every additive term that _is_base_ptr_term "
    "accepts, and a pointer hoisted into a local (`inp = x_ptr + rev_col`) is in "
    "ptr_taint, so a bare-Name address reduces to the EMPTY signature. Load and "
    "store then both signature-match on frozenset() and _is_matching_copy calls the "
    "kernel a memcpy -- exactly the gather/flip/concat kernels whose address "
    "arithmetic is the task, i.e. the ones BUG-7 was fixed to exclude. Fires only "
    "on the hoisted spelling (see the inline controls); 73 of the run's 527 F1.6 "
    "findings. Real samples: p369_s5 (hflip), p17827_s1 (gather)"
)

_PRE = """\
import torch
import torch.nn as nn
import triton
import triton.language as tl
"""

# The same gather in two spellings: address inline vs. hoisted into a local.
_GATHER_INLINE = _PRE + '''
@triton.jit
def gather_kernel(src_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask)
    val = tl.load(src_ptr + idx, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)
'''

_GATHER_HOISTED = _PRE + '''
@triton.jit
def gather_kernel(src_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask)
    in_ptr = src_ptr + idx
    val = tl.load(in_ptr, mask=mask)
    dst = out_ptr + offs
    tl.store(dst, val, mask=mask)
'''


def test_inline_gather_is_not_a_copy(analyze):
    # Passing control: with the addresses written inline the signatures differ
    # ({idx} vs {offs}) and the gather classifies correctly.
    assert analyze(_GATHER_INLINE).kernels["gather_kernel"].kind == "elementwise"


def test_hoisted_gather_is_not_a_copy(analyze):
    assert analyze(_GATHER_HOISTED).kernels["gather_kernel"].kind == "elementwise"


def test_f1_6_silent_on_hoisted_gather(fired):
    body = _GATHER_HOISTED + '''

class ModelNew(nn.Module):
    def forward(self, x, idx):
        out = torch.empty_like(x)
        gather_kernel[(1,)](x, idx, out, x.numel(), BLOCK=128)
        return out
'''
    assert not fired("F1.6", body)


def test_hoisted_hflip_is_not_a_copy(analyze):
    # p369_s5's shape: reads the reversed column, writes the forward one. The
    # reversal IS the task, and it lives entirely in the hoisted addresses.
    source = _PRE + '''
@triton.jit
def hflip_kernel(inp_ptr, out_ptr, stride_row, width, BLOCK: tl.constexpr):
    row_idx = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < width
    rev_col = width - 1 - col
    inp = inp_ptr + row_idx * stride_row + rev_col
    out = out_ptr + row_idx * stride_row + col
    val = tl.load(inp, mask=mask, other=0.0)
    tl.store(out, val, mask=mask)
'''
    assert analyze(source).kernels["hflip_kernel"].kind != "copy"


def test_hoisted_true_memcpy_is_still_a_copy(analyze):
    # Passing control the other way: hoisting must not make a real memcpy invisible.
    source = _PRE + '''
@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    src = x_ptr + offs
    dst = out_ptr + offs
    v = tl.load(src, mask=mask)
    tl.store(dst, v, mask=mask)
'''
    assert analyze(source).kernels["copy_kernel"].kind == "copy"


def test_hoisted_roll_is_not_a_copy(analyze):
    # A circular shift: reads column `(col + shift) % width`, writes column `col`. Like
    # the hflip, the whole task lives in the two hoisted addresses, which both collapse
    # to the empty signature -- a third distinct address-arithmetic kernel BUG-7 excludes.
    source = _PRE + '''
@triton.jit
def roll_kernel(inp_ptr, out_ptr, shift, width, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < width
    src_col = (col + shift) % width
    inp = inp_ptr + row * width + src_col
    out = out_ptr + row * width + col
    val = tl.load(inp, mask=mask)
    tl.store(out, val, mask=mask)
'''
    assert analyze(source).kernels["roll_kernel"].kind != "copy"


# ---------------------------------------------------------------------------
# BUG-29 -- open. See tests/BUGS.md.
# ---------------------------------------------------------------------------

_BUG29_REASON = (
    "BUG-29: a scalar offset *parameter* used purely additively on a load or store "
    "address's additive spine (`tl.store(dst_ptr + dst_offset + offs, v)`, or the load "
    "form `tl.load(src_ptr + src_offset + offs)`) is not recognised as a scalar -- "
    "_scalar_params only flags params used in a multiplicative or comparison context, so "
    "an offset that is only ever ADDED escapes it. _is_base_ptr_term then accepts the "
    "offset param as a base pointer, so _offset_signature strips it and the address's "
    "offset reduces to {offs}, matching the other side. _is_matching_copy calls the "
    "kernel a memcpy, so a slice-copy / torch.cat kernel that places its source at a "
    "parameter-controlled offset is classified `copy` and F1.6 reports it at fail as "
    "performing none of the task's computation -- exactly the concat/layout case BUG-7 "
    "excludes. (the offset param is also marked a phantom stored/loaded param, re-opening "
    "BUG-10 for the additive-only spelling; store form p7404_s4, load form p992_s2 / "
    "p17118_s2.) 147 of the run's 524 F1.6 findings. Real sample: p7404_s4 (a Triton "
    "torch.cat)."
)

_PRE29 = """\
import torch
import torch.nn as nn
import triton
import triton.language as tl
"""

# A concat/scatter: copies src into dst at a parameter-controlled offset. The value is
# a raw load, but the dst_offset placement is the task (this is how a Triton torch.cat
# is written). The address is INLINE (not hoisted -- that path is BUG-22).
_CONCAT_AT_PARAM_OFFSET = _PRE29 + '''
@triton.jit
def cat_kernel(src_ptr, dst_ptr, dst_offset, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    val = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + dst_offset + offs, val, mask=mask)
'''


def test_concat_at_scalar_param_offset_is_not_a_copy(analyze):
    assert analyze(_CONCAT_AT_PARAM_OFFSET).kernels["cat_kernel"].kind != "copy"


def test_f1_6_silent_on_concat_at_scalar_param_offset(fired):
    body = _CONCAT_AT_PARAM_OFFSET + '''

class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty(x.numel() + y.numel(), device=x.device, dtype=x.dtype)
        cat_kernel[(1,)](x, out, 0, x.numel(), BLOCK=128)
        cat_kernel[(1,)](y, out, x.numel(), y.numel(), BLOCK=128)
        return out
'''
    assert not fired("F1.6", body)


def test_concat_at_multiplied_offset_is_not_a_copy(analyze):
    # Passing control: the same placement written with a multiplied offset
    # (`row * stride`) is caught by _scalar_params, so the offset survives the
    # signature and the kernel classifies correctly. Pins the bug to the
    # purely-additive scalar param, not to the placement itself.
    source = _PRE29 + '''
@triton.jit
def cat_kernel(src_ptr, dst_ptr, row, stride, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    val = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + row * stride + offs, val, mask=mask)
'''
    assert analyze(source).kernels["cat_kernel"].kind != "copy"


def test_true_memcpy_with_offset_param_unused_is_still_a_copy(analyze):
    # Passing control the other way: when the offset param is genuinely not on the
    # store spine, the kernel is a real memcpy and must stay classified copy.
    source = _PRE29 + '''
@triton.jit
def copy_kernel(src_ptr, dst_ptr, dst_offset, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    val = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, val, mask=mask)
'''
    assert analyze(source).kernels["copy_kernel"].kind == "copy"


# The load-side face of BUG-29: the scalar offset param sits on the LOAD spine
# (`tl.load(src_ptr + src_offset + offs)`) rather than the store spine. Same mechanism --
# _scalar_params misses a purely-additive param, _is_base_ptr_term strips it, and the load
# signature collapses to {offs}, matching the store's {offs}. A slice-extraction copy that
# reads from a parameter-controlled offset. Real samples: p992_s2, p17118_s2.
_SLICE_AT_LOAD_OFFSET = _PRE29 + '''
@triton.jit
def slice_kernel(src_ptr, src_offset, dst_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    val = tl.load(src_ptr + src_offset + offs, mask=mask)
    tl.store(dst_ptr + offs, val, mask=mask)
'''


def test_slice_at_scalar_load_offset_is_not_a_copy(analyze):
    assert analyze(_SLICE_AT_LOAD_OFFSET).kernels["slice_kernel"].kind != "copy"


def test_f1_6_silent_on_slice_at_scalar_load_offset(fired):
    body = _SLICE_AT_LOAD_OFFSET + '''

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty(x.numel() // 2, device=x.device, dtype=x.dtype)
        slice_kernel[(1,)](x, x.numel() // 2, out, out.numel(), BLOCK=128)
        return out
'''
    assert not fired("F1.6", body)
