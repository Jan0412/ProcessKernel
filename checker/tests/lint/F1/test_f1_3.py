"""F1.3 discarded_output."""

from __future__ import annotations

import pytest

from conftest import ELEMENTWISE_KERNEL, src
from helpers import lint

from checker import build_model
from checker.lint.checks.family1 import f1_3_discarded_output


class TestF13DiscardedOutput:
    def test_fires_when_output_thrown_away(self, check):
        found = check(
            "F1.3",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return torch.add(x, y)
"""
            ),
        )
        assert len(found) == 1
        assert found[0].data["outputs"] == ["out"]

    def test_silent_when_returned(self, fired):
        assert not fired(
            "F1.3",
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

    def test_silent_for_inplace_kernel(self, fired):
        """An in-place kernel writes into a forward input; that is not discarded."""
        assert not fired(
            "F1.3",
            src(
                """
@triton.jit
def inplace_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs) * 2.0, mask=offs < n)

class ModelNew(nn.Module):
    def forward(self, x):
        inplace_kernel[(1,)](x, x.numel(), BLOCK=128)
        return x
"""
            ),
        )


class TestF13Guards:
    def test_silent_when_the_kernel_writes_nothing(self, fired):
        """No store target -- there is no output to discard. Not this check's business."""
        assert not fired(
            "F1.3",
            src(
                """
@triton.jit
def probe_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n)

class ModelNew(nn.Module):
    def forward(self, x):
        probe_kernel[(1,)](x, x.numel(), BLOCK=128)
        return x
"""
            ),
        )

    def test_silent_when_the_output_argument_is_unresolvable(self, fired):
        """The launch omits out_ptr entirely. We resolved no output buffer, so we stay
        quiet rather than invent a finding."""
        assert not fired(
            "F1.3",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        add_kernel[(1,)](x, y)
        return torch.relu(x)
"""
            ),
        )


# ---------------------------------------------------------------------------
# Regression tests for former linter bugs, now fixed (history in tests/BUGS.md). The 2026-07-13 audit found 4/4 sampled F1.3
# findings false; these pin the causes.
# ---------------------------------------------------------------------------

PIPELINE = '''
class ModelNew(nn.Module):
    def forward(self, x):
        tmp = torch.empty_like(x)
        work_kernel[(1,)](x, tmp, x.numel(), BLOCK=1024)
        out = torch.empty_like(x)
        work_kernel[(1,)](tmp, out, x.numel(), BLOCK=1024)
        return out
'''


def test_output_consumed_by_next_kernel_is_not_discarded():
    assert lint(PIPELINE, "F1.3") == []


SUBSCRIPT_READ = '''
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.zeros(1, dtype=x.dtype, device=x.device)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        mean = out[0] / x.numel()
        return mean
'''


def test_subscript_host_read_is_a_use():
    assert lint(SUBSCRIPT_READ, "F1.3") == []


BOUND_SLICE = '''
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        left = out[:, :2]
        return left
'''


def test_returning_a_bound_slice_is_a_use():
    assert lint(BOUND_SLICE, "F1.3") == []


RETURN_TORCH_CAT = '''
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return torch.cat([out, x], dim=1)
'''


def test_output_returned_through_torch_cat_is_a_use():
    assert lint(RETURN_TORCH_CAT, "F1.3") == []


# `partial` is passed *bare* to the launch (counted in launch_loads but never in
# `loads`, because the launch's generic_visit hits visit_Name, which records only
# `referenced`), then read once on the host by a call: `total = torch.sum(partial)`.
# _resolve_reads computes `loads(1) - launch_loads(1) = 0`, so the genuine host read
# is cancelled and the buffer looks discarded. A buffer read via a *compound* launch
# arg (`x.numel()`) survives, because the inner name is counted in `loads` too -- so
# the off-by-one bites only bare-Name launch args with exactly one host read.
BARE_ARG_ONE_HOST_READ = '''
class ModelNew(nn.Module):
    def forward(self, x):
        partial = torch.empty(4, device=x.device, dtype=x.dtype)
        work_kernel[(4,)](x, partial, x.numel(), BLOCK=1024)
        total = torch.sum(partial)
        return total
'''

BARE_ARG_TWO_HOST_READS = '''
class ModelNew(nn.Module):
    def forward(self, x):
        partial = torch.empty(4, device=x.device, dtype=x.dtype)
        work_kernel[(4,)](x, partial, x.numel(), BLOCK=1024)
        tmp = torch.relu(partial)
        total = torch.sum(tmp) + torch.sum(partial)
        return total
'''


def test_two_host_reads_of_a_bare_launch_arg_survive():
    # Control for BUG-15: with two host reads the count is `2 - 1 = 1 > 0`, so the
    # over-subtraction does not push it to zero and F1.3 correctly stays silent.
    # This pins that the bug is the off-by-one, not the shape of the read.
    assert lint(BARE_ARG_TWO_HOST_READS, "F1.3") == []


def test_single_call_host_read_of_a_bare_launch_arg_is_a_use():
    assert lint(BARE_ARG_ONE_HOST_READ, "F1.3") == []


# ---------------------------------------------------------------------------
# BUG-23 -- open. See tests/BUGS.md.
# ---------------------------------------------------------------------------

_BUG23_REASON = (
    "BUG-23: hostflow's visit_Assign records an alias for a method-call view "
    "(`y = x.view(...)`, via ALIAS_METHODS) and for a bare rebind (`y = x`), but "
    "never for a SUBSCRIPT view (`c = out[b]`) -- an ast.Subscript reaches neither "
    "branch. Writing a tensor slice is a write into the parent's storage, so the "
    "standard per-slice launch idiom (`for b in range(B): c = out[b]; k[grid](.., c, ..)`) "
    "leaves the store target as a fresh buffer nobody reads, and F1.3 reports the "
    "output discarded at fail while `out` is returned right below it. 389 of the "
    "run's 1209 F1.3-flagged files launch a subscript view. Real sample: p2141_s4"
)


def test_kernel_writing_a_subscript_view_is_not_discarded(fired):
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        for b in range(x.shape[0]):
            sl = out[b]
            add_kernel[(1,)](x[b], y[b], sl, sl.numel(), BLOCK=128)
        return out
"""
    )
    assert not fired("F1.3", body)


def test_kernel_writing_a_name_alias_is_not_discarded(fired):
    # Passing control: the bare-rebind spelling of the same dataflow.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        sl = out
        add_kernel[(1,)](x, y, sl, sl.numel(), BLOCK=128)
        return out
"""
    )
    assert not fired("F1.3", body)


def test_kernel_writing_a_method_view_is_not_discarded(fired):
    # Passing control: the `.view()` spelling of the same dataflow.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        sl = out.view(-1)
        add_kernel[(1,)](x, y, sl, sl.numel(), BLOCK=128)
        return out
"""
    )
    assert not fired("F1.3", body)


def test_kernel_writing_a_tuple_index_view_is_not_discarded(fired):
    # A multi-axis subscript `out[b, 0]` is still an ast.Subscript that reaches neither
    # alias branch -- the same gap for the (b, c)-indexed per-slice launch.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        for b in range(x.shape[0]):
            sl = out[b, 0]
            add_kernel[(1,)](x[b], y[b], sl, sl.numel(), BLOCK=128)
        return out
"""
    )
    assert not fired("F1.3", body)


def test_kernel_writing_a_bound_slice_view_is_not_discarded(fired):
    # A slice `out[b:b+1]` is an ast.Subscript too (with a Slice index); it writes into
    # the parent storage exactly like `out[b]`, and the alias is likewise never recorded.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        for b in range(x.shape[0]):
            sl = out[b:b + 1]
            add_kernel[(1,)](x[b], y[b], sl, sl.numel(), BLOCK=128)
        return out
"""
    )
    assert not fired("F1.3", body)


# ---------------------------------------------------------------------------
# BUG-28 -- open. See tests/BUGS.md.
# ---------------------------------------------------------------------------

_BUG28_REASON = (
    "BUG-28: a kernel launched inside a helper writes one of the helper's own "
    "*parameters* (`def launch(a, b, out): add_kernel[grid](a, b, out, ...)`), and the "
    "caller allocates that tensor and returns or reads it. F1.3 resolves the store "
    "target in the launch's *enclosing* scope -- the helper -- where `out` is a bare "
    "parameter buffer that carries no `returned`/`read_by_host`/`loaded_by` flag. "
    "_propagate_interprocedural pushes the store role UP to the caller's buffer (so "
    "forward::out becomes stored+returned) but never pushes the caller's use flags DOWN "
    "to the helper param, so `used()` is false on the helper buffer and the output reads "
    "as discarded. ~54 of the run's 1209 F1.3-flagged files; real sample p17053_s6 (a "
    "3-kernel GroupNorm whose launch_sum/launch_sumsq/launch_norm helpers each take a "
    "bare out-parameter)."
)

# The kernel is launched from a helper; `out` is one of the helper's parameters,
# passed in by the caller, which owns it and returns it.
_OUT_PARAM_HELPER = ELEMENTWISE_KERNEL + """
def launch_add(a, b, out):
    add_kernel[(1,)](a, b, out, out.numel(), BLOCK=128)
"""

_OUT_PARAM_RETURNED = src(
    _OUT_PARAM_HELPER
    + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        launch_add(x, y, out)
        return out
"""
)

_OUT_PARAM_HOST_READ = src(
    _OUT_PARAM_HELPER
    + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        launch_add(x, y, out)
        return out.sum()
"""
)


def test_out_parameter_of_a_launch_helper_that_is_returned_is_not_discarded(fired):
    assert not fired("F1.3", _OUT_PARAM_RETURNED)


def test_out_parameter_of_a_launch_helper_that_is_host_read_is_not_discarded(fired):
    assert not fired("F1.3", _OUT_PARAM_HOST_READ)


def test_same_launch_inline_in_forward_is_not_discarded(fired):
    # Passing control: move the launch out of the helper and the FP disappears --
    # pinning the bug to the helper boundary, not the shape of the dataflow.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, out.numel(), BLOCK=128)
        return out
"""
    )
    assert not fired("F1.3", body)


def test_out_parameter_genuinely_discarded_still_fires(fired):
    # Passing control the other way: when the caller really does throw the written
    # tensor away, F1.3 must keep firing even through the helper. This is why the fix
    # must propagate the caller's use flags to the helper param, not blanket-suppress
    # every launch that sits inside a helper.
    body = src(
        _OUT_PARAM_HELPER
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        launch_add(x, y, out)
        return torch.relu(x)
"""
    )
    assert fired("F1.3", body)


# The launch is two hops down: forward -> outer -> inner, with `out` threaded through
# both as a bare parameter. The caller's use flags must propagate down more than one
# frame, so a fix that only reaches the immediate caller still misses this.
_OUT_PARAM_TWO_HOPS = ELEMENTWISE_KERNEL + """
def inner(a, b, out):
    add_kernel[(1,)](a, b, out, out.numel(), BLOCK=128)

def outer(a, b, out):
    inner(a, b, out)
"""


def test_out_parameter_through_two_helper_hops_is_not_discarded(fired):
    body = src(
        _OUT_PARAM_TWO_HOPS
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        outer(x, y, out)
        return out
"""
    )
    assert not fired("F1.3", body)


def test_out_parameter_of_a_launch_helper_read_via_subscript_is_not_discarded(fired):
    # The caller consumes the helper-written tensor by a subscript host read rather than
    # returning it -- another use flag that never reaches the helper's parameter buffer.
    body = src(
        _OUT_PARAM_HELPER
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty(4, device=x.device, dtype=x.dtype)
        launch_add(x, y, out)
        return out[0] / 2.0
"""
    )
    assert not fired("F1.3", body)


# ---------------------------------------------------------------------------
# BUG-34 -- open. See tests/BUGS.md.
# ---------------------------------------------------------------------------

_BUG34_REASON = (
    "BUG-34: hostflow's `_returned_names` has no `ast.IfExp` case, so a kernel output "
    "returned through a conditional expression "
    "(`return out.mean() if flag else out.sum()`) is marked neither `returned` -- the "
    "IfExp recursion never runs, so even a bare `else out` branch is missed -- nor "
    "`read_by_host`, because visit_Return's `_count` still walks the whole IfExp with "
    "ast.walk and books both loads as `return_loads`, which `_resolve_reads` subtracts "
    "to zero. The output buffer carries no use flag at all and F1.3 reports it discarded "
    "at fail, while it is in fact consumed on the host and returned. The non-ternary "
    "spelling (`return out.mean()`) is accidentally correct because `_returned_names` "
    "recurses the method receiver and sets `returned`. 34 of the run's 1209 F1.3-flagged "
    "files return the output through a ternary (a further ~78 through BoolOp/UnaryOp/"
    "Compare, which the same missing recursion also drops). Real sample: p4416_s2 "
    "(focal loss, `return out.mean() if size_average else out.sum()`)."
)

# The kernel output is consumed by a host reduction whose scalar is returned through a
# ternary -- the real focal-loss shape.
_TERNARY_RETURN = src(
    ELEMENTWISE_KERNEL
    + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out.mean() if x.numel() > 1 else out.sum()
"""
)


def test_output_returned_through_a_ternary_is_not_discarded(fired):
    assert not fired("F1.3", _TERNARY_RETURN)


def test_output_consumed_by_a_non_ternary_host_reduction_is_not_discarded(fired):
    # Passing control: the same dataflow without the ternary. `_returned_names` recurses
    # the method receiver of `out.sum()` and sets `returned`, so F1.3 stays silent --
    # pinning the bug to the IfExp spelling, not to host-reducing the output.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out.sum()
"""
    )
    assert not fired("F1.3", body)


# BUGS.md notes ~78 further files return the output through BoolOp/UnaryOp/Compare, which
# the same missing recursion drops -- a complete fix recurses every operator node, not
# only ast.IfExp. These pin the wider blast radius so a narrow IfExp-only fix stays red.
def test_output_returned_through_a_unary_op_is_not_discarded(fired):
    # `return -out.sum()` (a negated loss): _returned_names has no ast.UnaryOp case.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return -out.sum()
"""
    )
    assert not fired("F1.3", body)


def test_output_returned_through_a_compare_is_not_discarded(fired):
    # `return out.sum() > 0`: _returned_names has no ast.Compare case either.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out.sum() > 0
"""
    )
    assert not fired("F1.3", body)


def test_ternary_with_a_bare_output_branch_is_not_discarded(fired):
    # Even the plainest IfExp branch -- a bare `out` -- is missed: the IfExp recursion
    # never runs, so the bare Name never marks `out` returned.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out.mean() if x.numel() > 1 else out
"""
    )
    assert not fired("F1.3", body)


def test_output_returned_through_a_tuple_is_not_discarded(fired):
    # Passing control: a tuple return `return out.mean(), x` IS recursed by
    # `_returned_names` (the ast.Tuple case), so `out` is marked returned and F1.3 stays
    # silent -- pinning the bug to the operator nodes that lack a recursion case.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out.mean(), x
"""
    )
    assert not fired("F1.3", body)


# ---------------------------------------------------------------------------
# BUG-35 -- open. See tests/BUGS.md.
# ---------------------------------------------------------------------------

_BUG35_REASON = (
    "BUG-35: `_record_binding` records an alias for a tensor-view method "
    "(`p = out.view(...)`, via ALIAS_METHODS) and for a plain rebind (`p = out`), but "
    "not for a `.data_ptr()` call. When a kernel's store target is launched through a "
    "raw pointer bound from `out.data_ptr()` (or `out[i].data_ptr()`) -- a common way "
    "to hand Triton a per-slice base address -- the pointer local becomes a phantom "
    "buffer with no returned/read_by_host/loaded_by flag and no alias back to `out`, so "
    "`used()` is false and F1.3 reports the output discarded at fail while `out` is "
    "returned. Fires only on the *bound* spelling: an inline `add_kernel[grid](.., "
    "out.data_ptr(), ..)` resolves through `_base_name` straight to `out` and is silent. "
    "349 of the run's 1209 F1.3-flagged files bind the reported output through "
    "`.data_ptr()`. Real sample: p92_s7 (Qwen3.6 level-2, per-batch "
    "`out_batch_ptr = out[i].data_ptr()` in a launch loop)."
)

_DATAPTR_TARGET = src(
    ELEMENTWISE_KERNEL
    + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        out_ptr = out.data_ptr()
        add_kernel[(1,)](x, y, out_ptr, x.numel(), BLOCK=128)
        return out
"""
)


def test_output_launched_through_a_bound_data_ptr_is_not_discarded(fired):
    assert not fired("F1.3", _DATAPTR_TARGET)


def test_output_launched_through_an_inline_data_ptr_is_not_discarded(fired):
    # Passing control: the inline spelling of the same launch. `_base_name` walks the
    # Call/Attribute chain of `out.data_ptr()` straight to `out`, which is returned, so
    # F1.3 stays silent -- pinning the bug to the intermediate binding, not to data_ptr.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out.data_ptr(), x.numel(), BLOCK=128)
        return out
"""
    )
    assert not fired("F1.3", body)


def test_output_launched_through_a_per_slice_data_ptr_is_not_discarded(fired):
    # The real p92_s7 shape: a per-batch loop binds `p = out[i].data_ptr()` and hands the
    # raw pointer to the kernel. Because `_base_name` walks Subscript, aliasing a
    # `.data_ptr()` binding to its receiver covers this per-slice form too -- so this and
    # the plain `out.data_ptr()` binding above must both go silent under one fix.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            p = out[i].data_ptr()
            add_kernel[(1,)](x[i], y[i], p, x[i].numel(), BLOCK=128)
        return out
"""
    )
    assert not fired("F1.3", body)


# ---------------------------------------------------------------------------
# Conservative-by-construction guards: an output the check cannot resolve stays silent.
# ---------------------------------------------------------------------------


def test_output_arg_without_a_base_name_is_skipped(fired):
    """The store-target argument is an expression with no leftmost Name (`out + 1`), so
    `_base_name` yields None and the launch resolves to no output buffer. The check has
    nothing to prove discarded and stays silent -- exercising the `var is None` branch
    of the output-resolution loop.
    """
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out + 1, x.numel(), BLOCK=128)
        return torch.relu(x)
"""
    )
    assert not fired("F1.3", body)


def test_launch_of_a_kernel_absent_from_the_model_is_skipped():
    """A launch site is only recorded for a defined kernel, so `kernel_name` is always a
    key of `model.kernels` after a normal build. The `kernel is None` guard still has to
    hold if that invariant is broken -- exercise it directly.
    """
    model = build_model(
        src(
            ELEMENTWISE_KERNEL
            + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return torch.relu(x)
"""
        ),
        "<t>",
    )
    del model.kernels["add_kernel"]
    assert f1_3_discarded_output.check(model) == []


def test_output_returned_through_a_boolop_is_not_discarded(fired):
    # BUG-34 class coverage: `return (out.sum() > 0) and flag` routes the output through a
    # BoolOp -- the same missing-recursion class as IfExp/UnaryOp/Compare.
    body = src(
        ELEMENTWISE_KERNEL
        + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return (out.sum() > 0) and bool(x.numel())
"""
    )
    assert not fired("F1.3", body)
