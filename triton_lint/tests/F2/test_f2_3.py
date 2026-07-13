"""F2.3 layout_churn."""

from __future__ import annotations

from conftest import src
from helpers import lint, lint_raw

from ._fixtures import NBYTES, SHAPES, TWO_ELEMENTWISE


class TestF23LayoutChurn:
    def test_fires_on_contiguous_after_permute(self, check):
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xt = x.permute(1, 0).contiguous()
        out = torch.empty_like(xt)
        exp_kernel[(1,)](xt, out, xt.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert "stride" in found[0].message

    def test_silent_on_contiguous_of_fresh_tensor(self, fired):
        """.contiguous() on an already-contiguous tensor is a no-op, not a copy."""
        assert not fired(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x).contiguous()
        exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )

    def test_silent_on_device_move(self, fired):
        """x.to('cuda') is not a cast and costs no pass."""
        assert not fired(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        x = x.to('cuda')
        out = torch.empty_like(x)
        exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )


class TestF23More:
    def test_fires_on_a_dtype_cast(self, check):
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xf = x.to(torch.float32)
        out = torch.empty_like(xf)
        exp_kernel[(1,)](xf, out, xf.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert found[0].data["op"] == "to"
        assert "tl.load(...).to(tl.float32)" in found[0].message

    def test_fires_on_a_dtype_keyword_cast(self, check):
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xf = x.to(dtype=torch.float16)
        out = torch.empty_like(xf)
        exp_kernel[(1,)](xf, out, xf.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1

    def test_silent_on_a_device_variable(self, fired):
        assert not fired(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        x = x.to(device)
        out = torch.empty_like(x)
        exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )

    def test_fires_on_contiguous_after_transpose_attribute(self, check):
        """x.T is a layout change, so .contiguous() on it really does copy."""
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xt = x.T.contiguous()
        out = torch.empty_like(xt)
        exp_kernel[(1,)](xt, out, xt.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert found[0].data["bytes"] == 2 * NBYTES

    def test_fires_on_contiguous_of_a_bound_permute(self, check):
        """The two-statement form: xt = x.permute(...); xt.contiguous().

        Regression: this was missed because the layout set is keyed by the raw scoped
        name (``forward::xt``) while the lookup canonicalised first, which resolves the
        alias back to its contiguous base (``forward::x``) and can never match.
        """
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xt = x.permute(1, 0)
        xc = xt.contiguous()
        out = torch.empty_like(xc)
        exp_kernel[(1,)](xc, out, xc.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert "stride" in found[0].message

    def test_silent_on_contiguous_of_a_slice(self, fired):
        """A slice of a contiguous tensor is contiguous -- no copy, no finding."""
        assert not fired(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xc = x[0].contiguous()
        out = torch.empty_like(xc)
        exp_kernel[(1,)](xc, out, xc.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )

    def test_fires_on_clone(self, check):
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xc = x.clone()
        out = torch.empty_like(xc)
        exp_kernel[(1,)](xc, out, xc.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert found[0].data["op"] == "clone"
        assert found[0].data["bytes"] == 2 * NBYTES
        assert "moves the whole tensor" in found[0].message

    def test_clone_of_an_expression_has_no_byte_count(self, check):
        """We cannot name the receiver's buffer, so we report the copy without a cost
        rather than attributing the wrong number of bytes to it."""
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xc = (x * 2).clone()
        out = torch.empty_like(xc)
        exp_kernel[(1,)](xc, out, xc.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert found[0].data["bytes"] is None
        assert "HBM traffic" not in found[0].message


CHURN_IN_BACKWARD = '''
class Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.permute(1, 0).contiguous()
        return g


class ModelNew(nn.Module):
    def forward(self, x):
        return Fn.apply(x)
'''


def test_layout_churn_in_backward_ignored():
    """The transpose only ever runs during training, never in the timed forward."""
    assert lint(CHURN_IN_BACKWARD, "F2.3") == []


# ---------------------------------------------------------------------------
# Regression tests for former linter bugs, now fixed (history in tests/BUGS.md).
# ---------------------------------------------------------------------------

#: `x` is made non-contiguous (permute), then REBOUND to a fresh contiguous tensor
#: (a relu output). The `noncontiguous` set never removes `x` on the rebind, so a
#: later bare `x.contiguous()` -- a genuine no-op -- is still flagged.
STALE_NONCONTIG = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def act_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < n
    tl.store(out_ptr + o, tl.sigmoid(tl.load(x_ptr + o, mask=m)), mask=m)

class ModelNew(nn.Module):
    def forward(self, x):
        x = x.permute(1, 0)     # x is now non-contiguous
        x = torch.relu(x)       # rebound to a fresh CONTIGUOUS tensor
        y = x.contiguous()      # no-op: x is already contiguous
        out = torch.empty_like(y)
        act_kernel[(1,)](y, out, y.numel(), BLOCK=128)
        return out
'''

#: The control: identical but WITHOUT the earlier permute, so `x` never enters the
#: non-contiguous set and the same `.contiguous()` is (correctly) silent. This pins
#: the bug to the stale flag, not to the relu rebind.
STALE_NONCONTIG_CONTROL = STALE_NONCONTIG.replace("        x = x.permute(1, 0)     # x is now non-contiguous\n", "")


def test_contiguous_of_a_freshly_computed_tensor_is_silent():
    """Control for BUG-19: relu output is contiguous, so .contiguous() is a no-op."""
    assert lint_raw(STALE_NONCONTIG_CONTROL, "F2.3") == []


def test_stale_noncontiguous_flag_does_not_fire():
    assert lint_raw(STALE_NONCONTIG, "F2.3") == []
