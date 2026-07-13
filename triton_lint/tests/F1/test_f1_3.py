"""F1.3 discarded_output."""

from __future__ import annotations

from conftest import ELEMENTWISE_KERNEL, src
from helpers import lint


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
