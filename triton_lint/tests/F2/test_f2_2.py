"""F2.2 launch_overhead -- launch_in_loop and launch_count."""

from __future__ import annotations

import pytest

from conftest import src
from helpers import lint, lint_raw

from ._fixtures import BIG_SHAPES, SHAPES, THREE_LAUNCHES, TWO_ELEMENTWISE


class TestF22LaunchOverhead:
    def test_launch_in_loop_is_a_failure(self, check):
        found = check(
            "F2.2",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            exp_kernel[(1,)](x[i], out[i], x[i].numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        loop = [f for f in found if f.data.get("kind") == "launch_in_loop"]
        assert len(loop) == 1
        assert loop[0].severity == "fail"

    def test_single_launch_is_silent(self, fired):
        assert not fired(
            "F2.2",
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


class TestF22LaunchCount:
    def test_warns_when_launch_overhead_dominates(self, check):
        """Small problem: 3 launches cost ~15 us against ~0.02 us of memory time, so
        kernel count *is* the runtime. That regime is the actionable part."""
        found = check("F2.2", src(TWO_ELEMENTWISE + THREE_LAUNCHES), SHAPES)

        counts = [f for f in found if f.data.get("kind") == "launch_count"]
        assert len(counts) == 1
        assert counts[0].severity == "warn"
        assert counts[0].data["n_launches"] == 3
        assert counts[0].data["kernels"] == ["exp_kernel", "scale_kernel", "exp_kernel"]
        assert "launch overhead exceeds the memory-transfer time" in counts[0].message

    def test_only_informational_when_the_problem_is_memory_bound(self, check):
        """67 MB inputs: memory time swamps the launch overhead, so fusing for launch
        count alone would be the wrong advice."""
        found = check("F2.2", src(TWO_ELEMENTWISE + THREE_LAUNCHES), BIG_SHAPES)

        counts = [f for f in found if f.data.get("kind") == "launch_count"]
        assert len(counts) == 1
        assert counts[0].severity == "info"
        assert "exceeds the memory-transfer time" not in counts[0].message

    def test_no_regime_claim_without_shapes(self, check):
        """An unresolvable input shape means no byte count -- report the launches, and
        say nothing about the regime."""
        found = check(
            "F2.2",
            src(
                TWO_ELEMENTWISE
                + THREE_LAUNCHES
                + """
def get_inputs():
    return [torch.rand(n)]
"""
            ),
        )
        counts = [f for f in found if f.data.get("kind") == "launch_count"]
        assert len(counts) == 1
        assert counts[0].severity == "info"

    def test_below_threshold_is_silent(self, fired):
        assert not fired(
            "F2.2",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        n = x.numel()
        tmp = torch.empty_like(x)
        exp_kernel[(1,)](x, tmp, n, BLOCK=128)
        out = torch.empty_like(x)
        scale_kernel[(1,)](tmp, out, n, BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )

    def test_launch_in_a_while_loop(self, check):
        found = check(
            "F2.2",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        i = 0
        while i < x.shape[0]:
            exp_kernel[(1,)](x[i], out[i], x[i].numel(), BLOCK=128)
            i += 1
        return out
"""
            ),
            SHAPES,
        )
        loop = [f for f in found if f.data.get("kind") == "launch_in_loop"]
        assert len(loop) == 1
        assert loop[0].severity == "fail"
        assert loop[0].data["loop_vars"] == ["while"]


LOOP_IN_BACKWARD = '''
class Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out = torch.empty_like(grad_output)
        for i in range(4):
            work_kernel[(1,)](grad_output, out, grad_output.numel(), BLOCK=1024)
        return out


class ModelNew(nn.Module):
    def forward(self, x):
        return Fn.apply(x)
'''


def test_launch_in_loop_inside_backward_ignored():
    """Only the timed forward counts -- backward never runs under the eval harness."""
    assert lint(LOOP_IN_BACKWARD, "F2.2") == []


# ---------------------------------------------------------------------------
# Regression tests for former linter bugs, now fixed (history in tests/BUGS.md).
# ---------------------------------------------------------------------------

#: A sequential recurrence: iteration t reads the state that iteration t-1 wrote.
#: The loop dimension carries a true data dependency, so it CANNOT be moved into the
#: launch grid -- doing so would run every timestep from the same initial state.
SEQUENTIAL_RECURRENCE = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def step_kernel(h_ptr, x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < n
    tl.store(out_ptr + o, tl.load(h_ptr + o, mask=m) * 0.9 + tl.load(x_ptr + o, mask=m), mask=m)

class ModelNew(nn.Module):
    def forward(self, x):            # x: (T, N)
        n = x.shape[1]
        h = torch.zeros((n,), device=x.device, dtype=x.dtype)
        for t in range(x.shape[0]):
            h_new = torch.empty_like(h)
            step_kernel[(1,)](h, x[t], h_new, n, BLOCK=128)  # h_t = f(h_{t-1}, x_t)
            h = h_new
        return h
'''


def test_recurrence_launch_in_loop_still_fires():
    """Control for BUG-18: the loop launch IS a real cost, so F2.2 should not go
    silent -- the finding itself is legitimate; only its prescription is wrong."""
    fs = [f for f in lint_raw(SEQUENTIAL_RECURRENCE, "F2.2")
          if f.data.get("kind") == "launch_in_loop"]
    assert len(fs) == 1
    assert fs[0].severity == "fail"


def test_recurrence_is_not_told_to_move_the_loop_into_the_grid():
    fs = [f for f in lint_raw(SEQUENTIAL_RECURRENCE, "F2.2")
          if f.data.get("kind") == "launch_in_loop"]
    assert fs and "into the launch grid" not in fs[0].message


# ---------------------------------------------------------------------------
# BUG-33 -- open. See tests/BUGS.md.
# ---------------------------------------------------------------------------

_BUG33_REASON = (
    "BUG-33: _detect_recurrence only recognises a carried dependency in the host-rebind "
    "spelling -- it pairs a launch input to a launch output via an Assign `h = h_new`. A "
    "recurrence carried IN PLACE inside the kernel (the same buffer both loaded and "
    "stored by the launch, passed unchanged across the loop) has no such Assign, so "
    "recurrence stays False and F2.2 emits 'Move that dimension into the launch grid and "
    "launch the kernel once.' Gridding the loop races on the shared buffer and runs every "
    "iteration from the initial state -- a correctness bug. This is the F2.2 analogue of "
    "BUG-13/BUG-18: the recurrence is real but only its rebind spelling is detected. 33 "
    "of 935 recurrence=False launch-in-loop files carry an in-place buffer. Real sample: "
    "p6918_s3 (`for _ in range(iterations): sinkhorn_iter_kernel[grid](Q, ...)`, a "
    "sequential fixed-point iteration with no data dimension to grid at all)."
)

#: The same recurrence as SEQUENTIAL_RECURRENCE, carried IN PLACE: `state` is loaded and
#: stored by the launch every iteration and passed unchanged across the loop. There is no
#: host rebind, so _detect_recurrence never sees the dependency.
INPLACE_RECURRENCE = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scan_kernel(x_ptr, state_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < n
    tl.store(state_ptr + o, tl.load(state_ptr + o, mask=m) + tl.load(x_ptr + o, mask=m), mask=m)

class ModelNew(nn.Module):
    def forward(self, x):            # x: (T, N)
        n = x.shape[1]
        state = torch.zeros((n,), device=x.device, dtype=x.dtype)
        for t in range(x.shape[0]):
            scan_kernel[(1,)](x[t], state, n, BLOCK=128)   # state += x_t, in place
        return state
'''


def test_inplace_recurrence_launch_in_loop_still_fires():
    """Control for BUG-33: the loop launch IS a real cost -- the finding is legitimate,
    only its prescription is wrong. It must keep firing."""
    fs = [f for f in lint_raw(INPLACE_RECURRENCE, "F2.2")
          if f.data.get("kind") == "launch_in_loop"]
    assert len(fs) == 1
    assert fs[0].severity == "fail"


@pytest.mark.xfail(strict=True, reason=_BUG33_REASON)
def test_inplace_recurrence_is_not_told_to_move_the_loop_into_the_grid():
    fs = [f for f in lint_raw(INPLACE_RECURRENCE, "F2.2")
          if f.data.get("kind") == "launch_in_loop"]
    assert fs and "into the launch grid" not in fs[0].message


#: The same in-place recurrence carried by a `while` loop rather than `for`. Neither loop
#: form produces the host `h = h_new` rebind that _detect_recurrence looks for, so the
#: while spelling is dropped for the same reason -- the fix must key on the shared
#: load+store buffer, not on the loop construct.
INPLACE_RECURRENCE_WHILE = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scan_kernel(x_ptr, state_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < n
    tl.store(state_ptr + o, tl.load(state_ptr + o, mask=m) + tl.load(x_ptr + o, mask=m), mask=m)

class ModelNew(nn.Module):
    def forward(self, x):            # x: (T, N)
        n = x.shape[1]
        state = torch.zeros((n,), device=x.device, dtype=x.dtype)
        t = 0
        while t < x.shape[0]:
            scan_kernel[(1,)](x[t], state, n, BLOCK=128)   # state += x_t, in place
            t += 1
        return state
'''


def test_inplace_recurrence_while_launch_in_loop_still_fires():
    """Control for BUG-33: the loop launch is a real cost -- the finding stays legitimate."""
    fs = [f for f in lint_raw(INPLACE_RECURRENCE_WHILE, "F2.2")
          if f.data.get("kind") == "launch_in_loop"]
    assert len(fs) == 1
    assert fs[0].severity == "fail"


@pytest.mark.xfail(strict=True, reason=_BUG33_REASON)
def test_inplace_recurrence_while_is_not_told_to_move_the_loop_into_the_grid():
    fs = [f for f in lint_raw(INPLACE_RECURRENCE_WHILE, "F2.2")
          if f.data.get("kind") == "launch_in_loop"]
    assert fs and "into the launch grid" not in fs[0].message
