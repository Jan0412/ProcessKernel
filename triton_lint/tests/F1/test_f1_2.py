"""F1.2 dead_kernel."""

from __future__ import annotations

from conftest import ELEMENTWISE_KERNEL, src
from helpers import lint, lint_raw


class TestF12DeadKernel:
    def test_fires_when_never_launched(self, check):
        found = check(
            "F1.2",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        return x + y
"""
            ),
        )
        assert len(found) == 1
        assert found[0].data["kernel"] == "add_kernel"

    def test_silent_when_launched_via_helper(self, fired):
        assert not fired(
            "F1.2",
            src(
                ELEMENTWISE_KERNEL
                + """
def helper(x, y):
    out = torch.empty_like(x)
    add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
    return out

class ModelNew(nn.Module):
    def forward(self, x, y):
        return helper(x, y)
"""
            ),
        )

    def test_silent_when_launched_via_submodule(self, fired):
        assert not fired(
            "F1.2",
            src(
                ELEMENTWISE_KERNEL
                + """
class AddTriton(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.add = AddTriton()
    def forward(self, x, y):
        return self.add(x, y)
"""
            ),
        )

    def test_silent_when_launched_via_autograd_function(self, fired):
        assert not fired(
            "F1.2",
            src(
                ELEMENTWISE_KERNEL
                + """
class AddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out

class ModelNew(nn.Module):
    def forward(self, x, y):
        return AddFn.apply(x, y)
"""
            ),
        )


class TestF12Guards:
    def test_silent_without_any_kernel(self, fired):
        """F1.1 already says "no kernel"; F1.2 must not pile on."""
        assert not fired(
            "F1.2",
            src("class ModelNew(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)\n"),
        )


# ---------------------------------------------------------------------------
# Timed-vs-conservative reachability.
# ---------------------------------------------------------------------------


def test_kernel_launched_only_from_unreached_method_is_dead():
    # The harness only calls forward(); a kernel reachable only through
    # predict() never runs (real sample: p17660_s7).
    body = '''
class ModelNew(nn.Module):
    def forward(self, x):
        return torch.relu(x)

    def predict(self, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out
'''
    findings = lint(body, "F1.2")
    assert [f.severity for f in findings] == ["fail"]


BACKWARD_KERNEL = '''
class GradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        out = torch.empty_like(grad_output)
        work_kernel[(1,)](grad_output, out, grad_output.numel(), BLOCK=1024)
        return out


class ModelNew(nn.Module):
    def forward(self, x):
        return GradFn.apply(x)
'''


def test_kernel_launched_only_from_backward_is_not_dead():
    # The conservative `reachable` set must keep treating backward as live,
    # or every hand-written Triton backward becomes an F1.2 false positive.
    assert lint(BACKWARD_KERNEL, "F1.2") == []


# ---------------------------------------------------------------------------
# Regression tests for former linter bugs, now fixed (history in tests/BUGS.md).
# ---------------------------------------------------------------------------

DEFAULT_ARG_CLASS = '''
def t_act(x):
    out = torch.empty_like(x)
    work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
    return out


class TritonAct(nn.Module):
    def forward(self, x):
        return t_act(x)


class ModelNew(nn.Module):
    def __init__(self, act_layer=TritonAct):
        super().__init__()
        self.act = act_layer()

    def forward(self, x):
        return self.act(x)
'''


def test_class_referenced_as_default_argument_is_reachable():
    assert lint(DEFAULT_ARG_CLASS, "F1.2") == []


# A @triton.jit *device function*: called from inside another kernel body (Triton
# inlines it), never launched with `[grid]`. It runs on every forward, yet F1.2
# only counts subscript launch sites, so it looks "dead" -- and the advice "launch
# it (or remove it)" is destructive: removing it breaks the kernel that calls it.
DEVICE_FUNCTION = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rsqrt_dev(x):
    return 1.0 / tl.sqrt(x)


@triton.jit
def norm_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, v * rsqrt_dev(v), mask=mask)


class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        norm_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out
'''


def test_orphan_jit_function_really_is_dead():
    # Control for BUG-14: a @triton.jit function that no kernel calls and no host
    # launches is genuinely dead -- F1.2 must keep firing on it. This pins the
    # boundary the fix has to respect (called-from-a-kernel vs. called-by-nobody).
    orphan = '''
@triton.jit
def orphan_dev(x):
    return x * 2.0


class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out
'''
    findings = lint(orphan, "F1.2")
    assert [f.data["kernel"] for f in findings] == ["orphan_dev"]


def test_device_function_called_from_kernel_is_not_dead():
    assert lint_raw(DEVICE_FUNCTION, "F1.2") == []
