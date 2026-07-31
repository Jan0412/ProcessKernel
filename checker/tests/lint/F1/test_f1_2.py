"""F1.2 dead_kernel."""

from __future__ import annotations

import pytest

from conftest import ELEMENTWISE_KERNEL, src
from helpers import lint, lint_raw

from checker import analyze_source, build_model
from checker.lint.checks.family1 import f1_2_dead_kernel
from checker.core.feedback import render


def analyze(source):
    return build_model(source, "<t>")


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


class TestF12MissingEntryClass:
    """BUG-25: a generation with no entry-point class at all.

    KernelBench loads a solution with ``getattr(module, "ModelNew")``
    (kernelbench/eval.py), so a file without that class cannot be loaded and scores
    zero. Every kernel in it is *provably* dead -- there is nothing that could launch
    one. That is the strongest possible instance of what F1.2 exists to catch, and it
    is the one condition under which the check mutes itself:
    ``severity = "fail" if model.entry else "info"``.
    """

    KERNEL_ONLY = src(ELEMENTWISE_KERNEL)  # a bare kernel: no ModelNew anywhere

    def test_the_kernel_really_is_unlaunchable(self):
        """Premise: no entry class, so nothing in the file can ever launch it."""
        model = analyze(self.KERNEL_ONLY)
        assert model.model_class is None
        assert model.entry is None
        assert model.notes == ["no entry-point class found"]
        assert model.reachable_launches == []

    def test_missing_entry_class_is_a_fail(self, check):
        assert [f.severity for f in check("F1.2", self.KERNEL_ONLY)] == ["fail"]

    def test_missing_entry_class_is_not_clean(self):
        assert render(analyze_source(self.KERNEL_ONLY, "<t>")) is not None

    def test_control_adding_a_broken_wrapper_makes_it_fail(self, check):
        """The perverse gradient. This file is strictly *worse* -- it has the same dead
        kernel plus a ModelNew that returns an uninitialised buffer -- and F1.2 rates
        it `fail` where the kernel-only file above is `clean`. Deleting the wrapper
        class is rewarded.
        """
        found = check(
            "F1.2",
            self.KERNEL_ONLY
            + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        return torch.empty_like(x)
""",
        )
        assert [f.severity for f in found] == ["fail"]

    def test_control_entry_class_inheriting_its_forward_stays_silent(self, check):
        """The other cause of entry=None, which is *not* this bug: the class exists and
        inherits a forward that does launch. `model_class` is what separates them, so a
        fix must gate on that rather than on `entry`.
        """
        source = self.KERNEL_ONLY + """
class Base(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out


class ModelNew(Base):
    pass
"""
        model = analyze(source)
        assert model.entry is None and model.model_class == "ModelNew"
        assert check("F1.2", source) == []

    #: Two kernels, no entry class -- both are provably unlaunchable.
    TWO_KERNELS_NO_ENTRY = src(
        ELEMENTWISE_KERNEL
        + """
@triton.jit
def scale_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)
"""
    )

    def test_all_kernels_in_a_no_entry_file_are_fails(self, check):
        found = check("F1.2", self.TWO_KERNELS_NO_ENTRY)
        assert sorted(f.severity for f in found) == ["fail", "fail"]
        assert {f.data["kernel"] for f in found} == {"add_kernel", "scale_kernel"}

    def test_control_two_kernels_with_an_entry_that_launches_both_stay_silent(self, check):
        """Boundary: the same two kernels, both launched from a real ModelNew, must stay
        silent -- the fix keys on the missing entry class, not on kernel count."""
        source = self.TWO_KERNELS_NO_ENTRY + """
class ModelNew(nn.Module):
    def forward(self, x):
        tmp = torch.empty_like(x)
        add_kernel[(1,)](x, x, tmp, x.numel(), BLOCK=128)
        out = torch.empty_like(x)
        scale_kernel[(1,)](tmp, out, x.numel(), BLOCK=128)
        return out
"""
        assert check("F1.2", source) == []


def test_call_edge_to_a_missing_kernel_is_skipped():
    """The live-set closure walks `kernel.calls` (a @triton.jit device function that
    another kernel inlines rather than [grid]-launches). The parser only records edges
    to *defined* kernels, so an edge whose target is absent from `model.kernels` is
    reachable only by constructing that state directly -- and when it is, the walk must
    skip it, not dereference `None`. This exercises the `kernel is None` guard.
    """
    model = analyze(
        src(
            ELEMENTWISE_KERNEL
            + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out
"""
        )
    )
    model.kernels["add_kernel"].calls.add("ghost_dev")
    # The ghost edge is followed off the launched kernel, found missing, and skipped;
    # the one live kernel stays live, so nothing is reported and nothing raises.
    assert f1_2_dead_kernel.DeadKernel().run(model) == []
