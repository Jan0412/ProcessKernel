"""F1.7 compile_offload."""

from __future__ import annotations

import pytest

from conftest import ELEMENTWISE_KERNEL, src
from helpers import lint, lint_raw


class TestF17CompileOffload:
    def test_fires_on_torch_compile(self, fired):
        assert fired(
            "F1.7",
            src(
                """
class ModelNew(nn.Module):
    def forward(self, x):
        f = torch.compile(lambda t: t * 2)
        return f(x)
"""
            ),
        )

    def test_silent_otherwise(self, fired):
        assert not fired("F1.7", src(ELEMENTWISE_KERNEL))


# ---------------------------------------------------------------------------
# Regression tests for former linter bugs, now fixed (history in tests/BUGS.md).
# ---------------------------------------------------------------------------


def test_module_level_jit_script_fails():
    body = '''
def plain(x):
    return x * 2

scripted = torch.jit.script(plain)

class ModelNew(nn.Module):
    def forward(self, x):
        return scripted(x)
'''
    findings = lint(body, "F1.7")
    assert [f.severity for f in findings] == ["fail"]


def test_torch_compile_decorator_fails():
    source = '''
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    @torch.compile
    def forward(self, x):
        return x * 2
'''
    findings = lint_raw(source, "F1.7")
    assert [f.severity for f in findings] == ["fail"]


# ---------------------------------------------------------------------------
# BUG-21 -- open. See tests/BUGS.md.
# ---------------------------------------------------------------------------


def test_script_if_tracing_is_not_a_compile_offload():
    # torch.jit.script_if_tracing compiles only when the function is called under
    # torch.jit.trace; in an ordinary eager forward it is a no-op passthrough and
    # generates no kernel. Real sample: p1510_s9 keeps the reference PyTorch
    # training path under this decorator and runs Triton for inference.
    body = '''
def fn(x):
    return x + 1

wrapped = torch.jit.script_if_tracing(fn)

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out
'''
    assert lint(body, "F1.7") == []


def test_jit_script_itself_still_fires():
    # Passing control for the test above: the prefix's intended target.
    body = '''
def fn(x):
    return x + 1

scripted = torch.jit.script(fn)

class ModelNew(nn.Module):
    def forward(self, x):
        return scripted(x)
'''
    assert [f.severity for f in lint(body, "F1.7")] == ["fail"]


def test_torch_compiler_query_is_not_a_compile_offload():
    # torch.compiler.is_compiling() asks a question; it compiles nothing. It is
    # caught only because "torch.compiler." starts with "torch.compile".
    body = '''
class ModelNew(nn.Module):
    def forward(self, x):
        if torch.compiler.is_compiling():
            return x
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out
'''
    assert lint(body, "F1.7") == []


def test_torch_compiler_disable_is_not_a_compile_offload():
    # The inverted case: @torch.compiler.disable explicitly opts the forward OUT
    # of torch.compile, and is reported as delegating compilation to torch.compile.
    source = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def work_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)


class ModelNew(nn.Module):
    @torch.compiler.disable
    def forward(self, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out
'''
    assert lint_raw(source, "F1.7") == []


def test_script_if_tracing_as_a_decorator_is_not_a_compile_offload():
    # Same no-op as the assignment form above, spelled as a decorator on a helper the
    # forward keeps for its eager training path. Exercises the decorator sweep, not the
    # call sweep, so a prefix fix must cover both.
    body = '''
@torch.jit.script_if_tracing
def fn(x):
    return x + 1

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out
'''
    assert lint(body, "F1.7") == []


def test_torch_compiler_namespace_noop_is_not_a_compile_offload():
    body = '''
class ModelNew(nn.Module):
    def forward(self, x):
        torch.compiler.cudagraph_mark_step_begin()
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out
'''
    assert lint(body, "F1.7") == []


def test_jit_trace_offload_still_fires():
    # Passing control: `torch.jit.trace` is a genuine offload (the prefix's intended
    # target). A prefix fix that whitelists longer names must keep the base names firing.
    body = '''
def plain(x):
    return x * 2

traced = torch.jit.trace(plain, torch.zeros(1))

class ModelNew(nn.Module):
    def forward(self, x):
        return traced(x)
'''
    assert [f.severity for f in lint(body, "F1.7")] == ["fail"]


def test_dynamo_optimize_offload_still_fires():
    # Passing control: `torch._dynamo.optimize` really does route compilation to Dynamo.
    body = '''
class ModelNew(nn.Module):
    def forward(self, x):
        f = torch._dynamo.optimize()(lambda t: t * 2)
        return f(x)
'''
    assert [f.severity for f in lint(body, "F1.7")] == ["fail"]
