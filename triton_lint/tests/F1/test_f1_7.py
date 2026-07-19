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


@pytest.mark.xfail(
    strict=True,
    reason="BUG-21: F1.7 matches host-call qualnames with str.startswith against the "
    "OFFLOAD_CALLS prefixes, so every longer name in those namespaces is swept in -- "
    "torch.jit.script_if_tracing (a no-op outside tracing; 11 of the run's 26 F1.7 "
    "findings, real sample p1510_s9) and the whole torch.compiler.* namespace, "
    "including torch.compiler.disable, which opts OUT of compilation. Reported at "
    "fail, the check's hardest verdict, on files with a genuine launched kernel",
)
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


@pytest.mark.xfail(
    strict=True,
    reason="BUG-21: F1.7 matches host-call qualnames with str.startswith against the "
    "OFFLOAD_CALLS prefixes, so every longer name in those namespaces is swept in -- "
    "torch.jit.script_if_tracing (a no-op outside tracing; 11 of the run's 26 F1.7 "
    "findings, real sample p1510_s9) and the whole torch.compiler.* namespace, "
    "including torch.compiler.disable, which opts OUT of compilation. Reported at "
    "fail, the check's hardest verdict, on files with a genuine launched kernel",
)
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


@pytest.mark.xfail(
    strict=True,
    reason="BUG-21: F1.7 matches host-call qualnames with str.startswith against the "
    "OFFLOAD_CALLS prefixes, so every longer name in those namespaces is swept in -- "
    "torch.jit.script_if_tracing (a no-op outside tracing; 11 of the run's 26 F1.7 "
    "findings, real sample p1510_s9) and the whole torch.compiler.* namespace, "
    "including torch.compiler.disable, which opts OUT of compilation. Reported at "
    "fail, the check's hardest verdict, on files with a genuine launched kernel",
)
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
