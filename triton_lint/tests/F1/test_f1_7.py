"""F1.7 compile_offload."""

from __future__ import annotations

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
