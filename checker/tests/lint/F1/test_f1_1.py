"""F1.1 no_triton_kernel."""

from __future__ import annotations

from conftest import ELEMENTWISE_KERNEL, src
from helpers import lint_raw

from checker.lint.checks.family1 import f1_1_no_triton_kernel
from checker.core.model import ModuleModel


class TestF11NoTritonKernel:
    def test_fires_when_no_kernel(self, fired):
        assert fired("F1.1", src("class ModelNew(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)\n"))

    def test_silent_when_kernel_present(self, fired):
        assert not fired("F1.1", src(ELEMENTWISE_KERNEL))

    def test_silent_on_an_unparseable_file(self):
        """A file we could not parse has no evidence either way -- never accuse it."""
        model = ModuleModel(parse_status="syntax_error")
        assert f1_1_no_triton_kernel.NoTritonKernel().run(model) == []


# ---------------------------------------------------------------------------
# Regression tests for former linter bugs, now fixed (history in tests/BUGS.md).
# ---------------------------------------------------------------------------

NESTED_KERNEL = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl


def launch_double(x):
    out = torch.empty_like(x)

    @triton.jit
    def double_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)

    double_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
    return out


class ModelNew(nn.Module):
    def forward(self, x):
        return launch_double(x)
'''


def test_kernel_nested_in_launcher_is_not_missing():
    assert lint_raw(NESTED_KERNEL, "F1.1") == []
