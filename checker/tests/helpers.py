"""Shared helpers for the checker test suite.

Two kinds of tests use these:

* synthetic tests (``F1/``, ``F2/``) build a small source string around
  :data:`PRELUDE` and lint it in-memory;
* real-sample tests (``real_samples/``) lint actual generated kernels copied
  from a run folder, with hand-verified ground truth.

The auditing history lives in ``BUGS.md``: bugs were first encoded as
``@pytest.mark.xfail(strict=True, reason="BUG-N: ...")`` tests, and once the linter
was fixed each marker was removed, leaving the assertion as a permanent regression
test that pins the fix.
"""

from __future__ import annotations

from checker import analyze_source
from checker.model import Finding, FileReport

#: A minimal, real kernel (elementwise x*2) plus the imports every synthetic
#: test body needs. Kernels under test are appended after it.
PRELUDE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def work_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)
'''


def lint(body: str, only: str) -> list[Finding]:
    """Lint PRELUDE + *body* with a single check and return its findings."""
    report = analyze_source(PRELUDE + body, path="test.py", only={only})
    assert report.parse_status == "ok"
    return [f for f in report.findings if f.check_id == only]


def lint_raw(source: str, only: str) -> list[Finding]:
    """Lint *source* as-is (no PRELUDE) with a single check."""
    report = analyze_source(source, path="test.py", only={only})
    assert report.parse_status == "ok"
    return [f for f in report.findings if f.check_id == only]


def forward_with(expr: str) -> str:
    """A ModelNew whose forward launches work_kernel and returns *expr*."""
    return f'''
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return {expr}
'''


def severities(findings: list[Finding]) -> list[str]:
    return [f.severity for f in findings]


def one(findings: list[Finding]) -> Finding:
    assert len(findings) == 1, f"expected exactly one finding, got {findings}"
    return findings[0]
