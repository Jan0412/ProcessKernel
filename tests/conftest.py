from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from triton_lint import analyze_source, build_model  # noqa: E402


@pytest.fixture
def analyze():
    """Analyze a source string; returns the ModuleModel."""
    return lambda src, shapes=None: build_model(src, "<test>", shapes)


@pytest.fixture
def check():
    """Run one check over a source string; returns its list of Findings."""

    def run(check_id: str, src: str, shapes=None):
        report = analyze_source(src, "<test>", only={check_id}, fallback_shapes=shapes)
        return [f for f in report.findings if f.check_id == check_id]

    return run


@pytest.fixture
def fired(check):
    """True if the check produced any finding."""
    return lambda check_id, src, shapes=None: bool(check(check_id, src, shapes))


PREAMBLE = """\
import torch
import torch.nn as nn
import triton
import triton.language as tl
"""


def src(body: str) -> str:
    return PREAMBLE + "\n" + body


ELEMENTWISE_KERNEL = '''
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)
'''
