"""Test-suite root for kernel_gen.

Shared data crosses the ``unit/`` and ``integration/`` subdirs as **fixtures**, never as
``from conftest import ...`` -- a bare cross-file import from a subdir does not resolve,
and putting this dir on ``pythonpath`` would shadow ``checker/tests``' own conftest.
The consumers that need the corpus at *collection* time, to parametrize, read it with
their own local loader -- a session fixture cannot feed ``parametrize``.

Repo-root import is bootstrapped here so ``kernel_gen`` / ``checker`` resolve without
any pythonpath entry.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# -- shared kernel sources (mirror of checker/tests/conftest, kept independent) --

_PREAMBLE = "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n"

_ELEMENTWISE = '''
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)
'''

#: A complete, well-formed generation: kernel + ModelNew + get_inputs.
GOOD_KERNEL_FILE = _PREAMBLE + _ELEMENTWISE + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out

def get_inputs():
    return [torch.rand([4, 8]), torch.rand([4, 8])]
"""

#: The same, except the kernel is never launched -- F1.2 fires (with a lineno).
DEAD_KERNEL_FILE = _PREAMBLE + _ELEMENTWISE + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        return x + y
"""


@pytest.fixture
def good_kernel_file() -> str:
    return GOOD_KERNEL_FILE


@pytest.fixture
def dead_kernel_file() -> str:
    return DEAD_KERNEL_FILE


# -- the golden corpus -----------------------------------------------------

_CORPUS = pathlib.Path(__file__).parent / "fixtures" / "completions" / "corpus.jsonl"


def _read_corpus() -> list[dict]:
    return [json.loads(line) for line in _CORPUS.read_text().splitlines()]


@pytest.fixture(scope="session")
def golden_completions() -> list[dict]:
    return _read_corpus()
