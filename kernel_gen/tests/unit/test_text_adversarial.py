"""Hand-built inputs for ``extract_code_block``, kept regardless of measured frequency.

Every count quoted elsewhere in this suite comes from one model (Qwen3.6) on one prompt
format. The correctness claims survive a model change; the frequency claims do not. This
battery is the part that does -- each case is a shape a *different* model could plausibly
emit, and several are already handled only by luck.

The ranking bugs KGEN-11 and KGEN-19 are covered here alongside the fence shapes.
"""

from __future__ import annotations

import pytest

from kernel_gen.core.text import ENTRY_CLASS, extract_code_block

KERNEL = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import triton\n"
    "import triton.language as tl\n"
    "\n\n"
    "@triton.jit\n"
    "def k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):\n"
    "    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
    "    tl.store(o_ptr + i, tl.load(x_ptr + i, mask=i < n), mask=i < n)\n"
    "\n\n"
    "class ModelNew(nn.Module):\n"
    "    def forward(self, x):\n"
    "        o = torch.empty_like(x)\n"
    "        k[(1,)](x, o, x.numel(), BLOCK=128)\n"
    "        return o\n"
)


def _fence(body: str, tag: str = "python") -> str:
    return f"```{tag}\n{body}```\n"


# -- fence syntax the parser must keep surviving ---------------------------------
# Several of these produce ZERO fenced blocks and are recovered only because the
# no-fence path picks them up. That is fine, but it is luck rather than design, so it
# is pinned: if _fenced_blocks ever starts recognising them, these still must pass.

@pytest.mark.parametrize(
    "name,raw",
    [
        ("plain", _fence(KERNEL)),
        ("no trailing newline at EOF", "```python\n" + KERNEL.rstrip() + "\n```"),
        ("trailing spaces after the info string", "```python   \n" + KERNEL + "```   \n"),
        ("indented opening fence", "1. Here:\n\n   ```python\n" + KERNEL + "   ```\n"),
        ("bare ``` opener then the real block", "```\nx = 1\n```\n" + _fence(KERNEL)),
        ("tilde fence", "~~~python\n" + KERNEL + "~~~\n"),
        ("four-backtick fence", "````python\n" + KERNEL + "````\n"),
        ("compound info string", "```python title=x\n" + KERNEL + "```\n"),
        ("info string with a space", "```python 3\n" + KERNEL + "```\n"),
        ("CRLF line endings", "```python\r\n" + KERNEL.replace("\n", "\r\n") + "```\r\n"),
        ("bare CR line endings", "```python\r" + KERNEL + "```\r"),
        ("BOM prefix", "﻿" + _fence(KERNEL)),
        ("unicode in a comment", _fence("# é中文\n" + KERNEL)),
        ("fence marker inside a string literal", _fence("s = '```python'\n" + KERNEL)),
        ("markdown fence in a comment", _fence("# see ```python below\n" + KERNEL)),
    ],
)
def test_the_kernel_survives_this_fence_shape(name, raw):
    out = extract_code_block(raw)
    assert ENTRY_CLASS in out, name
    assert "import torch" in out, name


# -- degenerate inputs: returning nothing is CORRECT here -------------------------

@pytest.mark.parametrize(
    "name,raw",
    [
        ("empty completion", ""),
        ("whitespace only", "   \n\t\n"),
        ("an opener with nothing after it", "```python"),
        ("prose with no code at all", "I considered a tiled matmul but ran out of time."),
    ],
)
def test_a_completion_with_no_kernel_does_not_invent_one(name, raw):
    out = extract_code_block(raw)  # must not raise
    assert ENTRY_CLASS not in out, name


# -- entry-class detection --------------------------------------------------------

def test_modelnew_bound_by_assignment_is_a_submission():
    # checker/submission S1.1 accepts a module-level binding, so extraction must not
    # discard one just because the `class ModelNew` spelling is absent.
    raw = _fence(KERNEL.replace("class ModelNew(nn.Module):", "class K(nn.Module):") + "\nModelNew = K\n")
    assert "ModelNew" in extract_code_block(raw)


def test_a_docstring_fence_does_not_truncate_the_block():
    # A ``` inside a docstring closes the block early at the text level, but the region
    # after it is now an ordinary candidate, so the kernel is still reachable.
    raw = _fence('s = """\n```\n"""\n' + KERNEL)
    assert ENTRY_CLASS in extract_code_block(raw)


# -- the ranking bugs KGEN-11 and KGEN-19 --------------------------------------

def test_an_answer_between_two_fenced_scraps_is_recovered():
    raw = _fence("x = 1\n") + "\nFinal:\n\n" + KERNEL + "\n" + _fence("z = 3\n")
    assert ENTRY_CLASS in extract_code_block(raw)


def test_a_class_named_ModelNewHelper_does_not_outrank_the_real_submission():
    raw = _fence(KERNEL) + _fence("class ModelNewHelper:\n    pass\n")
    out = extract_code_block(raw)
    assert "class ModelNewHelper" not in out
    assert ENTRY_CLASS in out


def test_class_ModelNew_in_a_comment_does_not_outrank_the_real_submission():
    raw = _fence(KERNEL) + _fence("# class ModelNew goes here\ny = 2\n")
    out = extract_code_block(raw)
    assert "# class ModelNew goes here" not in out
    assert ENTRY_CLASS in out
