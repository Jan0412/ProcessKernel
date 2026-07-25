"""``kernel_gen.core.text``: pull the kernel out of a completion, parse specs and ids.

Extraction is the single highest-stakes pure function in the pipeline: it decides which
bytes of a free-form completion become the kernel that eval scores. Three real bugs have
lived here — KGEN-1 (first vs last block), KGEN-2 (unterminated final block) and KGEN-3
(a stray closing fence swallowing the real block), all fixed, all "wrong element chosen
out of a completion" — so this file carries the regression corpus for that whole class.
"""

from __future__ import annotations

import ast

import pytest

from kernel_gen.core.text import extract_code_block, parse_int_spec, problem_id_from_name


def test_extract_code_block_takes_the_fenced_block():
    raw = "## Plan\nsome prose\n```python\nimport torch\n```\ntrailing chatter"
    assert extract_code_block(raw) == "import torch"


def test_extract_code_block_falls_back_to_the_whole_text():
    assert extract_code_block("import torch") == "import torch"


def test_extract_code_block_dedents_an_indented_fence():
    # The fence is nested under an explanation, so the whole block is indented. A bare
    # `\s*` capture would eat only the first line's indent and leave the rest -> an
    # IndentationError on line 2. The fix requires a newline then dedents.
    raw = (
        "Here is the solution:\n\n"
        "    ```python\n"
        "    import torch\n"
        "    import triton\n"
        "    ```\n"
    )
    out = extract_code_block(raw)
    assert out == "import torch\nimport triton"
    ast.parse(out)


def test_extract_code_block_accepts_py_and_bare_language_tags():
    assert extract_code_block("```py\nimport torch\n```") == "import torch"
    assert extract_code_block("```\nimport torch\n```") == "import torch"


def test_extract_code_block_drops_leading_prose_when_there_is_no_fence():
    # The model emitted a plan and then code with no ```python fence; the old fallback
    # returned the whole thing (prose is not valid Python). We resume at the first
    # statement, but only because that makes it parse.
    raw = "## Plan\n1. Analyze the model.\n2. Write it.\n\nimport torch\n@triton.jit\ndef k():\n    pass\n"
    out = extract_code_block(raw)
    assert out.startswith("import torch")
    ast.parse(out)


def test_extract_code_block_keeps_a_leading_comment_in_bare_code():
    # Regression guard: a fence-less file that merely opens with a comment already
    # parses, so the prose-skip must NOT fire and drop the comment.
    raw = "# my fused kernel\nimport torch\n"
    assert extract_code_block(raw) == "# my fused kernel\nimport torch"


def test_parse_int_spec():
    assert parse_int_spec("23") == [23]
    assert parse_int_spec("1-4") == [1, 2, 3, 4]
    assert parse_int_spec("1,5, 10") == [1, 5, 10]


def test_problem_id_is_the_name_prefix_not_the_index():
    # The HF split is ordered lexicographically: index 1 is problem 10.
    assert problem_id_from_name("10_Matmul.py", fallback=1) == 10
    assert problem_id_from_name("no_prefix.py", fallback=7) == 7


# -- extract_code_block: the model revises in the open (KGEN-1, fixed) ------
#
# Reasoning models do not answer once. They write a kernel, notice something ("Wait,
# tl.dot expects..."), and write it again -- so a completion holds several fenced
# blocks and only the last is the answer. Taking the first shipped the model's first
# draft, and sometimes an illustrative fragment that came before it, for every run this
# repo has ever done. Fixtures below are reduced from real Qwen3.6-27B completions.

REVISION = '''\
Here is my first attempt.

```python
import torch


class ModelNew(torch.nn.Module):
    def forward(self, x):
        return x  # first draft
```

Wait, `tl.dot` expects `a` to be `(BLOCK_K, 1)`. Let me fix that.

```python
import torch


class ModelNew(torch.nn.Module):
    def forward(self, x):
        return x * 2  # revised
```
'''


def test_the_last_complete_block_is_the_answer_not_the_first():
    assert "revised" in extract_code_block(REVISION)
    assert "first draft" not in extract_code_block(REVISION)


def test_a_leading_fragment_never_wins_over_a_complete_file():
    # The worst observed case: an interrupted snippet comes first, so the run shipped a
    # loop body with no imports and no class, which scores zero at eval.
    raw = (
        "Sketching the inner loop:\n\n"
        "```python\n    acc = tl.zeros((1, BLOCK_L), dtype=tl.float32)\n```\n\n"
        "Now the whole file:\n\n"
        "```python\nimport torch\n\n\nclass ModelNew(torch.nn.Module):\n    pass\n```\n"
    )
    out = extract_code_block(raw)

    assert out.startswith("import torch")
    assert "class ModelNew" in out


def test_a_later_block_that_does_not_parse_cannot_displace_a_good_one():
    raw = (
        "```python\nimport torch\n\n\nclass ModelNew:\n    pass\n```\n"
        "and a broken afterthought:\n"
        "```python\nclass ModelNew(nn.Module:\n```\n"
    )
    assert extract_code_block(raw) == "import torch\n\n\nclass ModelNew:\n    pass"


def test_a_later_block_without_the_entry_class_cannot_displace_the_submission():
    # Models often close with a usage example or a benchmark snippet. It parses, but it
    # is not the file being submitted.
    raw = (
        "```python\nimport torch\n\n\nclass ModelNew:\n    pass\n```\n"
        "Usage:\n"
        "```python\nm = ModelNew()\nprint(m)\n```\n"
    )
    assert "class ModelNew" in extract_code_block(raw)


def test_a_single_block_is_returned_exactly_as_before():
    # 2 of the 14 traced completions had one block; those must not move at all.
    raw = "```python\nimport torch\n\n\nclass ModelNew:\n    pass\n```"
    assert extract_code_block(raw) == "import torch\n\n\nclass ModelNew:\n    pass"


def test_when_nothing_parses_the_first_block_is_still_returned():
    # The historical fallback, kept deliberately: a completion where every block is
    # broken must degrade the way it always did, not start returning a different one.
    raw = "```python\nclass A(nn.Module:\n```\ntext\n```python\nclass B(nn.Module:\n```"
    assert extract_code_block(raw) == "class A(nn.Module:"


# -- KGEN-2 (fixed, audit-kernel-gen) --------------------------------------
#
# The unterminated final block. KGEN-1's fix ranked only CLOSED fenced blocks, so a
# final block cut off by max_tokens (no closing ```) was invisible, and extraction fell
# back to an earlier complete block -- a superseded draft, or a bare fragment. Measured
# at 83 of 3416 real Qwen3.6 completions before the fix. extract_code_block now recovers
# the tail; these tests pin that it stays recovered. See KERNEL_GEN_BUGS.md.

# The mechanism of the worst real case (level_1_problem_13_sample_5): two closed
# fragment blocks, then the real ModelNew as an UNTERMINATED final block.
UNTERMINATED_FINAL = (
    "Sketch of the inner loop:\n\n"
    "```python\na = tl.load(A + offs, mask=m)\n```\n\n"
    "Now the complete solution:\n\n"
    "```python\nimport torch\nimport torch.nn as nn\n\n\n"
    "class ModelNew(nn.Module):\n    def forward(self, x):\n        return x\n"
)  # note: no closing ``` — the model hit max_tokens mid-answer


def test_extract_recovers_an_unterminated_final_modelnew_block():
    out = extract_code_block(UNTERMINATED_FINAL)
    assert "class ModelNew" in out
    assert out.startswith("import torch")


def test_a_CLOSED_final_block_is_already_recovered():
    # The passing control that pins the boundary: identical completion with the final
    # block CLOSED is handled correctly today. Proves KGEN-2 is about the missing
    # closing fence, not block selection in general.
    out = extract_code_block(UNTERMINATED_FINAL + "```\n")
    assert "class ModelNew" in out
    assert out.startswith("import torch")


# -- KGEN-3 (fixed, audit-kernel-gen) --------------------------------------
#
# _FENCE = r"```(?:python|py)?[ \t]*\r?\n(.*?)```" cannot distinguish an OPENING fence
# from a CLOSING one -- a bare ```\n reads as an opener either way. When a lone/stray
# closing fence sits before the real block (the model closes an in-reasoning code example
# with ```\n, or writes ```\n</think>), the non-greedy (.*?) pairs that stray closer with
# the REAL block's ```python opener: it captures the inter-block prose as a "block" and
# CONSUMES the real opener, so the model's complete ModelNew is never seen as a block.
# KGEN-1 and KGEN-2 both operate on _FENCE's output, so neither addresses this -- the
# mis-pairing happens upstream, in _FENCE itself. Measured at >=12 of 6982 real Qwen3.6
# completions (level 1+2) where a complete, parseable ModelNew is dropped for prose.
# Real proof of the same bug on real data: test_golden_corpus.py, category kgen3_broken.

# A lone closing fence (the model closed an in-reasoning example) precedes the real,
# COMPLETE ModelNew block. Mirrors the real "...code.\n```\n</think>\n\n```python..." shape.
STRAY_CLOSING_FENCE = (
    "Let me sketch the kernel first.\n"
    "```\n"                                   # <-- stray closer: an in-reasoning example ending
    "</think>\n\n"
    "```python\n"
    "import torch\nimport torch.nn as nn\n\n\n"
    "class ModelNew(nn.Module):\n    def forward(self, x):\n        return x\n"
    "```\n"                                   # the real block IS properly closed
)


def test_a_stray_closing_fence_does_not_swallow_the_real_block():
    # KGEN-3, fixed: _fenced_blocks ignores a bare ``` seen outside a block (a stray
    # closer), so the real ```python block after it is opened and captured correctly.
    out = extract_code_block(STRAY_CLOSING_FENCE)
    assert "class ModelNew" in out
    assert out.startswith("import torch")


def test_the_same_completion_without_the_stray_fence_extracts_correctly():
    # The passing control: remove ONLY the stray closer, and extraction is correct today.
    # Proves KGEN-3 is about fence pairing, not about the block itself.
    well_formed = STRAY_CLOSING_FENCE.replace("```\n</think>\n\n", "</think>\n\n", 1)
    out = extract_code_block(well_formed)
    assert "class ModelNew" in out
    assert out.startswith("import torch")


def test_back_to_back_openers_close_the_first_and_open_the_next():
    # A ```python with no bare close before the next ```python: the open/close walk closes
    # the draft and opens the real block. The old non-greedy regex could not express this
    # (its close was the next ```, which was the second opener's own backticks).
    raw = (
        "```python\nx = tl.load(p)  # draft, never closed\n"
        "```python\nimport torch\n\n\nclass ModelNew:\n    pass\n```\n"
    )
    out = extract_code_block(raw)
    assert out == "import torch\n\n\nclass ModelNew:\n    pass"


def test_a_stray_fence_before_an_unterminated_tail_still_recovers_it():
    # KGEN-3 and KGEN-2 compounded: a stray closer, then the real ModelNew as the cut-off
    # final block. The stray ``` is ignored, ```python opens, and end-of-text closes it.
    raw = (
        "Sketch:\n```\n</think>\n\n"
        "```python\nimport torch\n\n\nclass ModelNew:\n"
        "    def forward(self, x):\n        return x\n"
    )  # no closing fence -- max_tokens hit mid-answer
    out = extract_code_block(raw)
    assert "class ModelNew" in out
    assert out.startswith("import torch")
