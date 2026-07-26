"""``kernel_gen.core.text``: pull the kernel out of a completion, parse specs and ids.

Extraction is the single highest-stakes pure function in the pipeline: it decides which
bytes of a free-form completion become the kernel that eval scores. Four real bugs have
lived here — KGEN-1 (first vs last block), KGEN-2 (unterminated final block), KGEN-3 (a
stray closing fence swallowing the real block) and KGEN-9 (an empty block outranking an
unfenced answer), all fixed, all "wrong element chosen out of a completion" — so this
file carries the regression corpus for that whole class.
"""

from __future__ import annotations

import ast

import pytest

from kernel_gen.core.text import (
    _largest_parseable_prefix,
    extract_code_block,
    parse_int_spec,
    problem_id_from_name,
)


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


# -- KGEN-9 (fixed, audit) -------------------------------------------------
#
# The two-pass sampler builds `PLAN_PREFIX + plan + CODE_FENCE + code`. When pass 2
# returns nothing, that injected ```python trails the text with no content after it --
# an EMPTY block. `ast.parse("")` succeeds, so it ranked as a valid candidate and won,
# and "" was shipped for a generation that contained a complete kernel. Round 0 is the
# paired baseline, so those empty files were scored as model failures.


def test_an_empty_trailing_block_is_not_a_candidate():
    # The bare mechanism: the model answered without fencing, the sampler appended its
    # fence, and nothing followed it.
    raw = (
        "## Plan\nUse a Triton kernel.\n\n"
        "import torch\n\n\nclass ModelNew:\n    def forward(self, x):\n        return x\n"
        "```python"
    )
    out = extract_code_block(raw)

    assert out != ""
    assert "class ModelNew" in out
    ast.parse(out)


def test_the_injected_fence_does_not_eat_the_line_it_is_glued_to():
    # vLLM re-inserts CODE_FENCE with no separator, so it lands on the model's last line:
    # "        return triton_relu(x)```python". Deleting the marker outright would splice
    # that line onto the next; trimming the line would lose the return statement.
    raw = (
        "## Plan\nplan prose\n\n"
        "import torch\n\n\nclass ModelNew:\n"
        "    def forward(self, x):\n        return triton_relu(x)```python"
    )
    out = extract_code_block(raw)

    assert "return triton_relu(x)" in out
    tree = ast.parse(out)
    model_new = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelNew")
    assert "forward" in [m.name for m in model_new.body if isinstance(m, ast.FunctionDef)]


def test_trailing_commentary_after_the_kernel_is_trimmed_off():
    # An unfenced answer is code followed by prose. The largest parseable prefix stops at
    # the prose, which is what makes the salvage return a file rather than a blob.
    raw = (
        "## Plan\nprose\n\n"
        "import torch\n\n\nclass ModelNew:\n    def forward(self, x):\n        return x\n\n"
        "This kernel fuses the two ops, and should be about 2x faster than eager.\n"
        "```python"
    )
    out = extract_code_block(raw)

    ast.parse(out)
    assert "class ModelNew" in out
    assert "should be about 2x faster" not in out


def test_a_salvage_without_the_entry_class_is_discarded():
    # The salvage is gated on ENTRY_CLASS: prose that happens to start with an import must
    # not be promoted into a "kernel". Degrades to the historical best-effort string.
    raw = "## Plan\n\nimport torch\n\nThen we would tile the loop, but I am out of budget.\n"
    out = extract_code_block(raw)

    assert "class ModelNew" not in out


def test_an_empty_block_does_not_displace_a_real_one():
    # The empty block is dropped as a candidate, not treated as the answer, even when it
    # comes last -- which is exactly the position the ranking prefers.
    raw = (
        "```python\nimport torch\n\n\nclass ModelNew:\n    def forward(self, x):\n"
        "        return x\n```\n\n```python"
    )
    out = extract_code_block(raw)

    assert "class ModelNew" in out
    ast.parse(out)


# -- the salvage helper's own edges ----------------------------------------


def test_the_salvage_gives_up_when_nothing_leading_parses():
    # Trimming reaches line 0 without ever parsing. Must return "" rather than loop or
    # raise -- the caller then falls through to the historical best-effort string.
    assert _largest_parseable_prefix("def f(:\n    ???\n") == ""


def test_the_salvage_rejects_a_source_python_cannot_even_encode():
    # A lone surrogate makes ast.parse raise UnicodeEncodeError -- a ValueError, not a
    # SyntaxError, so it carries no lineno to trim at. Detokenized model output really can
    # contain one, and it must come back as "" rather than escape as an exception.
    assert _largest_parseable_prefix("import torch\n\udcff\n") == ""


def test_the_salvage_gives_up_at_line_zero_rather_than_looping():
    blob = "\n".join(f"line {i} is prose, not python: <{i}>" for i in range(200))
    assert _largest_parseable_prefix(blob, max_trims=3) == ""


def test_the_salvage_is_bounded_by_its_trim_budget():
    # Each trim only steps back to the reported error line, so a file whose errors walk
    # backwards slowly would cost one ast.parse per line. The budget caps that; running
    # out returns "" and the caller degrades to its historical best effort.
    src = "x = 1\nx = 2\nx = 3\nx = 4\ndef f(:\n"  # first error is on the last line
    assert _largest_parseable_prefix(src, max_trims=1) == ""
    assert _largest_parseable_prefix(src, max_trims=2) == "x = 1\nx = 2\nx = 3\nx = 4"


def test_the_trim_jumps_to_the_error_line_rather_than_stepping_one_at_a_time():
    # The error is on line 3 of 6, so ONE trim must land on line 2 -- not walk back a line
    # per iteration. Pinned through the budget: a rule that stepped by one, or that
    # overshot past the error, would need more trims than this allows and return "".
    src = "x = 1\nx = 2\ndef f(:\nx = 4\nx = 5\nx = 6\n"

    assert _largest_parseable_prefix(src, max_trims=2) == "x = 1\nx = 2"
    assert _largest_parseable_prefix(src, max_trims=1) == ""


def test_the_salvage_can_come_down_to_a_single_line():
    # Everything after line 1 is prose. The give-up guard must not fire while one good
    # line is still on the table.
    src = "import torch\n$$$ not python $$$\n"
    assert _largest_parseable_prefix(src) == "import torch"


def test_the_salvage_keeps_the_code_and_drops_the_prose_that_follows():
    src = "import torch\n\n\nclass ModelNew:\n    pass\n\nThen we tile the loop.\n"
    assert _largest_parseable_prefix(src) == "import torch\n\n\nclass ModelNew:\n    pass\n"


def test_a_completion_with_no_python_statement_at_all_is_returned_as_is():
    # No fence, nothing that parses, and no import/def/class to resume at -- the model
    # answered in pure prose. Best effort is the text itself; the salvage never runs.
    raw = "I cannot write this kernel: the reduction axis is not known at compile time.\n"
    assert extract_code_block(raw) == raw.strip()


def test_a_fence_glued_mid_line_becomes_a_break_not_a_splice():
    # Why the no-fence path substitutes a newline rather than deleting the marker. Both
    # fences here yield no block (a stray closer, then a trailing opener with nothing
    # after it), so the whole text goes down the fallback. Deleting the markers would
    # splice "x = 1" onto "y = 2" and the result would not parse.
    raw = "import torch\nx = 1```\ny = 2```python"
    out = extract_code_block(raw)

    ast.parse(out)
    assert "x = 1" in out and "y = 2" in out
    assert "x = 1y = 2" not in out
