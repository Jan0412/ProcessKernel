"""kernel_gen.core pure logic: code extraction, id parsing, and the keep-best rule.

``Trajectory.final()`` is the loop's only non-regression guard -- refinement can make
a kernel worse, and this is what stops a run from shipping the regression.
"""

from __future__ import annotations

import ast

from kernel_gen.core.model import Attempt, Problem, Review, Trajectory
from kernel_gen.core.text import extract_code_block, parse_int_spec, problem_id_from_name

PROBLEM = Problem(level=1, problem_id=19, name="19_ReLU.py", ref_arch_src="class Model: pass")


def _attempt(round_: int, *, code="x", clean=False, n_fail=0, n_warn=0, parse_status="ok"):
    return Attempt(
        round=round_,
        raw=code,
        code=code,
        review=Review(
            text="",
            clean=clean,
            data={"n_fail": n_fail, "n_warn": n_warn, "parse_status": parse_status},
        ),
    )


def _traj(*attempts) -> Trajectory:
    return Trajectory(problem=PROBLEM, sample_id=0, attempts=list(attempts))


# -- text ------------------------------------------------------------------


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


# -- final(): the non-regression guard -------------------------------------


def test_final_is_the_first_clean_attempt():
    traj = _traj(
        _attempt(0, n_fail=2),
        _attempt(1, code="clean", clean=True),
        _attempt(2, n_fail=1),  # cannot happen in practice; must not win if it does
    )
    assert traj.final().code == "clean"


def test_final_keeps_the_best_round_when_nothing_ever_goes_clean():
    traj = _traj(
        _attempt(0, code="r0", n_fail=2, n_warn=1),
        _attempt(1, code="r1", n_fail=0, n_warn=3),  # fewer fails wins over fewer warns
        _attempt(2, code="r2", n_fail=1, n_warn=0),
    )
    assert traj.final().code == "r1"


def test_final_never_prefers_an_attempt_that_does_not_parse():
    # A syntactically broken file scores zero at eval no matter how few findings it has.
    traj = _traj(
        _attempt(0, code="r0", n_fail=5, n_warn=5),
        _attempt(1, code="r1", n_fail=0, n_warn=0, parse_status="syntax_error"),
    )
    assert traj.final().code == "r0"


def test_final_breaks_ties_toward_the_earliest_round():
    # A round that changed nothing measurable gets no credit for the previous one's work.
    traj = _traj(_attempt(0, code="r0", n_fail=1), _attempt(1, code="r1", n_fail=1))
    assert traj.final().code == "r0"


def test_final_survives_a_missing_review():
    traj = _traj(Attempt(round=0, raw="", code="", review=None), _attempt(1, code="r1"))
    assert traj.final().code == "r1"  # empty code loses to code that exists


def test_to_dict_carries_the_per_round_history():
    traj = _traj(_attempt(0, n_fail=2), _attempt(1, code="clean", clean=True))
    record = traj.to_dict()
    assert record["problem_id"] == 19
    assert record["sample_id"] == 0
    assert record["final_round"] == 1
    assert record["clean"] is True
    assert [r["round"] for r in record["rounds"]] == [0, 1]
    assert record["rounds"][0]["n_fail"] == 2


# -- extract_code_block: the model revises in the open ---------------------
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
