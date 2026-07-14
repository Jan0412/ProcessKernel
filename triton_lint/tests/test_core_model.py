"""kernel_gen.core pure logic: code extraction, id parsing, and the keep-best rule.

``Trajectory.final()`` is the loop's only non-regression guard -- refinement can make
a kernel worse, and this is what stops a run from shipping the regression.
"""

from __future__ import annotations

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
