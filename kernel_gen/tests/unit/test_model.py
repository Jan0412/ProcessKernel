"""``kernel_gen.core.model``: the dataclasses, and the keep-best rule.

``Trajectory.final()`` is the loop's only non-regression guard -- refinement can make a
kernel worse, and this is what stops a run from shipping the regression. ``to_dict`` is
the shape ``lint_loop.jsonl`` and ``--skip-existing`` depend on.
"""

from __future__ import annotations

from kernel_gen.core.model import Attempt, Problem, Review, Trajectory

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


def test_final_is_none_when_a_slot_never_produced_an_attempt():
    # A slot skipped entirely (e.g. it was filtered before generation) has nothing to
    # ship; artifacts must be able to tell that apart from "produced empty code".
    assert Trajectory(problem=PROBLEM, sample_id=0).final() is None


# -- prompt capture: for the trace, not for the journal --------------------


def test_attempt_captures_the_prompt_but_keeps_it_out_of_the_journal():
    # The exact user turn the model saw is captured so a trace can reconstruct the whole
    # conversation. But to_dict feeds lint_loop.jsonl, which --skip-existing reads
    # start-to-finish before every resumed run, so prompt must NOT appear there -- exactly
    # like trace. Default "" keeps every existing Attempt(...) call working unchanged.
    assert Attempt(round=0, raw="r", code="c").prompt == ""
    attempt = Attempt(round=0, raw="r", code="c", prompt="solve problem 19")
    assert attempt.prompt == "solve problem 19"
    assert "prompt" not in attempt.to_dict()
