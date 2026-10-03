"""``processkernel.generation.core.model``: the dataclasses and the journal shape.

``to_dict`` is the shape ``generation.jsonl`` and ``--skip-existing`` depend on.
"""

from __future__ import annotations

from processkernel.generation.core.model import Attempt, Problem, Trajectory

PROBLEM = Problem(level=1, problem_id=19, name="19_ReLU.py", ref_arch_src="class Model: pass")


def test_last_is_the_slots_attempt():
    attempt = Attempt(raw="r", code="c")
    assert Trajectory(problem=PROBLEM, sample_id=0, attempts=[attempt]).last is attempt


def test_last_is_none_when_a_slot_never_produced_an_attempt():
    # A slot skipped entirely has nothing to ship; artifacts must be able to tell that
    # apart from "produced empty code".
    assert Trajectory(problem=PROBLEM, sample_id=0).last is None


def test_to_dict_carries_the_slot_key_and_the_code_length():
    traj = Trajectory(problem=PROBLEM, sample_id=3, attempts=[Attempt(raw="r", code="abc")])
    assert traj.to_dict() == {
        "level": 1,
        "problem_id": 19,
        "sample_id": 3,
        "problem_name": "19_ReLU.py",
        "n_chars": 3,
    }


def test_to_dict_of_a_slot_without_an_attempt_still_names_the_slot():
    record = Trajectory(problem=PROBLEM, sample_id=1).to_dict()
    assert (record["problem_id"], record["sample_id"]) == (19, 1)
    assert "n_chars" not in record


# -- prompt capture: for the trace, not for the journal --------------------


def test_attempt_captures_the_prompt_but_keeps_it_out_of_the_journal():
    # The exact user turn the model saw is captured so a trace can reconstruct the whole
    # conversation. But to_dict feeds generation.jsonl, which --skip-existing reads
    # start-to-finish before every resumed run, so prompt must NOT appear there -- exactly
    # like trace.
    assert Attempt(raw="r", code="c").prompt == ""
    attempt = Attempt(raw="r", code="c", prompt="solve problem 19")
    assert attempt.prompt == "solve problem 19"
    assert "prompt" not in attempt.to_dict()
