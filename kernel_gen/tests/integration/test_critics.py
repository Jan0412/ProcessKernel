"""The critic's full findings survive, and the journal does not notice.

Two claims, and the second is the reason the first is safe to make. ``lint_loop.jsonl``
is read end-to-end by ``--skip-existing`` before every resumed run over a directory that
will hold 160k slots, so anything added to a ``Review``'s serialized form is paid for on
every restart forever. The line numbers are worth keeping and not worth putting there.

The kernel sources come from ``conftest`` fixtures (``good_kernel_file`` /
``dead_kernel_file``) rather than a module import, so this file works from the
``integration/`` subdir without a shared top-level module.
"""

from __future__ import annotations

import pytest

from kernel_gen.core.critics import lint_critic
from kernel_gen.core.model import Attempt, Problem, Review, Trajectory


@pytest.fixture
def problem(good_kernel_file) -> Problem:
    return Problem(level=1, problem_id=1, name="1_Add.py", ref_arch_src=good_kernel_file)


def test_the_critic_keeps_every_finding_with_its_line_number(problem, dead_kernel_file):
    review = lint_critic()(problem, dead_kernel_file, set())

    assert not review.clean
    assert {f["check_id"] for f in review.findings} >= {"F1.2"}
    dead = next(f for f in review.findings if f["check_id"] == "F1.2")
    assert dead["data"]["lineno"] > 0
    assert dead["severity"] == "fail"
    assert dead["message"]  # the prose, not just the id


def test_the_line_number_points_at_the_offending_line_of_the_kernel(problem, dead_kernel_file):
    # The whole value of keeping it: a finding is joinable to a source line, and from
    # there to the tokens that produced it.
    review = lint_critic()(problem, dead_kernel_file, set())
    dead = next(f for f in review.findings if f["check_id"] == "F1.2")

    line = dead_kernel_file.splitlines()[dead["data"]["lineno"] - 1]
    assert "add_kernel" in line


def test_findings_stay_out_of_the_serialized_review(problem, dead_kernel_file):
    review = lint_critic()(problem, dead_kernel_file, set())
    record = review.to_dict()

    assert review.findings  # captured ...
    assert "findings" not in record  # ... and not journaled
    # Two keys have been added since, against a journal that is read end-to-end on every
    # resumed run: submission_ok (KGEN-14), a bool per round, and shown_check_ids
    # (KGEN-17/18), the ids the prompt actually contained -- bounded by max_findings, so
    # at most 8 short strings per round.
    assert set(record) == {
        "clean",
        "parse_status",
        "n_fail",
        "n_warn",
        "check_ids",
        "submission_ok",
        "shown_check_ids",
    }


def test_the_journal_record_for_a_whole_slot_is_unchanged_in_shape(problem, dead_kernel_file):
    review = lint_critic()(problem, dead_kernel_file, set())
    traj = Trajectory(problem=problem, sample_id=0)
    traj.attempts.append(Attempt(round=0, raw="raw", code=dead_kernel_file, review=review))

    record = traj.to_dict()
    assert set(record["rounds"][0]) == {
        "round",
        "n_chars",
        "clean",
        "parse_status",
        "n_fail",
        "n_warn",
        "check_ids",
        "submission_ok",
        "shown_check_ids",  # KGEN-17/18; see the note in the test above
    }


def test_a_clean_review_reports_no_findings(problem, good_kernel_file):
    review = lint_critic()(problem, good_kernel_file, set())
    assert review.clean
    assert review.findings == []


def test_review_and_attempt_default_to_no_findings_and_no_trace():
    # Every existing construction site passes neither; both must stay optional.
    assert Review(text="", clean=True).findings == []
    assert Attempt(round=0, raw="", code="").trace is None


def test_shape_inference_failure_degrades_to_no_shapes_not_a_crash(monkeypatch, dead_kernel_file):
    # Shapes are best-effort (they only enrich F2's byte estimates); findings are not.
    # A reference the shape inferencer cannot read must still produce a lint verdict.
    import checker.lint.shapes as shapes_mod

    def boom(_src):
        raise RuntimeError("cannot infer shapes from this reference")

    monkeypatch.setattr(shapes_mod, "shapes_from_source", boom)
    problem = Problem(level=1, problem_id=2, name="2_X.py", ref_arch_src="not python (((")

    review = lint_critic()(problem, dead_kernel_file, set())
    assert not review.clean  # the linter still ran and still complained
    assert review.findings
