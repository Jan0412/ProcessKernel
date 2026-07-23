"""The critic's full findings survive, and the journal does not notice.

Two claims, and the second is the reason the first is safe to make. ``lint_loop.jsonl``
is read end-to-end by ``--skip-existing`` before every resumed run over a directory that
will hold 160k slots, so anything added to a ``Review``'s serialized form is paid for on
every restart forever. The line numbers are worth keeping and not worth putting there.
"""

from __future__ import annotations

from kernel_gen.core.critics import lint_critic
from kernel_gen.core.model import Attempt, Problem, Review, Trajectory

from conftest import DEAD_KERNEL_FILE, GOOD_KERNEL_FILE

PROBLEM = Problem(level=1, problem_id=1, name="1_Add.py", ref_arch_src=GOOD_KERNEL_FILE)


def test_the_critic_keeps_every_finding_with_its_line_number():
    review = lint_critic()(PROBLEM, DEAD_KERNEL_FILE, set())

    assert not review.clean
    assert {f["check_id"] for f in review.findings} >= {"F1.2"}
    dead = next(f for f in review.findings if f["check_id"] == "F1.2")
    assert dead["data"]["lineno"] > 0
    assert dead["severity"] == "fail"
    assert dead["message"]  # the prose, not just the id


def test_the_line_number_points_at_the_offending_line_of_the_kernel():
    # The whole value of keeping it: a finding is joinable to a source line, and from
    # there to the tokens that produced it.
    review = lint_critic()(PROBLEM, DEAD_KERNEL_FILE, set())
    dead = next(f for f in review.findings if f["check_id"] == "F1.2")

    line = DEAD_KERNEL_FILE.splitlines()[dead["data"]["lineno"] - 1]
    assert "add_kernel" in line


def test_findings_stay_out_of_the_serialized_review():
    review = lint_critic()(PROBLEM, DEAD_KERNEL_FILE, set())
    record = review.to_dict()

    assert review.findings  # captured ...
    assert "findings" not in record  # ... and not journaled
    assert set(record) == {"clean", "parse_status", "n_fail", "n_warn", "check_ids"}


def test_the_journal_record_for_a_whole_slot_is_unchanged_in_shape():
    review = lint_critic()(PROBLEM, DEAD_KERNEL_FILE, set())
    traj = Trajectory(problem=PROBLEM, sample_id=0)
    traj.attempts.append(Attempt(round=0, raw="raw", code=DEAD_KERNEL_FILE, review=review))

    record = traj.to_dict()
    assert set(record["rounds"][0]) == {
        "round",
        "n_chars",
        "clean",
        "parse_status",
        "n_fail",
        "n_warn",
        "check_ids",
    }


def test_a_clean_review_reports_no_findings():
    review = lint_critic()(PROBLEM, GOOD_KERNEL_FILE, set())
    assert review.clean
    assert review.findings == []


def test_review_and_attempt_default_to_no_findings_and_no_trace():
    # Every existing construction site passes neither; both must stay optional.
    assert Review(text="", clean=True).findings == []
    assert Attempt(round=0, raw="", code="").trace is None
