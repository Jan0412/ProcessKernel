"""The A5 readout: the paired transition table.

The one number the whole experiment turns on is ``broken`` -- slots that were CORRECT at
round 0 and that the refinement destroyed. An unpaired correctness rate hides those
completely, which is the entire reason round 0 is generated as part of the same run.
"""

from __future__ import annotations

import json

from kernel_gen.core.artifacts import round_dir, write_config
from kernel_gen.readout import load_slots, report_lint, transitions

CONFIG = {"model": "m", "level": 1, "num_samples": 2, "run_name": "r"}


def _eval_results(run_dir, outcomes: dict[tuple[int, int], bool]) -> None:
    """outcomes: {(problem_id, sample_id): correct}."""
    results: dict[str, list] = {}
    for (problem_id, sample_id), correct in outcomes.items():
        results.setdefault(str(problem_id), []).append(
            {
                "sample_id": sample_id,
                "compiled": True,
                "correctness": correct,
                "runtime": 1.0 if correct else None,
                "runtime_stats": {"min": 1.0},
                "metadata": {"hardware": "H100"},
            }
        )
    with open(f"{run_dir}/eval_results.json", "w") as fh:
        json.dump(results, fh)


def test_the_transition_table_separates_fixes_from_regressions(tmp_path):
    write_config(str(tmp_path), CONFIG, dataset="kernelbench")

    # Same four slots, evaluated twice. Round 0 gets two right; refinement fixes one of
    # the wrong ones and BREAKS one of the right ones -- a wash that a bare correctness
    # rate (2/4 before, 2/4 after) would report as "no change".
    _eval_results(round_dir(str(tmp_path), 0), {
        (1, 0): True, (1, 1): False, (2, 0): True, (2, 1): False,
    })
    _eval_results(str(tmp_path), {
        (1, 0): True, (1, 1): True, (2, 0): False, (2, 1): False,
    })

    baseline = load_slots(round_dir(str(tmp_path), 0))
    refined = load_slots(str(tmp_path))
    buckets = transitions(baseline, refined)

    assert buckets["fixed"] == [(1, 1)]
    assert buckets["broken"] == [(2, 0)]  # invisible in an unpaired comparison
    assert buckets["kept"] == [(1, 0)]
    assert buckets["neither"] == [(2, 1)]

    n_correct_before = sum(1 for s in baseline.values() if s["correct"])
    n_correct_after = sum(1 for s in refined.values() if s["correct"])
    assert n_correct_before == n_correct_after == 2  # …exactly the trap


def test_slots_missing_from_one_side_are_not_compared(tmp_path):
    write_config(str(tmp_path), CONFIG, dataset="kernelbench")
    _eval_results(round_dir(str(tmp_path), 0), {(1, 0): True, (1, 1): True})
    _eval_results(str(tmp_path), {(1, 0): False})  # sample 1 never got evaluated

    buckets = transitions(
        load_slots(round_dir(str(tmp_path), 0)), load_slots(str(tmp_path))
    )
    assert buckets["broken"] == [(1, 0)]
    assert sum(len(v) for v in buckets.values()) == 1


# -- the mechanism table (KGEN-18) ------------------------------------------
# This table answers "which checks resist repair", and its caption points the reader at
# F1.6 specifically. Attributing a round to a check the model was never shown does not
# just add noise -- it invents the trend the caption tells you to look for.


def _round(r: int, *, shown=None, check_ids=(), clean=False, submission_ok=None,
           parse_status="ok") -> dict:
    """One journal round. `shown` omitted entirely = a pre-gate journal."""
    entry = {
        "round": r, "n_chars": 100, "clean": clean, "parse_status": parse_status,
        "n_fail": len(check_ids), "n_warn": 0, "check_ids": list(check_ids),
    }
    if shown is not None:
        entry["shown_check_ids"] = list(shown)
    if submission_ok is not None:
        entry["submission_ok"] = submission_ok
    return entry


def _traj(key, rounds: list[dict], clean=False) -> dict:
    return {
        "problem_id": key[0], "sample_id": key[1], "n_rounds": len(rounds),
        "final_round": rounds[-1]["round"], "clean": clean, "rounds": rounds,
    }


def _table(trajectories: dict, rounds: int = 3) -> list[dict]:
    return report_lint(trajectories, rounds)["check_counts_by_round"]


def test_the_check_table_reports_a_submission_only_defect():
    """A file that will not parse runs no lint check at all, so it used to land in no row
    while still counting toward the denominator -- 1137 of 8948 real rounds."""
    trajectories = {
        (1, 0): _traj((1, 0), [_round(0, shown=["S1.0"], check_ids=[],
                                      submission_ok=False, parse_status="syntax_error")])
    }
    assert _table(trajectories)[0] == {"S1.0": 1}


def test_a_suppressed_lint_finding_is_not_counted_as_mechanism():
    """The false conclusion, verbatim: 100 slots blocked on S1.0 that simply persist used
    to render F1.6 climbing 10% -> 100% directly under a caption saying F1.6 is the one to
    watch because a loop is exactly the thing that could induce one. Nothing happened."""
    trajectories = {}
    for i in range(90):  # repaired at round 0, having actually been shown F1.2
        trajectories[(1, i)] = _traj(
            (1, i), [_round(0, shown=["F1.2"], check_ids=["F1.2"], submission_ok=True)],
            clean=True,
        )
    for i in range(10):  # gate-blocked, F1.6 fires but is never shown, and persists
        trajectories[(2, i)] = _traj((2, i), [
            _round(r, shown=["S1.0"], check_ids=["F1.6"], submission_ok=False)
            for r in (0, 1)
        ])

    table = _table(trajectories)

    assert "F1.6" not in table[0] and "F1.6" not in table[1]
    assert table[0] == {"F1.2": 90, "S1.0": 10}
    assert table[1] == {"S1.0": 10}


def test_a_pre_gate_journal_is_read_exactly_as_before():
    """No shown_check_ids on the record means the information was never captured. The
    readout must reproduce today's numbers rather than pretend otherwise."""
    trajectories = {
        (1, 0): _traj((1, 0), [_round(0, check_ids=["F1.2", "F1.4"]),
                               _round(1, check_ids=["F1.2"])])
    }
    table = _table(trajectories)

    assert table[0] == {"F1.2": 1, "F1.4": 1}
    assert table[1] == {"F1.2": 1}


def test_both_families_share_one_table():
    trajectories = {
        (1, 0): _traj((1, 0), [_round(0, shown=["S1.3"], submission_ok=False)]),
        (1, 1): _traj((1, 1), [_round(0, shown=["F1.2"], check_ids=["F1.2"],
                                      submission_ok=True)]),
    }
    assert _table(trajectories)[0] == {"S1.3": 1, "F1.2": 1}


def test_a_round_that_showed_nothing_is_counted_in_no_row():
    """An empty prompt is honest: `shown_check_ids: []` means the model was told nothing,
    which must not silently fall back to what fired."""
    trajectories = {
        (1, 0): _traj((1, 0), [_round(0, shown=[], check_ids=["F1.2"], clean=True)],
                      clean=True)
    }
    assert _table(trajectories)[0] == {}


# -- the blocked rate --------------------------------------------------------


def test_the_blocked_rate_is_counted_over_the_rounds_that_recorded_it():
    trajectories = {
        (1, 0): _traj((1, 0), [_round(0, shown=["S1.0"], submission_ok=False)]),
        (1, 1): _traj((1, 1), [_round(0, shown=["F1.2"], check_ids=["F1.2"],
                                      submission_ok=True)]),
        (1, 2): _traj((1, 2), [_round(0, check_ids=["F1.2"])]),  # pre-gate: no flag
    }
    summary = report_lint(trajectories, 3)

    # the denominator is the rounds that carry the flag, not every round
    assert summary["n_blocked_by_round"][0] == 1
    assert summary["n_gated_by_round"][0] == 2


def test_a_pre_gate_journal_prints_no_blocked_line(capsys):
    trajectories = {(1, 0): _traj((1, 0), [_round(0, check_ids=["F1.2"])])}
    summary = report_lint(trajectories, 3)

    assert "blocked" not in capsys.readouterr().out.lower()
    assert summary["n_gated_by_round"] == [0, 0, 0]


def test_rounds_beyond_the_requested_window_are_ignored():
    # `--rounds` narrower than the journal: the extra rounds must not widen the table or
    # the denominators, or the rates silently describe a different window than the header.
    trajectories = {
        (1, 0): _traj((1, 0), [_round(0, shown=["F1.2"], submission_ok=True),
                               _round(1, shown=["F1.4"], submission_ok=False)])
    }
    summary = report_lint(trajectories, 1)

    assert summary["check_counts_by_round"] == [{"F1.2": 1}]
    assert summary["n_gated_by_round"] == [1]


def test_an_empty_trajectory_set_is_still_handled(capsys):
    assert report_lint({}, 3) == {}
    assert "skipping the mechanism section" in capsys.readouterr().out


# -- the caption must not claim more than the journal recorded ---------------
# The same defect one level up: a caption asserting "these are the checks the model was
# shown" is false for a journal that never recorded what was shown.


def test_a_pre_gate_journal_is_captioned_as_raw_findings(capsys):
    report_lint({(1, 0): _traj((1, 0), [_round(0, check_ids=["F1.2"])])}, 3)
    out = capsys.readouterr().out

    assert "predates" in out and "upper bounds" in out
    assert "prompt actually named" not in out


def test_a_post_gate_journal_is_captioned_as_what_was_shown(capsys):
    report_lint(
        {(1, 0): _traj((1, 0), [_round(0, shown=["F1.2"], check_ids=["F1.2"],
                                       submission_ok=True)])},
        3,
    )
    out = capsys.readouterr().out

    assert "prompt actually named" in out
    assert "predates" not in out


# -- E.2: the accounting invariant ------------------------------------------


def test_every_round_that_did_not_go_clean_names_at_least_one_check():
    """The rows must sum to the denominator. The one documented exception is a generation
    that contained no code at all: it fires no check because there is nothing to check --
    2 of 3812 real non-clean rounds, and it must not grow silently."""
    shapes = [
        _round(0, shown=["S1.0"], submission_ok=False, parse_status="syntax_error"),
        _round(0, shown=["S1.3"], check_ids=["F1.2"], submission_ok=False),
        _round(0, shown=["F1.2"], check_ids=["F1.2", "F1.4"], submission_ok=True),
        _round(0, shown=["F2.1"], check_ids=["F2.1"], submission_ok=True),
    ]
    for shape in shapes:
        table = _table({(1, 0): _traj((1, 0), [shape])})
        assert table[0], f"non-clean round attributed to nothing: {shape}"

    empty = _round(0, shown=[], check_ids=[], submission_ok=True, parse_status="empty")
    assert _table({(1, 0): _traj((1, 0), [empty])})[0] == {}
