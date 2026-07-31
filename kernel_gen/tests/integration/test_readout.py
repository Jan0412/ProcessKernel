"""The A5 readout: the paired transition table.

The one number the whole experiment turns on is ``broken`` -- slots that were CORRECT at
round 0 and that the refinement destroyed. An unpaired correctness rate hides those
completely, which is the entire reason round 0 is generated as part of the same run.
"""

from __future__ import annotations

import json

from kernel_gen.core.artifacts import round_dir, write_config
from kernel_gen.readout import load_slots, transitions

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
