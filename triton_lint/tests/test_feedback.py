"""The A3/A4 feedback blocks.

These two strings ARE the experiment: the arms are identical in every other respect, so the
difference between them is the independent variable. The tests below pin the properties that
make the comparison valid -- above all, that A3 never sees tuning information (that would
contaminate the control) and that A4 always carries the anti-hardcoding instruction (without
it, the model copies the winning constant and the whole round-2 pass reduces to something the
sweep already did for free).
"""

from __future__ import annotations

import pytest

from autotune.feedback import build_feedback, build_timing_feedback, build_tuning_feedback, select_seed

PEAKED = {  # a kernel where the knobs matter a lot
    "kernel": "level_1_problem_19_sample_3", "problem_id": 19, "sample_id": 3,
    "identity_ms": 0.0662, "best_ms": 0.0410, "tuning_gain": 1.615,
    "best_config": {"BLOCK_SIZE": 1024, "num_warps": 8}, "best_config_id": 23,
    "at_grid_edge": True, "n_configs": 25, "n_correct": 22, "n_wrong_result": 2,
    "identity_config": {"BLOCK_SIZE": 128, "num_warps": "default"},
    "table": [
        {"config": {"BLOCK_SIZE": 1024, "num_warps": 8}, "runtime_ms": 0.0410, "config_id": 23, "status": "ok"},
        {"config": {}, "runtime_ms": 0.0662, "config_id": 0, "status": "ok"},
        {"config": {"BLOCK_SIZE": 64, "num_warps": 1}, "runtime_ms": 0.0981, "config_id": 1, "status": "ok"},
        {"config": {"BLOCK_SIZE": 2048, "num_warps": 8}, "runtime_ms": None, "config_id": 24, "status": "wrong_result"},
    ],
}

FLAT = {  # a kernel where they do not
    "kernel": "level_1_problem_5_sample_0", "problem_id": 5, "sample_id": 0,
    "identity_ms": 1.000, "best_ms": 0.990, "tuning_gain": 1.010,
    "best_config": {"BLOCK_SIZE": 256, "num_warps": 4}, "best_config_id": 9,
    "at_grid_edge": False, "n_configs": 25, "n_correct": 25, "n_wrong_result": 0,
    "identity_config": {"BLOCK_SIZE": 512, "num_warps": 4},
    "table": [
        {"config": {"BLOCK_SIZE": 256, "num_warps": 4}, "runtime_ms": 0.990, "config_id": 9, "status": "ok"},
        {"config": {}, "runtime_ms": 1.000, "config_id": 0, "status": "ok"},
    ],
}

BASELINE_MS = 0.05


class TestSeedSelection:
    def test_picks_the_lowest_tuned_runtime(self):
        summary = {
            "k_a": {"problem_id": 7, "best_ms": 0.9, "identity_ms": 1.0},
            "k_b": {"problem_id": 7, "best_ms": 0.4, "identity_ms": 0.5},  # winner
            "k_c": {"problem_id": 8, "best_ms": 0.1, "identity_ms": 0.2},  # other problem
        }
        assert select_seed(summary, 7)["kernel"] == "k_b"

    def test_no_correct_sample_yields_none(self):
        assert select_seed({"k": {"problem_id": 7, "best_ms": None}}, 7) is None
        assert select_seed({}, 7) is None


class TestA3Control:
    def test_reports_runtime_and_baseline(self):
        text = build_timing_feedback(PEAKED, BASELINE_MS)
        assert "0.0662 ms" in text          # the AS-GENERATED time, not the tuned one
        assert f"{BASELINE_MS:.4f} ms" in text
        assert "CORRECT" in text

    @pytest.mark.parametrize("leak", [
        "0.0410",       # the tuned runtime
        "BLOCK_SIZE",   # any config name
        "num_warps",
        "sweep", "swept", "tuned", "tuning", "configuration",
    ])
    def test_no_tuning_information_leaks_into_the_control(self, leak):
        # If A3 saw any of this, A4-vs-A3 would no longer isolate the tuning signal.
        assert leak.lower() not in build_timing_feedback(PEAKED, BASELINE_MS).lower()


class TestA4Proposal:
    def test_contains_the_full_table_not_just_the_winner(self):
        text = build_tuning_feedback(PEAKED, BASELINE_MS)
        assert "0.0410" in text and "0.0662" in text and "0.0981" in text
        assert "WRONG RESULT" in text
        assert "1024" in text and "2048" in text

    def test_marks_the_models_own_config(self):
        assert "YOUR ORIGINAL CONSTANTS" in build_tuning_feedback(PEAKED, BASELINE_MS)

    def test_reports_the_gain(self):
        assert "1.61x faster than your own constants" in build_tuning_feedback(PEAKED, BASELINE_MS)

    def test_anti_hardcoding_instruction_is_present(self):
        # Without this the model pastes the winning constant back and the round-2 pass
        # reproduces what the sweep already did -- an expensive identity function.
        text = build_tuning_feedback(PEAKED, BASELINE_MS)
        assert "do NOT hardcode" in text
        assert "tl.constexpr" in text
        assert "STRUCTURE" in text

    def test_grid_edge_is_called_out(self):
        assert "EDGE" in build_tuning_feedback(PEAKED, BASELINE_MS)
        assert "EDGE" not in build_tuning_feedback(FLAT, BASELINE_MS)

    def test_broken_correctness_is_called_out_as_a_bug(self):
        text = build_tuning_feedback(PEAKED, BASELINE_MS)
        assert "2 configuration(s) produced a WRONG RESULT" in text
        assert "bounds mask" in text
        assert "WRONG RESULT" not in build_tuning_feedback(FLAT, BASELINE_MS).replace(
            "| latency", ""
        ).split("IMPORTANT")[1]  # not in the instructions, only in the table area

    def test_flat_surface_tells_the_model_the_knobs_are_not_the_problem(self):
        # The most valuable message the sweep can produce: no constant will save this
        # kernel, so change the algorithm.
        text = build_tuning_feedback(FLAT, BASELINE_MS)
        assert "NOT your bottleneck" in text
        assert "structural" in text
        assert "NOT your bottleneck" not in build_tuning_feedback(PEAKED, BASELINE_MS)


class TestDispatch:
    def test_arms_produce_different_text(self):
        a3 = build_feedback("timing", PEAKED, BASELINE_MS)
        a4 = build_feedback("tuning", PEAKED, BASELINE_MS)
        assert a3 != a4 and len(a4) > len(a3)

    def test_unknown_arm_raises(self):
        with pytest.raises(ValueError):
            build_feedback("nonsense", PEAKED, BASELINE_MS)
