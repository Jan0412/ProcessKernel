"""``data.labels``: the shared label math, and the baseline parse both builders now import."""

from __future__ import annotations

import json

import pytest

from reranker.src.data.labels import code_hash, compute_label, load_baseline_times, speed_p

TIMING = {
    "level1": {
        "1_Square_matrix_multiplication_.py": {"mean": 0.172, "std": 0.0007, "min": 0.171},
        "23_Softmax.py": {"mean": 2, "min": 1},
    },
    "level6": {"0_SumAggregator.py": {"mean": 0.0156, "min": 0.015}},
}


def write(tmp_path, payload):
    path = tmp_path / "baseline_time_torch.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_levels_and_problem_ids_come_out_as_ints(tmp_path):
    times = load_baseline_times(write(tmp_path, TIMING))
    assert times == {
        1: {1: {"mean": 0.172, "min": 0.171}, 23: {"mean": 2.0, "min": 1.0}},
        6: {0: {"mean": 0.0156, "min": 0.015}},
    }
    assert all(isinstance(v, float) for p in times.values() for s in p.values() for v in s.values())


def test_a_missing_file_is_not_an_error(tmp_path, capsys):
    assert load_baseline_times(str(tmp_path / "nope.json")) == {}
    assert "not found" in capsys.readouterr().out


def test_an_entry_without_a_min_keeps_only_its_mean(tmp_path):
    times = load_baseline_times(write(tmp_path, {"level1": {"1_A.py": {"mean": 0.5}}}))
    assert times == {1: {1: {"mean": 0.5}}}


@pytest.mark.parametrize(
    "payload",
    [
        {"hardware": {"1_A.py": {"mean": 0.5}}},  # not a level key
        {"level1x": {"1_A.py": {"mean": 0.5}}},
        {"level1": "not-a-dict"},
    ],
)
def test_keys_that_are_not_a_level_of_problems_are_skipped(tmp_path, payload):
    assert load_baseline_times(write(tmp_path, payload)) == {}


@pytest.mark.parametrize(
    "problems",
    [
        {"Square_matrix.py": {"mean": 0.5}},  # no integer prefix
        {"1_A.py": {"mean": None}},
        {"1_A.py": "not-a-dict"},
    ],
)
def test_problems_that_cannot_be_keyed_or_timed_are_skipped(tmp_path, problems):
    assert load_baseline_times(write(tmp_path, {"level1": problems})) == {1: {}}


def test_speed_p_maps_the_log_scale_onto_zero_to_one():
    assert speed_p(0.25, 0.25, 4.0) == 0.0
    assert speed_p(1.0, 0.25, 4.0) == pytest.approx(0.5)
    assert speed_p(4.0, 0.25, 4.0) == 1.0
    assert speed_p(100.0, 0.25, 4.0) == 1.0
    assert speed_p(1.0, 0.25, 4.0, quant=0.25) == 0.5


@pytest.mark.parametrize(
    ("compiled", "correct", "expected"),
    [(True, True, (1, "compiled_and_correct")), (True, False, (0, "incorrect")),
     (False, False, (0, "not_compiled"))],
)
def test_only_a_compiled_and_correct_kernel_is_positive(compiled, correct, expected):
    result = compute_label(compiled=compiled, correct=correct)
    assert (result.label, result.reason) == expected


def test_code_hash_is_stable_and_content_addressed():
    assert code_hash("x = 1") == code_hash("x = 1") != code_hash("x = 2")
    assert code_hash("\udcff") == code_hash("")  # undecodable input hashes, never raises
