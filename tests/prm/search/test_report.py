"""``report``: what the pool's verdicts say about the ORM's pick and the PRM's pruning."""

from __future__ import annotations

import json

import pytest

from processkernel.prm.search import report


def tree(tmp_path, level, pid, pool):
    body = {"level": level, "problem_id": pid, "name": f"{pid}_T.py", "steps": [],
            "pool": pool, "picked": pool[0] if pool else None}
    d = tmp_path / "trees"
    d.mkdir(exist_ok=True)
    (d / f"problem_{level}_{pid}.json").write_text(json.dumps(body))
    return str(d)


def entry(cid, sample_id, orm, prm, stop="eos"):
    return {"cid": cid, "sample_id": sample_id, "orm_score": orm, "prm_scores": [prm],
            "stop": stop, "tokens": 100}


def test_a_pick_that_is_correct_leaves_no_selection_regret(tmp_path):
    d = tree(tmp_path, 6, 1, [entry("a", 0, 2.0, 0.9), entry("b", 1, 1.0, 0.1)])
    out = report.report(d, {(1, 0): {"compiled": True, "correctness": True},
                            (1, 1): {"compiled": True, "correctness": False}})
    assert out["overall"]["picked_correct"] == 1.0
    assert out["overall"]["selection_regret"] == 0.0


def test_a_correct_kernel_the_orm_ranked_second_shows_up_as_regret(tmp_path):
    d = tree(tmp_path, 6, 1, [entry("a", 0, 2.0, 0.9), entry("b", 1, 1.0, 0.1)])
    out = report.report(d, {(1, 0): {"compiled": True, "correctness": False},
                            (1, 1): {"compiled": True, "correctness": True}})
    assert out["overall"]["picked_correct"] == 0.0
    assert out["overall"]["oracle_correct"] == 1.0
    assert out["overall"]["selection_regret"] == 1.0


def test_a_pool_with_nothing_correct_has_no_regret_either(tmp_path):
    d = tree(tmp_path, 6, 1, [entry("a", 0, 2.0, 0.9)])
    out = report.report(d, {(1, 0): {"compiled": True, "correctness": False}})
    assert out["overall"]["oracle_correct"] == 0.0
    assert out["overall"]["selection_regret"] == 0.0


def test_prm_top1_hit_reads_the_highest_final_prm_score_not_the_orm(tmp_path):
    d = tree(tmp_path, 6, 1, [entry("a", 0, 9.0, 0.1), entry("b", 1, 0.0, 0.9)])
    out = report.report(d, {(1, 0): {"compiled": True, "correctness": False},
                            (1, 1): {"compiled": True, "correctness": True}})
    assert out["overall"]["prm_top1_hit"] == 1.0
    assert out["overall"]["picked_correct"] == 0.0


def test_levels_are_reported_separately_and_together(tmp_path):
    tree(tmp_path, 6, 1, [entry("a", 0, 1.0, 0.5)])
    d = tree(tmp_path, 1, 2, [entry("b", 0, 1.0, 0.5)])
    out = report.report(d, {(1, 0): {"compiled": True, "correctness": True},
                            (2, 0): {"compiled": True, "correctness": False}})
    assert set(out["by_level"]) == {1, 6}
    assert out["overall"]["picked_correct"] == 0.5


def test_stop_reasons_are_counted(tmp_path):
    d = tree(tmp_path, 6, 1, [entry("a", 0, 1.0, 0.5), entry("b", 1, 0.5, 0.4, stop="steps")])
    out = report.report(d, {(1, 0): {"compiled": True, "correctness": True},
                            (1, 1): {"compiled": True, "correctness": False}})
    assert out["overall"]["stop_reasons"] == {"eos": 1, "steps": 1}


def test_a_candidate_with_no_verdict_is_counted_not_silently_dropped(tmp_path):
    d = tree(tmp_path, 6, 1, [entry("a", 0, 1.0, 0.5), entry("b", 1, 0.5, 0.4)])
    out = report.report(d, {(1, 0): {"compiled": True, "correctness": True}})
    assert out["overall"]["n_missing_verdict"] == 1


def verdict(ok, runtime=1.0):
    return {"compiled": True, "correctness": ok, "runtime": runtime}


def base(level, pid, mean):
    return {f"level{level}": {f"{pid}_T.py": {"mean": mean}}}


def test_without_a_baseline_there_is_no_speed_block(tmp_path):
    d = tree(tmp_path, 6, 1, [entry("a", 0, 2.0, 0.9)])
    out = report.report(d, {(1, 0): verdict(True)})
    assert "speed" not in out["overall"]
    assert "n_bad_runtime" not in out["overall"]


def test_each_selector_is_scored_on_its_own_kernel(tmp_path):
    # picked is pool[0]; "c" has the top prm score; "b" is the fastest correct one.
    d = tree(tmp_path, 6, 1, [entry("a", 0, 9.0, 0.1), entry("b", 1, 0.0, 0.2),
                              entry("c", 2, 0.0, 0.9)])
    out = report.report(d, {(1, 0): verdict(True, 10.0), (1, 1): verdict(True, 2.0),
                            (1, 2): verdict(True, 5.0)}, base(6, 1, 10.0))
    speed = out["overall"]["speed"]
    assert speed["orm"]["geo_mean_speedup"] == 1.0
    assert speed["prm_top1"]["geo_mean_speedup"] == 2.0
    assert speed["oracle"]["geo_mean_speedup"] == pytest.approx(5.0)


def test_the_speed_oracle_ignores_a_faster_kernel_that_is_wrong(tmp_path):
    d = tree(tmp_path, 6, 1, [entry("a", 0, 9.0, 0.9), entry("b", 1, 0.0, 0.1)])
    out = report.report(d, {(1, 0): verdict(True, 10.0), (1, 1): verdict(False, 1.0)},
                        base(6, 1, 10.0))
    assert out["overall"]["speed"]["oracle"]["geo_mean_speedup"] == 1.0


def test_a_problem_with_no_baseline_leaves_the_speed_denominator(tmp_path):
    tree(tmp_path, 6, 1, [entry("a", 0, 1.0, 0.5)])
    d = tree(tmp_path, 6, 2, [entry("b", 0, 1.0, 0.5)])
    out = report.report(d, {(1, 0): verdict(True, 5.0), (2, 0): verdict(True, 5.0)},
                        base(6, 1, 10.0))
    speed = out["overall"]["speed"]
    assert out["overall"]["n_problems"] == 2
    assert (speed["n"], speed["n_missing_baseline"]) == (1, 1)
    assert speed["orm"]["correctness_rate"] == 1.0


def test_fast_p_is_strictly_greater_than_p_over_every_problem(tmp_path):
    tree(tmp_path, 6, 1, [entry("a", 0, 1.0, 0.5)])
    d = tree(tmp_path, 6, 2, [entry("b", 0, 1.0, 0.5)])
    out = report.report(d, {(1, 0): verdict(True, 10.0), (2, 0): verdict(False, -1.0)},
                        {"level6": {"1_T.py": {"mean": 10.0}, "2_T.py": {"mean": 10.0}}})
    orm = out["overall"]["speed"]["orm"]
    # One correct kernel at exactly 1.0x, over two problems.
    assert orm["fast_p"]["0.0"] == 0.5
    assert orm["fast_p"]["1.0"] == 0.0


def test_the_mean_speedup_is_geometric_not_arithmetic(tmp_path):
    tree(tmp_path, 6, 1, [entry("a", 0, 1.0, 0.5)])
    d = tree(tmp_path, 6, 2, [entry("b", 0, 1.0, 0.5)])
    out = report.report(d, {(1, 0): verdict(True, 10.0), (2, 0): verdict(True, 2.5)},
                        {"level6": {"1_T.py": {"mean": 10.0}, "2_T.py": {"mean": 10.0}}})
    # 1.0x and 4.0x: geometric 2.0, arithmetic would be 2.5.
    assert out["overall"]["speed"]["orm"]["geo_mean_speedup"] == 2.0


def test_a_correct_kernel_with_no_runtime_is_counted_not_scored(tmp_path):
    d = tree(tmp_path, 6, 1, [entry("a", 0, 1.0, 0.5)])
    out = report.report(d, {(1, 0): verdict(True, -1.0)}, base(6, 1, 10.0))
    assert out["overall"]["n_bad_runtime"] == 1
    assert out["overall"]["speed"]["orm"]["correct_count"] == 0
    assert out["overall"]["picked_correct"] == 1.0


def test_speed_is_reported_per_level_too(tmp_path):
    tree(tmp_path, 6, 1, [entry("a", 0, 1.0, 0.5)])
    d = tree(tmp_path, 1, 2, [entry("b", 0, 1.0, 0.5)])
    out = report.report(d, {(1, 0): verdict(True, 5.0), (2, 0): verdict(True, 1.0)},
                        {"level6": {"1_T.py": {"mean": 10.0}},
                         "level1": {"2_T.py": {"mean": 10.0}}})
    assert out["by_level"][6]["speed"]["oracle"]["geo_mean_speedup"] == 2.0
    assert out["by_level"][1]["speed"]["oracle"]["geo_mean_speedup"] == pytest.approx(10.0)
