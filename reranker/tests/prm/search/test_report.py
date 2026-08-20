"""``report``: what the pool's verdicts say about the ORM's pick and the PRM's pruning."""

from __future__ import annotations

import json

from reranker.src.prm.search import report


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
