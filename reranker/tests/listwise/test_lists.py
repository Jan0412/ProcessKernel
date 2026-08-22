"""``listwise.lists``: what a candidate row is narrowed to before the whole corpus is held."""

from __future__ import annotations

import json

from reranker.src.config import RerankerConfig
from reranker.src.data.labels import code_hash
from reranker.src.listwise.lists import _LIST_FIELDS, _read_candidates, build_lists


def row(problem_id, sample_id, *, label=1, speedup_min=1.0, src="kernel", run="runA"):
    return {
        "run_name": run,
        "level": 6,
        "problem_id": problem_id,
        "problem_name": f"{problem_id}_P.py",
        "sample_id": sample_id,
        "ref_arch_src": "class Model(nn.Module): ...",
        "kernel_src": src,
        "compiled": True,
        "correct": bool(label),
        "runtime": 1.0,
        "runtime_min": 1.0,
        "runtime_std": 0.0,
        "speedup": speedup_min,
        "speedup_min": speedup_min,
        "label": label,
    }


def written(tmp_path, rows, name="ds.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_the_sources_are_hashed_and_dropped_rather_than_held(tmp_path):
    # The point of the projection: the two big blobs are ~6.7 KB of a ~7 KB row, and holding
    # 132k of them to read nine fields is what the 8 GiB cgroup killed.
    got = _read_candidates(written(tmp_path, [row(1, 0, src="abc")]))
    assert "kernel_src" not in got[0]
    assert "ref_arch_src" not in got[0]
    assert got[0]["code_hash"] == code_hash("abc")


def test_the_projection_keeps_every_field_a_list_is_built_from(tmp_path):
    got = _read_candidates(written(tmp_path, [row(1, 0)]))
    assert set(got[0]) == set(_LIST_FIELDS) | {"code_hash"}
    assert (got[0]["level"], got[0]["problem_id"], got[0]["run_name"]) == (6, 1, "runA")
    assert (got[0]["label"], got[0]["compiled"], got[0]["speedup_min"]) == (1, True, 1.0)


def test_a_blank_line_is_skipped(tmp_path):
    path = tmp_path / "ds.jsonl"
    path.write_text(json.dumps(row(1, 0)) + "\n\n")
    assert len(_read_candidates(str(path))) == 1


def configured(tmp_path, source):
    cfg = RerankerConfig()
    cfg.data.dataset_jsonl = source
    cfg.listwise.lists_train_jsonl = str(tmp_path / "tr.jsonl")
    cfg.listwise.lists_val_jsonl = str(tmp_path / "va.jsonl")
    cfg.listwise.lists_splits_json = str(tmp_path / "sp.json")
    cfg.listwise.split_ratios = [1.0, 0.0]
    cfg.listwise.speedup_stat = "min"
    return cfg


def test_a_list_still_grades_and_references_its_candidates(tmp_path):
    # End to end over the projection: a fast correct, a slow correct, and a wrong kernel.
    rows = [
        row(1, 0, speedup_min=4.0, src="fast"),
        row(1, 1, speedup_min=0.2, src="slow"),
        row(1, 2, label=0, speedup_min=None, src="wrong"),
    ]
    cfg = configured(tmp_path, written(tmp_path, rows))
    train, _, _ = build_lists(cfg)
    lists = [json.loads(line) for line in open(train)]
    assert len(lists) == 1
    by_sample = {c["sample_id"]: c for c in lists[0]["candidates"]}
    assert by_sample[0]["rel"] == 2.0  # correct at speedup_hi
    assert by_sample[1]["rel"] == 1.0  # correct at speedup_lo
    assert by_sample[2]["rel"] == 0.0  # compiled but wrong
    assert {c["run_name"] for c in lists[0]["candidates"]} == {"runA"}


def test_two_runs_with_identical_code_still_dedup_to_one_candidate(tmp_path):
    rows = [
        row(1, 0, speedup_min=4.0, src="same", run="runA"),
        row(1, 0, speedup_min=4.0, src="same", run="runB"),
        row(1, 1, speedup_min=0.2, src="other", run="runA"),
    ]
    cfg = configured(tmp_path, written(tmp_path, rows))
    train, _, _ = build_lists(cfg)
    lists = [json.loads(line) for line in open(train)]
    assert len(lists[0]["candidates"]) == 2
