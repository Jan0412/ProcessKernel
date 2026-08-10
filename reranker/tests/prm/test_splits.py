"""``prm.splits``: one problem-level split over the union of every part (PLAN §10)."""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

from reranker.src.data.splits import load_splits
from reranker.src.prm import build, splits
from reranker.tests.prm.runfixture import attempt, prm_config, verdict, write_round

PROSE = "a\nb\nc\nd\n"


def build_parts(tmp_path, monkeypatch, runs, **over):
    """Build one part per named run, each holding the given problem ids."""
    monkeypatch.setattr(build, "token_counter", lambda name: len)
    dirs = []
    for name, pids in runs.items():
        write_round(
            tmp_path / name,
            "shard_00",
            0,
            [attempt(problem_id=p, raw=PROSE) for p in pids],
            {str(p): [verdict(correctness=False)] for p in pids},
        )
        dirs.append(tmp_path / name)
    cfg = prm_config(tmp_path, dirs, **over)
    build.run_build(cfg)
    return cfg


def assignment(cfg):
    return load_splits(os.path.join(cfg.prm.out_dir, splits.SPLITS))


def test_a_problem_in_both_runs_lands_in_exactly_one_split(tmp_path, monkeypatch):
    # The runs overlap on ids 10..19; a per-run split would put those in one run's train
    # and the other's val, so the key is (level, problem_id) over the union.
    cfg = build_parts(tmp_path, monkeypatch, {"runA": range(20), "runB": range(10, 30)})
    splits.build_splits(cfg)
    assigned = assignment(cfg)
    assert sorted(assigned) == [(6, p) for p in range(30)]


def test_the_ratios_hold_and_every_problem_gets_a_split(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": range(30)})
    splits.build_splits(cfg)
    assigned = assignment(cfg)
    counts = {s: sum(1 for v in assigned.values() if v == s) for s in ("train", "val", "test")}
    assert counts["train"] == 21 and sum(counts.values()) == 30


def test_the_same_seed_splits_the_same_way_and_a_different_one_does_not(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": range(30)})
    splits.build_splits(cfg)
    first = assignment(cfg)
    splits.build_splits(cfg)
    assert assignment(cfg) == first
    cfg.prm = dataclasses.replace(cfg.prm, split_seed=7)
    splits.build_splits(cfg)
    assert assignment(cfg) != first


def test_the_keys_come_from_the_parts_deduped_and_sorted(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runB": [5, 1], "runA": [1, 3]})
    parts_dir = os.path.join(cfg.prm.out_dir, build.PARTS)
    assert splits.problem_keys(parts_dir) == [(6, 1), (6, 3), (6, 5)]


def test_a_build_that_produced_no_rows_raises_rather_than_writing_an_empty_split(
    tmp_path, monkeypatch
):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [0]}, max_length=1)
    with pytest.raises(FileNotFoundError, match="prm.build"):
        splits.build_splits(cfg)


def test_a_bad_config_raises_before_the_parts_are_read(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [0]})
    cfg.prm = dataclasses.replace(cfg.prm, split_ratios=[0.5, 0.5])
    with pytest.raises(ValueError, match="split_ratios"):
        splits.build_splits(cfg)
    assert not os.path.exists(os.path.join(cfg.prm.out_dir, splits.SPLITS))


def test_an_interrupted_write_leaves_the_previous_splits_intact(tmp_path, monkeypatch):
    # A truncated splits.json still parses, just with problems missing from every split,
    # so the rename has to be the only thing that publishes it.
    cfg = build_parts(tmp_path, monkeypatch, {"runA": range(10)})
    splits.build_splits(cfg)
    path = os.path.join(cfg.prm.out_dir, splits.SPLITS)
    before = open(path).read()

    def boom(*a):
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        splits.build_splits(cfg)
    assert open(path).read() == before


def test_main_splits_the_build_its_config_points_at(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": range(4)})
    path = tmp_path / "c.yaml"
    path.write_text(json.dumps({"prm": dataclasses.asdict(cfg.prm)}))
    splits.main(["--config", str(path)])
    assert len(assignment(cfg)) == 4
