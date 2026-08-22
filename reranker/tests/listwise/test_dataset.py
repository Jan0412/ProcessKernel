"""``listwise.dataset``: which source rows a training run has to hold in memory."""

from __future__ import annotations

import json

import pytest

from reranker.src.listwise.dataset import _rows_for_lists, _row_key


def source(tmp_path, rows):
    path = tmp_path / "ds.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def row(run, pid, sid, *, ref="ref", src="kernel"):
    return {
        "run_name": run, "level": 6, "problem_id": pid, "sample_id": sid,
        "ref_arch_src": ref, "kernel_src": src, "compiled": True, "label": 1,
        "runtime": 1.0, "speedup": 1.0, "speedup_min": 1.0,
    }


def lst(pid, cands):
    return {"level": 6, "problem_id": pid,
            "candidates": [{"run_name": r, "sample_id": s, "rel": 1.0} for r, s in cands]}


def test_only_the_rows_the_lists_cite_are_kept(tmp_path):
    rows = [row("runA", 1, 0), row("runA", 1, 1), row("runA", 2, 0), row("runB", 3, 0)]
    got = _rows_for_lists(source(tmp_path, rows), [lst(1, [("runA", 0)]), lst(3, [("runB", 0)])])
    assert set(got) == {_row_key("runA", 6, 1, 0), _row_key("runB", 6, 3, 0)}


def test_a_kept_row_carries_only_what_encoding_reads(tmp_path):
    rows = [row("runA", 1, 0, ref="R", src="K")]
    got = _rows_for_lists(source(tmp_path, rows), [lst(1, [("runA", 0)])])
    assert got[_row_key("runA", 6, 1, 0)] == {"ref_arch_src": "R", "kernel_src": "K"}


def test_the_same_sample_id_in_two_runs_stays_two_rows(tmp_path):
    # The shard collision again, from the training side: these must not fold together.
    rows = [row("runA", 1, 0, src="fromA"), row("runB", 1, 0, src="fromB")]
    got = _rows_for_lists(source(tmp_path, rows), [lst(1, [("runA", 0), ("runB", 0)])])
    assert got[_row_key("runA", 6, 1, 0)]["kernel_src"] == "fromA"
    assert got[_row_key("runB", 6, 1, 0)]["kernel_src"] == "fromB"


def test_a_list_citing_a_row_that_is_not_there_is_left_for_the_lookup_to_raise(tmp_path):
    got = _rows_for_lists(source(tmp_path, [row("runA", 1, 0)]), [lst(1, [("runA", 9)])])
    assert got == {}


def test_a_blank_line_is_skipped(tmp_path):
    path = tmp_path / "ds.jsonl"
    path.write_text(json.dumps(row("runA", 1, 0)) + "\n\n")
    assert len(_rows_for_lists(str(path), [lst(1, [("runA", 0)])])) == 1


def test_the_same_row_cited_by_two_lists_is_stored_once(tmp_path):
    rows = [row("runA", 1, 0)]
    got = _rows_for_lists(source(tmp_path, rows), [lst(1, [("runA", 0)]), lst(1, [("runA", 0)])])
    assert len(got) == 1
