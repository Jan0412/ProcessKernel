"""``prm.stats``: the §12 acceptance table, measured off a built dataset and checked."""

from __future__ import annotations

import json
import os

import pytest

from reranker.src.prm import build, corpus, splits, stats
from reranker.tests.prm.runfixture import (
    LENGTH,
    attempt,
    baseline_json,
    prm_config,
    verdict,
    write_round,
)

PROSE = "a\nb\nc\nd\n"
FENCED = "intro\n```python\nx = 1\n```\ntail\n"
OPEN_FENCE = "intro\n```python\nx = 1\n"


def build_parts(tmp_path, monkeypatch, runs, *, shards=("shard_00",), rnds=(0,), **over):
    """Build a dataset from ``{run: [(attempt, verdict), ...]}`` and return its config."""
    monkeypatch.setattr(build, "token_counter", lambda name: len)
    # A baseline for every problem the fixture uses, all at 2.0, so `right()` grades 0.75
    # instead of being dropped as `no_baseline` the moment a test names a new id.
    pids = sorted({a["problem_id"] for pairs in runs.values() for a, _ in pairs})
    over.setdefault("baseline", baseline_json(tmp_path, [(p, 2.0, 2.0) for p in pids]))
    dirs = []
    for name, pairs in runs.items():
        for shard in shards:
            for rnd in rnds:
                write_round(
                    tmp_path / name,
                    shard,
                    rnd,
                    [a for a, _ in pairs],
                    _verdicts([(a, v) for a, v in pairs]),
                )
        dirs.append(tmp_path / name)
    cfg = prm_config(tmp_path, dirs, rounds=list(rnds), **over)
    build.run_build(cfg)
    return cfg


def _verdicts(pairs):
    out = {}
    for a, v in pairs:
        out.setdefault(str(a["problem_id"]), []).append(v)
    return out


def wrong(pid, *, raw=PROSE, trace=None, sample_id=0):
    """An attempt that compiles but is incorrect -- graded 0.0, so no baseline is needed."""
    kw = {"trace": trace} if trace else {}
    return (
        attempt(problem_id=pid, sample_id=sample_id, raw=raw, **kw),
        verdict(sample_id=sample_id, correctness=False),
    )


def right(pid, *, raw=PROSE, trace=None, sample_id=0):
    """A correct kernel at 1x the fixture baseline (2.0/2.0) -- graded 0.75."""
    kw = {"trace": trace} if trace else {}
    return (
        attempt(problem_id=pid, sample_id=sample_id, raw=raw, **kw),
        verdict(sample_id=sample_id, correctness=True, runtime=2.0, min_runtime=2.0),
    )


def run_report(cfg, *, with_splits=True):
    if with_splits:
        splits.build_splits(cfg)
    return stats.report(cfg)


def failed(out):
    return {c["check"] for c in out["checks"] if not c["ok"]}


# ---------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    "raw, expected",
    [
        (FENCED, False),                       # closed: the span stops at the closing marker
        (OPEN_FENCE, True),                    # the max_tokens tail
        ("no fence here\n", False),
        ("```\nstray closer outside a block\n", False),  # bare ``` outside a block is skipped
        ("a\n```python\nx=1\n```\n```python\ny=2\n", True),  # second block left open
    ],
)
def test_unterminated_reads_the_last_span(raw, expected):
    assert stats.unterminated(raw) is expected


@pytest.mark.parametrize(
    "a, b, n",
    [("abcdef", "abcXYZ", 3), ("", "x", 0), ("same", "same", 4), ("x", "", 0), ("ab", "abcd", 2)],
)
def test_lcp(a, b, n):
    assert stats._lcp(a, b) == n


def test_unit_key_survives_underscores_in_the_run_name():
    name = "DeepSeek-V4-Flash_kb6_lintloop_triton_v5_traced__shard_07__round2.jsonl"
    assert stats._unit_key(name) == ("DeepSeek-V4-Flash_kb6_lintloop_triton_v5_traced", 2)


def test_pct_of_nothing_is_zero_not_a_zero_division():
    assert stats._pct(3, 0) == 0.0
    assert stats._pct(1, 4) == 25.0


def test_sibling_prefixes_reports_no_pairs_for_lone_samples():
    assert stats.sibling_prefixes({("r", 6, 1, 0): ["only one"]}) == {"pairs": 0}


def test_sibling_prefixes_counts_pairs_over_the_threshold():
    shared = "s" * (stats.SIBLING_OVER + 5)
    out = stats.sibling_prefixes(
        {
            ("r", 6, 1, 0): [shared + "a", shared + "b"],   # 1 pair, over
            ("r", 6, 2, 0): ["ab", "ac", "ad"],             # 3 pairs, 1 char each
        }
    )
    assert out["pairs"] == 4
    assert out["over_200"] == 1
    assert out["median"] == 1
    assert out["over_200_pct"] == pytest.approx(25.0)


def test_ledger_splits_counters_by_run_and_by_round():
    manifest = {
        "units": {
            "runA__shard_00__round0.jsonl": {"counts": {"rows": 2, "too_long": 1}},
            "runA__shard_01__round1.jsonl": {"counts": {"rows": 3}},
            "runB__shard_00__round0.jsonl": {"counts": {"rows": 5, "too_long": 2}},
        }
    }
    out = stats.ledger(manifest)
    assert out["by_run"] == {"runA": {"rows": 5, "too_long": 1}, "runB": {"rows": 5, "too_long": 2}}
    assert out["by_round"] == {"0": {"rows": 7, "too_long": 3}, "1": {"rows": 3}}


# ---------------------------------------------------------------- verdict side


def test_verdict_totals_counts_every_verdict_not_only_kept_rows(tmp_path, monkeypatch):
    # One attempt is dropped as `truncated`, so the parts hold 1 row but 2 verdicts exist.
    cfg = build_parts(
        tmp_path,
        monkeypatch,
        {"runA": [wrong(0), right(1, trace=LENGTH, sample_id=0)]},
    )
    found, _ = corpus.units(cfg.prm.run_dirs, cfg.prm.rounds)
    total, correct = stats.verdict_totals(found, {build.part_name(u) for u in found})
    assert (total, correct) == (2, 1)


def test_verdict_totals_skips_a_unit_with_no_part(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [right(0)]})
    found, _ = corpus.units(cfg.prm.run_dirs, cfg.prm.rounds)
    assert stats.verdict_totals(found, set()) == (0, 0)


# ---------------------------------------------------------------- end to end


def test_a_clean_build_passes_every_check(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(p) for p in range(8)]})
    out = run_report(cfg)
    assert failed(out) == set()
    assert out["totals"]["rows"] == 8
    assert out["totals"]["attempts"] == 8
    # 4 prose lines each, no code: I4 says nothing rescales these.
    assert out["cuts"]["mean"] == 4.0
    assert out["cuts"]["prose"] == 32
    assert out["cuts"]["code"] == 0


def test_report_writes_stats_json_beside_the_manifest(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(0)]})
    out = run_report(cfg)
    written = json.loads((tmp_path / "out" / stats.STATS).read_text())
    assert written["totals"] == out["totals"]
    assert written["checks"] == out["checks"]


def test_targets_land_on_the_graded_ladder_with_an_empty_band(tmp_path, monkeypatch):
    cfg = build_parts(
        tmp_path, monkeypatch, {"runA": [wrong(0), wrong(1), right(2), right(3)]}
    )
    out = run_report(cfg)
    # 1x baseline grades 0.75 under quant=0.1, never 0.769 (§5).
    assert out["targets"]["histogram"] == {"0.0": 2, "0.75": 2}
    assert out["targets"]["empty_band"] == 0
    assert out["targets"]["zero_pct"] == 50.0
    assert "nothing in the empty (0, 0.5) band" not in failed(out)


def test_finish_reasons_come_out_of_the_ledger(tmp_path, monkeypatch):
    cfg = build_parts(
        tmp_path,
        monkeypatch,
        {"runA": [wrong(0), wrong(1, trace=LENGTH), wrong(2, trace={"passes": 2})]},
    )
    out = run_report(cfg)
    # stop is the remainder: an attempt that is neither truncated nor missing a reason.
    assert out["finish_reasons"] == {"stop": 1, "length": 1, "unknown": 1}


def test_unterminated_fences_are_reported_per_run_and_do_not_fail_a_check(tmp_path, monkeypatch):
    cfg = build_parts(
        tmp_path,
        monkeypatch,
        {"runA": [wrong(0, raw=OPEN_FENCE), wrong(1, raw=FENCED)], "runB": [wrong(2, raw=OPEN_FENCE)]},
    )
    out = run_report(cfg)
    assert out["fences"]["unterminated_pct"] == pytest.approx(200 / 3)
    assert out["fences"]["by_run"] == {"runA": 50.0, "runB": 100.0}
    assert failed(out) == set()  # the gpt-oss shape is correct data, never a failure


def test_base_rate_shift_compares_the_corpus_before_and_after_dropping(tmp_path, monkeypatch):
    # 8 wrong kept, plus 2 correct dropped as `truncated`: all-attempts is 20% correct and
    # kept is 0%, a 20 point shift -- far past the gate.
    cfg = build_parts(
        tmp_path,
        monkeypatch,
        {
            "runA": [wrong(p) for p in range(8)]
            + [right(8, trace=LENGTH), right(9, trace=LENGTH)]
        },
    )
    out = run_report(cfg)
    assert out["bias"]["all_correct_pct"] == pytest.approx(20.0)
    assert out["bias"]["kept_correct_pct"] == 0.0
    assert out["bias"]["dropped_correct_pct"] == pytest.approx(100.0)
    assert out["bias"]["shift_points"] == pytest.approx(-20.0)
    assert "base-rate shift under 0.5 points" in failed(out)


def test_a_tiny_base_rate_shift_passes_the_gate(tmp_path, monkeypatch):
    # 400 wrong kept and 1 wrong dropped: the base rate does not move at all.
    cfg = build_parts(
        tmp_path,
        monkeypatch,
        {"runA": [wrong(p) for p in range(400)] + [wrong(400, trace=LENGTH)]},
    )
    out = run_report(cfg)
    assert out["bias"]["shift_points"] == 0.0
    assert "base-rate shift under 0.5 points" not in failed(out)


def test_an_attempt_with_no_verdict_fails_the_precondition(tmp_path, monkeypatch):
    # An eval that never reached this attempt. The reconciliation identity still holds, so
    # only the precondition catches it -- PLAN §13 forbids building against a live eval.
    a, v = wrong(0)
    extra, _ = wrong(1)
    monkeypatch.setattr(build, "token_counter", lambda name: len)
    write_round(tmp_path / "runA", "shard_00", 0, [a, extra], {"0": [v]})
    cfg = prm_config(
        tmp_path, [tmp_path / "runA"], baseline=baseline_json(tmp_path, [(0, 2.0, 2.0)])
    )
    build.run_build(cfg)
    out = run_report(cfg)
    assert "no attempt is missing its verdict" in failed(out)
    assert "every attempt reconciles with a verdict" not in failed(out)


def test_a_fully_evaluated_corpus_passes_the_precondition(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(0), wrong(1)]})
    out = run_report(cfg)
    assert "no attempt is missing its verdict" not in failed(out)


def test_truncated_in_the_hundreds_fails_the_detector_gate(tmp_path, monkeypatch):
    many = [wrong(p, trace=LENGTH) for p in range(stats.TRUNCATED_MAX + 1)]
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(9999)] + many})
    out = run_report(cfg)
    assert f"truncated under {stats.TRUNCATED_MAX}" in failed(out)


def test_the_v5_corpus_truncated_count_does_not_trip_the_gate(tmp_path, monkeypatch):
    # 20 over all 285,149 attempts (§12). §13 step 6 says "single digits", which would fail
    # on the corpus this build targets -- pinned so the threshold cannot drift back under it.
    assert stats.TRUNCATED_MAX > 20
    many = [wrong(p, trace=LENGTH) for p in range(20)]
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(9999)] + many})
    out = run_report(cfg)
    assert f"truncated under {stats.TRUNCATED_MAX}" not in failed(out)


def test_off_scale_target_is_caught(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(0), wrong(1)]})
    _rewrite_first_row(cfg, {"target": 0.3})
    out = run_report(cfg)
    assert failed(out) >= {"every target on the graded scale", "nothing in the empty (0, 0.5) band"}


def test_a_correct_row_graded_below_the_ladder_is_off_scale(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [right(0)]})
    _rewrite_first_row(cfg, {"target": 0.0})  # label 1 with a 0.0 target
    out = run_report(cfg)
    assert "every target on the graded scale" in failed(out)
    assert out["targets"]["empty_band"] == 0  # 0.0 is on the scale, just not for this label


def test_rows_disagreeing_with_the_manifest_are_caught(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(0), wrong(1)]})
    part = _first_part(cfg)
    rows = part.read_text().splitlines(keepends=True)
    part.write_text("".join(rows[:1]))  # a row vanishes; the .meta still counts two
    out = run_report(cfg)
    assert failed(out) >= {"rows on disk match the manifest", "cuts on disk match the manifest"}


def test_a_config_whose_row_knobs_moved_since_the_build_is_caught(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(0)]})
    splits.build_splits(cfg)
    import dataclasses

    cfg.prm = dataclasses.replace(cfg.prm, min_frac=0.5)
    out = stats.report(cfg)
    assert "report config matches the build's" in failed(out)


def test_changing_only_a_non_row_knob_does_not_trip_the_config_check(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(0)]})
    splits.build_splits(cfg)
    import dataclasses

    cfg.prm = dataclasses.replace(cfg.prm, num_workers=4, split_seed=7)
    out = stats.report(cfg)
    assert "report config matches the build's" not in failed(out)


def test_a_unit_the_config_resolves_but_never_built_is_caught(tmp_path, monkeypatch):
    cfg = build_parts(
        tmp_path, monkeypatch, {"runA": [wrong(0)]}, shards=("shard_00", "shard_01"), max_shards=1
    )
    import dataclasses

    cfg.prm = dataclasses.replace(cfg.prm, max_shards=2)
    out = stats.report(cfg)
    assert "one part per unit the config resolves" in failed(out)


def test_missing_splits_are_reported_rather_than_crashing(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(0), wrong(1)]})
    out = stats.report(cfg)  # no splits.json written
    assert "every problem has a split" in failed(out)
    assert out["splits"]["assigned"] == {}
    # Every problem is unassigned, not zero of them: a reader of stats.json must not see
    # n_missing 0 next to a failed check.
    assert out["splits"]["n_missing"] == 2
    assert out["splits"]["missing"] == ["6:0", "6:1"]


def test_a_problem_absent_from_splits_json_is_named(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(0), wrong(1)]})
    splits.build_splits(cfg)
    path = os.path.join(cfg.prm.out_dir, splits.SPLITS)
    with open(path) as f:
        assigned = json.load(f)
    assigned.pop("6:1")
    with open(path, "w") as f:
        json.dump(assigned, f)
    out = stats.report(cfg)
    assert "every problem has a split" in failed(out)
    assert out["splits"]["missing"] == ["6:1"]


def test_splits_and_problem_counts_agree(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(p) for p in range(10)]})
    out = run_report(cfg)
    assert out["splits"]["problems"] == 10
    assert sum(out["splits"]["assigned"].values()) == 10


# ---------------------------------------------------------------- failure modes


def test_report_without_a_manifest_says_so(tmp_path):
    cfg = prm_config(tmp_path, [tmp_path / "runA"])
    os.makedirs(cfg.prm.out_dir, exist_ok=True)
    with pytest.raises(FileNotFoundError, match=MANIFEST_HINT):
        stats.report(cfg)


MANIFEST_HINT = "run reranker.src.prm.build first"


def test_scan_parts_without_parts_says_so(tmp_path):
    os.makedirs(tmp_path / "parts", exist_ok=True)
    with pytest.raises(FileNotFoundError, match=MANIFEST_HINT):
        stats.scan_parts(str(tmp_path / "parts"))


def test_a_bad_knob_aborts_before_anything_is_read(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(0)]})
    import dataclasses

    cfg.prm = dataclasses.replace(cfg.prm, label_mode="binry")
    with pytest.raises(ValueError, match="label_mode"):
        stats.report(cfg)


# ---------------------------------------------------------------- CLI


def test_render_puts_the_headline_numbers_on_the_page(tmp_path, monkeypatch):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(0), right(1)]})
    text = stats.render(run_report(cfg))
    assert "rows 2 of 2 attempts" in text
    assert "checks passed" in text
    assert "0.75" in text  # the graded ladder value, not a smooth score


def test_render_reports_sibling_prefixes_when_pairs_exist(tmp_path, monkeypatch):
    # Two samples of one problem: the pair exists, so §11's fact is measured, not assumed.
    cfg = build_parts(
        tmp_path,
        monkeypatch,
        {"runA": [wrong(0, sample_id=0), wrong(0, sample_id=1, raw=PROSE + "e\n")]},
    )
    text = stats.render(run_report(cfg))
    assert "siblings   1 pairs" in text
    assert "median shared prefix 8 chars" in text


def test_main_exits_nonzero_when_a_check_fails(tmp_path, monkeypatch, capsys):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(0)]})
    monkeypatch.setattr(stats, "load_config", lambda argv: cfg)
    with pytest.raises(SystemExit) as exc:  # no splits.json
        stats.main([])
    assert exc.value.code == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_is_silent_about_failure_on_a_clean_dataset(tmp_path, monkeypatch, capsys):
    cfg = build_parts(tmp_path, monkeypatch, {"runA": [wrong(0)]})
    splits.build_splits(cfg)
    monkeypatch.setattr(stats, "load_config", lambda argv: cfg)
    stats.main([])
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert stats.STATS in out


def _first_part(cfg):
    import pathlib

    parts = sorted(pathlib.Path(cfg.prm.out_dir, build.PARTS).glob("*.jsonl"))
    return parts[0]


def _rewrite_first_row(cfg, over):
    part = _first_part(cfg)
    rows = [json.loads(line) for line in part.read_text().splitlines()]
    rows[0].update(over)
    part.write_text("".join(json.dumps(r) + "\n" for r in rows))
