"""report.py -- joining static findings back onto measured outcomes."""

from __future__ import annotations

import json

import pytest
from conftest import DEAD_KERNEL_FILE, GOOD_KERNEL_FILE

from checker.report import (
    _fmt,
    _mean,
    _median,
    load_findings,
    print_summary,
    rows,
    summarize,
)
from checker.scan import scan_run

#: Problem 1 is correct and fast; problem 2 is wrong -- and is the one F1.2 flags.
EVAL_RESULTS = {
    "1": [
        {
            "sample_id": 0,
            "compiled": True,
            "correctness": True,
            "runtime": 1.0,
            "runtime_stats": {"min": 0.9},
            "metadata": {"hardware": "NVIDIA A100-SXM4-80GB"},
        }
    ],
    "2": [
        {
            "sample_id": 0,
            "compiled": True,
            "correctness": False,
            "runtime": 4.0,
            "runtime_stats": {"min": 3.9},
            "metadata": {"hardware": "NVIDIA A100-SXM4-80GB"},
        }
    ],
}


@pytest.fixture
def scanned(make_run, tmp_path, fake_kernelbench):
    """A scanned run: returns (run_dir, findings_path)."""
    run_dir = make_run(
        "Qwen_level1_triton",
        files={(1, 0): GOOD_KERNEL_FILE, (2, 0): DEAD_KERNEL_FILE},
        eval_results=EVAL_RESULTS,
    )
    findings = tmp_path / "findings.jsonl"
    scan_run(run_dir, str(findings), workers=1)
    return run_dir, str(findings)


class TestLoadFindings:
    def test_indexes_by_level_problem_sample(self, scanned):
        _, findings_path = scanned
        index = load_findings(findings_path)
        assert set(index) == {(1, 1, 0), (1, 2, 0)}

    def test_skips_rows_without_full_identity(self, tmp_path):
        """A file analyzed outside a run has no level/problem/sample -- unjoinable."""
        path = tmp_path / "findings.jsonl"
        path.write_text(
            json.dumps({"level": None, "problem_id": None, "sample_id": None, "path": "x.py"})
            + "\n"
            + json.dumps({"level": 1, "problem_id": 2, "sample_id": 0, "path": "y.py"})
            + "\n"
        )
        assert set(load_findings(str(path))) == {(1, 2, 0)}


class TestRows:
    def test_one_row_per_sample_with_outcome_and_checks(self, scanned):
        run_dir, findings_path = scanned
        by_problem = {r["problem_id"]: r for r in rows(run_dir, findings_path)}
        assert set(by_problem) == {1, 2}

        clean = by_problem[1]
        assert clean["correct"] is True
        assert clean["speedup"] == 2.0  # baseline mean 2.0 / runtime 1.0
        assert clean["n_launches"] == 1
        assert clean["F1.2"] is False

        flagged = by_problem[2]
        assert flagged["correct"] is False
        assert flagged["speedup"] is None  # incorrect samples are not timed
        assert flagged["F1.2"] is True
        assert flagged["n_fail"] >= 1

    def test_samples_missing_from_the_scan_are_skipped(self, make_run, tmp_path, fake_kernelbench):
        run_dir = make_run(
            files={(1, 0): GOOD_KERNEL_FILE}, eval_results=EVAL_RESULTS
        )  # eval has problem 2, but no kernel file was scanned for it
        findings = tmp_path / "findings.jsonl"
        scan_run(run_dir, str(findings), workers=1)

        assert [r["problem_id"] for r in rows(run_dir, str(findings))] == [1]


class TestSummarize:
    def test_contrasts_flagged_against_clean(self, scanned):
        run_dir, findings_path = scanned
        summary = summarize(run_dir, findings_path)

        assert summary["n_samples"] == 2
        assert summary["n_correct"] == 1

        f12 = summary["checks"]["F1.2"]
        assert f12["n"] == 1
        assert f12["rate"] == 0.5
        assert f12["correct_rate_when_flagged"] == 0.0
        assert f12["correct_rate_when_clean"] == 1.0
        assert f12["median_speedup_when_flagged"] is None  # the flagged sample was wrong
        assert f12["median_speedup_when_clean"] == 2.0

        assert summary["checks"]["F1.1"] == {"n": 0, "rate": 0.0}  # never fired

    def test_no_joined_rows(self, make_run, tmp_path):
        run_dir = make_run(eval_results={})
        findings = tmp_path / "findings.jsonl"
        findings.write_text("")
        assert summarize(run_dir, str(findings)) == {"n": 0}


class TestPrintSummary:
    def test_prints_a_row_per_firing_check(self, scanned, capsys):
        run_dir, findings_path = scanned
        print_summary(run_dir, findings_path)
        out = capsys.readouterr().out

        assert "samples: 2" in out
        assert "correct: 1 (50.0%)" in out
        assert "F1.2" in out
        assert "F1.1" not in out  # checks that never fired are not listed
        assert "2.00x" in out  # median speedup of the clean samples
        assert "-" in out  # ...and "-" where there is nothing to report

    def test_reports_an_empty_join(self, make_run, tmp_path, capsys):
        run_dir = make_run(eval_results={})
        findings = tmp_path / "findings.jsonl"
        findings.write_text("")

        print_summary(run_dir, str(findings))

        assert "no rows joined" in capsys.readouterr().out


class TestFormatters:
    def test_mean_ignores_none(self):
        assert _mean([True, False, None]) == 0.5
        assert _mean([None]) is None

    def test_median_ignores_none(self):
        assert _median([1.0, 3.0, None]) == 2.0
        assert _median([]) is None

    def test_fmt(self):
        assert _fmt(None) == "-"
        assert _fmt(0.5, pct=True) == "50.0%"
        assert _fmt(1.5) == "1.50x"
