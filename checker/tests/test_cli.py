"""cli.py -- the three subcommands, plus `python -m checker`."""

from __future__ import annotations

import json
import os
import runpy
import sys

import pytest
from conftest import DEAD_KERNEL_FILE, GOOD_KERNEL_FILE

from checker.cli import main

EVAL_RESULTS = {
    "2": [
        {
            "sample_id": 0,
            "compiled": True,
            "correctness": False,
            "runtime": 4.0,
            "metadata": {"hardware": "NVIDIA A100-SXM4-80GB"},
        }
    ]
}


@pytest.fixture
def run_dir(make_run, fake_kernelbench):
    return make_run(
        "Qwen_level1_triton",
        files={(1, 0): GOOD_KERNEL_FILE, (2, 0): DEAD_KERNEL_FILE},
        eval_results=EVAL_RESULTS,
    )


class TestCheck:
    def test_prints_findings(self, run_dir, capsys):
        path = os.path.join(run_dir, "level_1_problem_2_sample_0_kernel.py")

        assert main(["check", path]) == 0

        out = capsys.readouterr().out
        assert "[ok]" in out
        assert "summary:" in out
        assert "[FAIL] F1.2" in out
        assert "never launched" in out

    def test_reports_a_clean_file(self, run_dir, capsys):
        path = os.path.join(run_dir, "level_1_problem_1_sample_0_kernel.py")

        assert main(["check", path]) == 0

        assert "no findings" in capsys.readouterr().out

    def test_json_output(self, run_dir, capsys):
        path = os.path.join(run_dir, "level_1_problem_2_sample_0_kernel.py")

        assert main(["check", "--json", path]) == 0

        report = json.loads(capsys.readouterr().out)
        assert report["level"] == 1
        assert report["problem_id"] == 2
        assert report["sample_id"] == 0
        # The kernel is never launched (F1.2) and forward() adds in PyTorch (F1.4).
        assert {f["check_id"] for f in report["findings"]} == {"F1.2", "F1.4"}


class TestScan:
    def test_writes_findings_and_prints_per_check_counts(self, run_dir, tmp_path, capsys):
        out_path = tmp_path / "findings.jsonl"

        assert main(["scan", run_dir, "--out", str(out_path), "--workers", "1"]) == 0

        out = capsys.readouterr().out
        assert f"wrote 2 rows to {out_path}" in out
        assert "parse status: {'ok': 2}" in out
        assert "files per check:" in out
        assert "F1.2" in out and "(50.0%)" in out

        with open(out_path, encoding="utf-8") as fh:
            assert len(fh.readlines()) == 2

    def test_check_filter_and_limit(self, run_dir, tmp_path, capsys):
        out_path = tmp_path / "findings.jsonl"

        code = main(
            [
                "scan", run_dir,
                "--out", str(out_path),
                "--workers", "1",
                "--limit", "1",
                "--checks", "F1.1,F1.4",
            ]
        )

        assert code == 0
        out = capsys.readouterr().out
        assert "wrote 1 rows" in out
        assert "F1.2" not in out  # not in --checks, so never run


class TestReport:
    def test_prints_summary(self, run_dir, tmp_path, capsys):
        findings = tmp_path / "findings.jsonl"
        main(["scan", run_dir, "--out", str(findings), "--workers", "1"])
        capsys.readouterr()

        assert main(["report", run_dir, "--findings", str(findings)]) == 0

        out = capsys.readouterr().out
        assert "samples: 1" in out
        assert "F1.2" in out

    def test_out_writes_joined_rows(self, run_dir, tmp_path, capsys):
        findings = tmp_path / "findings.jsonl"
        joined = tmp_path / "joined.jsonl"
        main(["scan", run_dir, "--out", str(findings), "--workers", "1"])
        capsys.readouterr()

        assert main(["report", run_dir, "--findings", str(findings), "--out", str(joined)]) == 0

        assert f"wrote joined rows to {joined}" in capsys.readouterr().out
        with open(joined, encoding="utf-8") as fh:
            row = json.loads(fh.readline())
        assert row["problem_id"] == 2
        assert row["F1.2"] is True
        assert row["correct"] is False


class TestModuleEntryPoint:
    def test_python_m_checker(self, run_dir, monkeypatch, capsys):
        path = os.path.join(run_dir, "level_1_problem_1_sample_0_kernel.py")
        monkeypatch.setattr(sys, "argv", ["checker", "check", path])

        with pytest.raises(SystemExit) as exc:
            runpy.run_module("checker", run_name="__main__")

        assert exc.value.code == 0
        assert "no findings" in capsys.readouterr().out

    def test_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            main([])
