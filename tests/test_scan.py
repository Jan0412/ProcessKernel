"""scan.py -- the batch scanner over a run folder."""

from __future__ import annotations

import json

import pytest
from conftest import DEAD_KERNEL_FILE, GOOD_KERNEL_FILE

from triton_lint import scan
from triton_lint.scan import _tally, iter_kernel_files, scan_run

TWO_FILES = {(1, 0): GOOD_KERNEL_FILE, (2, 0): DEAD_KERNEL_FILE}


@pytest.fixture(autouse=True)
def _hermetic(fake_kernelbench):
    """scan_run() resolves reference shapes per file. Point that at the fixture tree so
    the tests never read the repo's real KernelBench/ folder."""


def read_rows(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


class TestIterKernelFiles:
    def test_finds_kernels_and_ignores_everything_else(self, make_run):
        run_dir = make_run(files=TWO_FILES, eval_results={})
        names = [p.rsplit("/", 1)[-1] for p in iter_kernel_files(run_dir)]
        assert names == [
            "level_1_problem_1_sample_0_kernel.py",
            "level_1_problem_2_sample_0_kernel.py",
        ]

    def test_sorts_numerically_not_lexically(self, make_run):
        """Problem 10 must come after problem 2 -- the sort key is the parsed metadata."""
        files = {(2, 0): GOOD_KERNEL_FILE, (10, 0): GOOD_KERNEL_FILE}
        paths = iter_kernel_files(make_run(files=files))
        assert paths[0].endswith("problem_2_sample_0_kernel.py")
        assert paths[1].endswith("problem_10_sample_0_kernel.py")

    def test_limit(self, make_run):
        assert len(iter_kernel_files(make_run(files=TWO_FILES), limit=1)) == 1


class TestScanRun:
    def test_writes_one_row_per_file_and_tallies_checks(self, make_run, tmp_path):
        run_dir = make_run("Qwen_level1_triton", files=TWO_FILES)
        out = tmp_path / "findings.jsonl"

        stats = scan_run(run_dir, str(out), workers=1)

        assert stats["total"] == 2
        assert stats["written"] == 2
        assert stats["by_status"] == {"ok": 2}
        assert stats["by_check"]["F1.2"] == 1  # only the dead-kernel file

        rows = read_rows(out)
        assert {r["problem_id"] for r in rows} == {1, 2}
        assert all(r["run_name"] == "Qwen_level1_triton" for r in rows)
        assert all(r["level"] == 1 for r in rows)

    def test_only_runs_the_requested_checks(self, make_run, tmp_path):
        run_dir = make_run(files=TWO_FILES)
        out = tmp_path / "findings.jsonl"

        stats = scan_run(run_dir, str(out), workers=1, only={"F1.1"})

        assert "F1.2" not in stats["by_check"]  # filtered out, though the file has one

    def test_limit(self, make_run, tmp_path):
        run_dir = make_run(files=TWO_FILES)
        stats = scan_run(run_dir, str(tmp_path / "f.jsonl"), workers=1, limit=1)
        assert stats["total"] == 1 and stats["written"] == 1

    def test_process_pool(self, make_run, tmp_path):
        """The >1-worker path forks a pool; results are the same, order is not."""
        run_dir = make_run(files=TWO_FILES)
        out = tmp_path / "findings.jsonl"

        stats = scan_run(run_dir, str(out), workers=2)

        assert stats["written"] == 2
        assert {r["problem_id"] for r in read_rows(out)} == {1, 2}

    def test_run_name_falls_back_to_the_folder_name(self, make_run, tmp_path):
        """An unloadable run (no config, no level in the name) still scans."""
        run_dir = make_run("bare_run", config="", files={(1, 0): GOOD_KERNEL_FILE})
        out = tmp_path / "findings.jsonl"

        scan_run(run_dir, str(out), workers=1)

        assert read_rows(out)[0]["run_name"] == "bare_run"

    def test_a_crashing_file_does_not_abort_the_batch(self, make_run, tmp_path, monkeypatch):
        """One malformed generation must not kill a 175k-file scan."""

        def boom(path, only=None):
            raise RuntimeError("analysis exploded")

        monkeypatch.setattr(scan, "analyze_file", boom)
        run_dir = make_run(files=TWO_FILES)
        out = tmp_path / "findings.jsonl"

        stats = scan_run(run_dir, str(out), workers=1)

        assert stats["by_status"] == {"read_error": 2}
        row = read_rows(out)[0]
        assert row["summary"]["notes"] == ["RuntimeError: analysis exploded"]


class TestTally:
    def test_counts_status_and_checks(self):
        stats = {"total": 0, "written": 0, "by_status": {}, "by_check": {}}
        line = json.dumps({"parse_status": "ok", "summary": {"check_ids": ["F1.2", "F2.1"]}})

        _tally(stats, line)
        _tally(stats, line)

        assert stats["by_status"] == {"ok": 2}
        assert stats["by_check"] == {"F1.2": 2, "F2.1": 2}
