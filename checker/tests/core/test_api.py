"""The public API: build_model / analyze_source / analyze_file, and their degradation."""

from __future__ import annotations

import os

from conftest import GOOD_KERNEL_FILE, src

import checker
from checker import analyze_file, analyze_source, build_model


class TestAnalyzeSource:
    def test_summary_and_findings(self):
        report = analyze_source(GOOD_KERNEL_FILE, "<test>")
        assert report.parse_status == "ok"
        assert report.findings == []
        assert report.summary["n_kernels"] == 1
        assert report.summary["n_launches"] == 1
        assert report.summary["check_ids"] == []

    def test_recovers_identity_from_the_filename(self):
        report = analyze_source(GOOD_KERNEL_FILE, "runs/r/level_5_problem_2_sample_7_kernel.py")
        assert (report.level, report.problem_id, report.sample_id) == (5, 2, 7)

    def test_no_identity_for_a_plain_filename(self):
        report = analyze_source(GOOD_KERNEL_FILE, "scratch.py")
        assert report.level is None

    def test_syntax_error_runs_no_checks(self):
        report = analyze_source("def f(:\n", "<test>")
        assert report.parse_status == "syntax_error"
        assert report.findings == []
        assert report.summary["notes"]

    def test_to_json_roundtrips(self):
        report = analyze_source(GOOD_KERNEL_FILE, "level_1_problem_2_sample_0_kernel.py")
        import json

        payload = json.loads(report.to_json())
        assert payload["problem_id"] == 2
        assert payload["parse_status"] == "ok"
        assert payload["summary"]["n_kernels"] == 1


class TestDegradation:
    def test_a_raising_stage_downgrades_to_partial(self, monkeypatch):
        """One bad file must never abort a 175k-file scan: the stage is isolated and
        the checks still run on whatever was recovered."""

        def boom(model):
            raise RuntimeError("stage exploded")

        monkeypatch.setattr(checker, "analyze_host", boom)

        model = build_model(GOOD_KERNEL_FILE, "<test>")

        assert model.parse_status == "partial"
        assert "stage raised RuntimeError: stage exploded" in model.notes
        assert model.kernels  # the earlier stage's work survives

    def test_a_raising_check_is_skipped_not_fatal(self, monkeypatch):
        from checker.lint import checks

        def boom(model):
            raise RuntimeError("check exploded")

        monkeypatch.setattr(checks.CHECKS[0], "run", boom)

        report = analyze_source(GOOD_KERNEL_FILE, "<test>")

        assert report.parse_status == "ok"
        assert any("raised RuntimeError: check exploded" in n for n in report.summary["notes"])


class TestAnalyzeFile:
    def test_reads_and_analyzes(self, make_run, fake_kernelbench):
        run_dir = make_run(files={(2, 0): GOOD_KERNEL_FILE})
        path = os.path.join(run_dir, "level_1_problem_2_sample_0_kernel.py")

        report = analyze_file(path)

        assert report.parse_status == "ok"
        assert (report.level, report.problem_id, report.sample_id) == (1, 2, 0)

    def test_falls_back_to_the_reference_shapes(self, make_run, fake_kernelbench):
        """Most generations drop get_inputs(); the KernelBench reference always has it,
        and it is what the kernel was evaluated against."""
        no_get_inputs = src(
            """
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) * 2.0, mask=offs < n)

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.zeros_like(x)
        k[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
        )
        run_dir = make_run(files={(2, 0): no_get_inputs})
        path = os.path.join(run_dir, "level_1_problem_2_sample_0_kernel.py")

        report = analyze_file(path)

        # reference 2_Add.py declares (4, 8) float32 inputs -> F2.4 can price the memset
        f24 = [f for f in report.findings if f.check_id == "F2.4"]
        assert f24 and f24[0].data["bytes"] == 4 * 8 * 4

    def test_missing_file_is_a_read_error(self, tmp_path):
        path = str(tmp_path / "level_1_problem_2_sample_0_kernel.py")

        report = analyze_file(path)

        assert report.parse_status == "read_error"
        assert report.findings == []
        assert report.summary["notes"]
        assert (report.level, report.problem_id, report.sample_id) == (1, 2, 0)

    def test_missing_file_without_kernel_naming(self, tmp_path):
        report = analyze_file(str(tmp_path / "whatever.py"))
        assert report.parse_status == "read_error"
        assert report.level is None

    def test_unreadable_reference_shapes_are_not_fatal(self, make_run, monkeypatch):
        """Shapes are best-effort: losing them costs byte counts, not the analysis."""
        from checker.lint import shapes

        def boom(level, problem_id):
            raise RuntimeError("KernelBench is on fire")

        monkeypatch.setattr(shapes, "reference_input_shapes", boom)
        checker._reference_shapes.cache_clear()

        run_dir = make_run(files={(2, 0): GOOD_KERNEL_FILE})
        report = analyze_file(os.path.join(run_dir, "level_1_problem_2_sample_0_kernel.py"))

        assert report.parse_status == "ok"
        checker._reference_shapes.cache_clear()

    def test_only_runs_the_requested_check(self, make_run, fake_kernelbench):
        dead = src(
            """
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) * 2.0, mask=offs < n)

class ModelNew(nn.Module):
    def forward(self, x):
        return torch.relu(x)
"""
        )
        run_dir = make_run(files={(2, 0): dead})
        path = os.path.join(run_dir, "level_1_problem_2_sample_0_kernel.py")

        report = analyze_file(path, only={"F1.2"})

        assert {f.check_id for f in report.findings} == {"F1.2"}


class TestReferenceShapeCache:
    def test_repeated_lookups_hit_the_cache(self, fake_kernelbench):
        checker._reference_shapes.cache_clear()

        first = checker._reference_shapes(1, 2)
        second = checker._reference_shapes(1, 2)

        assert first == second == (((4, 8), "float32"), ((4, 8), "float32"))
        assert checker._reference_shapes.cache_info().hits == 1
