"""The Analyzer template method.

``analyze`` fixes the shape of a report -- build, run, summarize, stamp -- so that a second
analyzer cannot accidentally produce something the run-dir readers and the critic do not
recognise. These pin that a subclass overriding one step still gets the others.
"""

from __future__ import annotations

from checker.core.analyzer import Analyzer
from checker.core.check import Check, Registry
from checker.core.model import Finding, ModuleModel

SOURCE = "import torch\n\n\nclass ModelNew:\n    def forward(self, x):\n        return x\n"

REGISTRY = Registry("test")


@REGISTRY.add
class Always(Check):
    check_id = "T1.0"
    name = "always"
    severity = "fail"

    def run(self, model: ModuleModel) -> list[Finding]:
        return [self.finding("always fires", lineno=1)]


class Plain(Analyzer):
    registry = REGISTRY


class TestTemplateMethod:
    def test_builds_runs_and_summarizes(self):
        report = Plain().analyze(SOURCE, "<test>")

        assert report.parse_status == "ok"
        assert [f.check_id for f in report.findings] == ["T1.0"]
        assert report.summary["n_fail"] == 1
        assert report.summary["check_ids"] == ["T1.0"]

    def test_only_is_passed_through_to_the_registry(self):
        assert Plain().analyze(SOURCE, "<test>", only={"T9.9"}).findings == []

    def test_a_subclass_overriding_only_build_still_gets_the_rest(self):
        class NoTree(Plain):
            def build(self, source, path=""):
                model = super().build(source, path)
                model.tree = None
                return model

        report = NoTree().analyze(SOURCE, "<test>")

        # No tree means no checks ran, but the report is still well formed.
        assert report.findings == []
        assert report.summary["check_ids"] == []

    def test_a_subclass_overriding_only_summarize_still_gets_the_rest(self):
        class Extra(Plain):
            def summarize(self, model, findings):
                return {**super().summarize(model, findings), "extra": True}

        report = Extra().analyze(SOURCE, "<test>")

        assert report.summary["extra"] is True
        assert report.summary["n_fail"] == 1

    def test_notes_from_a_degraded_build_reach_the_summary(self):
        class Noting(Plain):
            def build(self, source, path=""):
                model = super().build(source, path)
                model.notes.append("stage raised RuntimeError: boom")
                return model

        assert Noting().analyze(SOURCE, "<test>").summary["notes"] == [
            "stage raised RuntimeError: boom"
        ]

    def test_a_syntax_error_reports_rather_than_raises(self):
        report = Plain().analyze("def (:\n", "<test>")

        assert report.parse_status == "syntax_error"
        assert report.findings == []


class TestFilenameStamping:
    def test_a_kernel_filename_populates_the_identity_columns(self):
        report = Plain().analyze(SOURCE, "runs/x/level_2_problem_37_sample_4_kernel.py")
        assert (report.level, report.problem_id, report.sample_id) == (2, 37, 4)

    def test_a_non_kernel_filename_leaves_them_none(self):
        report = Plain().analyze(SOURCE, "<test>")
        assert (report.level, report.problem_id, report.sample_id) == (None, None, None)


class TestAnalyzePath:
    def test_reads_the_file_and_analyzes_it(self, tmp_path):
        path = tmp_path / "level_1_problem_3_sample_0_kernel.py"
        path.write_text(SOURCE)

        report = Plain().analyze_path(str(path))

        assert report.parse_status == "ok"
        assert report.problem_id == 3

    def test_an_unreadable_file_is_a_report_not_an_exception(self, tmp_path):
        # A batch scan covers 175k files; one bad inode must not end it.
        missing = tmp_path / "level_1_problem_9_sample_0_kernel.py"

        report = Plain().analyze_path(str(missing))

        assert report.parse_status == "read_error"
        assert report.summary["notes"]
        assert report.problem_id == 9
