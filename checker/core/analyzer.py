"""What an analyzer is: a front end, a registry, and a way to summarise the result.

``analyze`` is a template method, so both analyzers produce a ``FileReport`` with the same
shape and the same degradation behaviour, and the pieces that genuinely differ -- how the
model is built, what the summary carries -- are the only things a subclass writes.

Having the registry be an attribute rather than a global is what makes the batch driver
and the eval-outcome join generic: ``scan`` runs whatever analyzer it is handed, and
``report`` reads its columns off ``analyzer.registry.check_ids`` instead of a hardcoded
tuple that goes stale the moment a check is added.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from .check import Registry
from .model import Finding, FileReport, ModuleModel
from .naming import parse_kernel_filename
from .parsing import build_skeleton
from .summary import build_summary


class Analyzer(ABC):
    registry: ClassVar[Registry]

    def build(self, source: str, path: str = "") -> ModuleModel:
        """Source -> model. The shared skeleton; the linter adds its stages on top."""
        return build_skeleton(source, path)

    def summarize(self, model: ModuleModel, findings: list[Finding]) -> dict:
        """The generic counts. The linter extends these with its own derived keys."""
        return build_summary(findings)

    def analyze(
        self, source: str, path: str = "", only: set[str] | None = None
    ) -> FileReport:
        model = self.build(source, path)
        findings = self.registry.run(model, only=only) if model.tree is not None else []

        summary = self.summarize(model, findings)
        if model.notes:
            summary["notes"] = model.notes

        report = FileReport(
            path=path,
            parse_status=model.parse_status,
            findings=findings,
            summary=summary,
        )

        meta = parse_kernel_filename(path.rsplit("/", 1)[-1])
        if meta:
            report.level, report.problem_id, report.sample_id = meta
        return report

    def analyze_path(self, path: str, only: set[str] | None = None) -> FileReport:
        """Analyze a file. An unreadable one is a report, not an exception: the batch
        driver scans whole run folders and must not stop at a bad inode."""
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                source = handle.read()
        except OSError as exc:
            report = FileReport(path=path, parse_status="read_error")
            report.summary = {"notes": [str(exc)]}
            meta = parse_kernel_filename(path.rsplit("/", 1)[-1])
            if meta:
                report.level, report.problem_id, report.sample_id = meta
            return report

        return self.analyze(source, path, only=only)
