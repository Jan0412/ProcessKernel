"""Static anti-pattern checks for LLM-generated Triton kernels.

Pure ``ast`` analysis: no GPU, no torch import, no compilation. Runs on a login node
at roughly a millisecond per file, which is what makes it usable both for scanning
175k-file run folders and, later, inside a generation-time feedback loop.

    from checker import analyze_file
    report = analyze_file("runs/<run>/level_5_problem_2_sample_0_kernel.py")
    for finding in report.findings:
        print(finding.check_id, finding.message)
"""

from __future__ import annotations

from functools import lru_cache

from .core.model import (
    Finding,
    FileReport,
    ModuleModel,
)
from .core.naming import parse_kernel_filename, staged_kernel_filename
from .lint import LintAnalyzer
from .lint.checks import CHECKS, run_checks

__all__ = [
    "analyze_source",
    "analyze_file",
    "CHECKS",
    "Finding",
    "FileReport",
    "ModuleModel",
    "parse_kernel_filename",
    "staged_kernel_filename",
]


def build_model(
    source: str, path: str = "", fallback_shapes: list | None = None
) -> ModuleModel:
    return LintAnalyzer(fallback_shapes).build(source, path)


def analyze_source(
    source: str,
    path: str = "",
    only: set[str] | None = None,
    fallback_shapes: list | None = None,
) -> FileReport:
    return LintAnalyzer(fallback_shapes).analyze(source, path, only=only)


def analyze_file(path: str, only: set[str] | None = None) -> FileReport:
    return LintAnalyzer().analyze_path(path, only=only)
