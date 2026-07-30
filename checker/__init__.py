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

from .checks import CHECKS, run_checks
from .hostflow import analyze_host
from .kernelbody import analyze_kernels
from .model import (
    Finding,
    FileReport,
    ModuleModel,
    build_summary,
    parse_kernel_filename,
    staged_kernel_filename,
)
from .parsing import build_skeleton
from .shapes import infer as infer_shapes

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
    """Run the analysis pipeline, degrading rather than raising.

    Each stage is isolated: a failure in (say) shape inference downgrades the file to
    ``parse_status="partial"`` and the checks still run on what we did recover. One
    malformed file must never abort a 175k-file scan.
    """
    model = build_skeleton(source, path)
    if model.tree is None:
        return model

    stages = (
        analyze_kernels,
        analyze_host,
        lambda m: infer_shapes(m, fallback_shapes),
    )
    for stage in stages:
        try:
            stage(model)
        except Exception as exc:  # noqa: BLE001 - defensive by design
            model.parse_status = "partial"
            model.notes.append(f"stage raised {type(exc).__name__}: {exc}")

    return model


def analyze_source(
    source: str,
    path: str = "",
    only: set[str] | None = None,
    fallback_shapes: list | None = None,
) -> FileReport:
    model = build_model(source, path, fallback_shapes)
    findings = run_checks(model, only=only) if model.tree is not None else []

    summary = build_summary(model, findings)
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


def analyze_file(path: str, only: set[str] | None = None) -> FileReport:
    meta = parse_kernel_filename(path.rsplit("/", 1)[-1])

    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    except OSError as exc:
        report = FileReport(path=path, parse_status="read_error")
        report.summary = {"notes": [str(exc)]}
        if meta:
            report.level, report.problem_id, report.sample_id = meta
        return report

    # Most generations drop get_inputs(); the KernelBench reference the kernel was
    # evaluated against always has it, so fall back to that for tensor shapes.
    fallback = None
    if meta:
        from .shapes import reference_input_shapes

        try:
            fallback = _reference_shapes(meta[0], meta[1])
        except Exception:  # noqa: BLE001 - shapes are best-effort
            fallback = None

    return analyze_source(source, path, only=only, fallback_shapes=fallback)


@lru_cache(maxsize=4096)
def _reference_shapes(level: int, problem_id: int) -> tuple:
    from .shapes import reference_input_shapes

    return tuple(reference_input_shapes(level, problem_id))
