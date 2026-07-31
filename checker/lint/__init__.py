"""The Triton linter: "is this good Triton?"

Family 1 asks whether the kernel is real -- whether the work it claims to do on the GPU is
actually reachable and not quietly delegated back to torch. Family 2 asks what it wastes
once it is. Both run over the model :mod:`checker.core.parsing` builds, extended here by
the kernel-body, host-flow and shape stages.
"""

from __future__ import annotations

from functools import lru_cache

from ..core.analyzer import Analyzer
from ..core.model import Finding, FileReport, ModuleModel
from ..core.naming import parse_kernel_filename
from ..core.parsing import build_skeleton
from .checks import LINT_REGISTRY
from .hostflow import analyze_host
from .kernelbody import analyze_kernels
from .shapes import infer as infer_shapes
from .summary import lint_summary


class LintAnalyzer(Analyzer):
    registry = LINT_REGISTRY

    def __init__(self, fallback_shapes: list | None = None) -> None:
        #: Most generations drop ``get_inputs()``; the reference the kernel was evaluated
        #: against always has it, so a caller can seed the shapes from there.
        self.fallback_shapes = fallback_shapes

    def build(self, source: str, path: str = "") -> ModuleModel:
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
            lambda m: infer_shapes(m, self.fallback_shapes),
        )
        for stage in stages:
            try:
                stage(model)
            except Exception as exc:  # noqa: BLE001 - defensive by design
                model.parse_status = "partial"
                model.notes.append(f"stage raised {type(exc).__name__}: {exc}")

        return model

    def summarize(self, model: ModuleModel, findings: list[Finding]) -> dict:
        return lint_summary(model, findings)

    def analyze_path(self, path: str, only: set[str] | None = None) -> FileReport:
        """As the base, but seeded with the reference's input shapes.

        Only ~31% of generated kernels keep ``get_inputs()``, and Family 2 reports waste in
        bytes -- so where the filename identifies the problem, the shapes come from the
        KernelBench reference the kernel was actually evaluated against.
        """
        meta = parse_kernel_filename(path.rsplit("/", 1)[-1])
        fallback = None
        if meta:
            try:
                fallback = _reference_shapes(meta[0], meta[1])
            except Exception:  # noqa: BLE001 - shapes are best-effort
                fallback = None

        previous = self.fallback_shapes
        self.fallback_shapes = fallback
        try:
            return super().analyze_path(path, only=only)
        finally:
            self.fallback_shapes = previous


@lru_cache(maxsize=4096)
def _reference_shapes(level: int, problem_id: int) -> tuple:
    from .shapes import reference_input_shapes

    return tuple(reference_input_shapes(level, problem_id))
