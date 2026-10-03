"""F1.1 -- the solution contains no Triton kernel at all.

Reference: AutoTriton (Li et al., 2025, arXiv:2507.05687) -- the rule-based
component of its reward assigns 0 to any generation without a ``@triton.jit``
decorator.
"""

from __future__ import annotations

from ....core.check import Check
from ....core.model import Finding, ModuleModel
from .. import LINT_REGISTRY


@LINT_REGISTRY.add
class NoTritonKernel(Check):
    check_id = "F1.1"
    name = "no_triton_kernel"
    severity = "fail"

    def run(self, model: ModuleModel) -> list[Finding]:
        if model.parse_status not in ("ok", "partial"):
            return []
        if model.kernels:
            return []
        return [
            self.finding(
                "The solution contains no @triton.jit kernel. The computation must be "
                "implemented as a Triton kernel, not with PyTorch operators.",
                n_kernels=0,
            )
        ]
