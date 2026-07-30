"""F1.1 -- the solution contains no Triton kernel at all.

Reference: AutoTriton (Li et al., 2025, arXiv:2507.05687) -- the rule-based
component of its reward assigns 0 to any generation without a ``@triton.jit``
decorator.
"""

from __future__ import annotations

from ....core.model import Finding, ModuleModel
from .. import register


@register("F1.1", "no_triton_kernel", "fail")
def check(model: ModuleModel) -> list[Finding]:
    if model.parse_status not in ("ok", "partial"):
        return []
    if model.kernels:
        return []
    return [
        Finding(
            check_id="F1.1",
            severity="fail",
            message=(
                "The solution contains no @triton.jit kernel. The computation must be "
                "implemented as a Triton kernel, not with PyTorch operators."
            ),
            data={"n_kernels": 0},
        )
    ]
