"""F1.2 -- a kernel is defined but never launched from the entry point.

The degenerate form of cheating: the model writes a Triton kernel (satisfying any
"must contain @triton.jit" rule) and then never calls it, computing the answer in
PyTorch instead.

Reference: "Fine-Tuning GPT-5 for GPU Kernel Generation" (Tehrani et al., 2026,
arXiv:2602.11000), Static Reachability Analysis -- AST worklist traversal from the
entry point; at least one kernel must be reachable, else reward = 0. The failure
mode itself is documented in AutoTriton (arXiv:2507.05687).

We anchor on actual launch *sites* rather than on referenced names, which is
strictly stronger: a kernel merely mentioned in dead code does not count as live.
"""

from __future__ import annotations

from ...model import Finding, ModuleModel
from .. import register


@register("F1.2", "dead_kernel", "fail")
def check(model: ModuleModel) -> list[Finding]:
    if not model.kernels:
        return []  # F1.1 covers that

    launched = {ls.kernel_name for ls in model.reachable_launches}
    dead = [name for name in model.kernels if name not in launched]
    if not dead:
        return []

    # Without a resolvable entry point we cannot prove unreachability.
    severity = "fail" if model.entry else "info"
    findings = []
    for name in sorted(dead):
        kernel = model.kernels[name]
        findings.append(
            Finding(
                check_id="F1.2",
                severity=severity,
                message=(
                    f"Kernel `{name}` is defined but never launched from "
                    f"{model.entry or 'the entry point'}. Launch it (or remove it) -- "
                    f"a kernel that never runs does no work."
                ),
                data={"kernel": name, "lineno": kernel.lineno, "entry": model.entry},
            )
        )
    return findings
