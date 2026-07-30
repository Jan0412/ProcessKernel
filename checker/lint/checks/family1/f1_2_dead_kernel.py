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

from ....core.model import Finding, ModuleModel
from .. import register


@register("F1.2", "dead_kernel", "fail")
def check(model: ModuleModel) -> list[Finding]:
    if not model.kernels:
        return []  # F1.1 covers that

    launched = {ls.kernel_name for ls in model.reachable_launches}
    # A @triton.jit *device function* is called by name from inside another kernel's
    # body (Triton inlines it), never [grid]-launched. It is live whenever a launched
    # kernel reaches it through the call graph -- and "launch it or remove it" would
    # break the caller. Expand the live set by that transitive closure.
    live = set(launched)
    frontier = list(launched)
    while frontier:
        name = frontier.pop()
        kernel = model.kernels.get(name)
        if kernel is None:
            continue
        for callee in kernel.calls:
            if callee not in live:
                live.add(callee)
                frontier.append(callee)

    dead = [name for name in model.kernels if name not in live]
    if not dead:
        return []

    # A file with no entry-point class at all cannot be loaded by KernelBench's
    # `getattr(module, "ModelNew")`, so every kernel is provably dead -- a hard fail (the
    # strongest F1.2 instance, and the one the check used to mute to a non-actionable info).
    # When we resolved the forward the benchmark enters we can likewise prove unreachability;
    # only a model_class whose forward we could *not* resolve stays info (BUG-25).
    severity = "fail" if (model.forward_entry or model.model_class is None) else "info"
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
