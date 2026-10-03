"""F3.1 -- the kernel computes one element per program.

A Triton program is a block, not a thread: ``tl.arange`` is what gives it a vector to work
on. A launched kernel that loads and stores without ever building a range handles exactly
one element per program, so the block dimension is wasted and the grid has to be as large
as the output.
"""

from __future__ import annotations

import ast

from ....core.check import Check
from ....core.model import Finding, ModuleModel
from .. import LINT_REGISTRY


def _tl_calls(node: ast.AST) -> set[str]:
    """Names of ``tl.*`` / ``triton.language.*`` functions called anywhere in *node*."""
    out = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        base = call.func.value
        if isinstance(base, ast.Name) and base.id in ("tl", "triton"):
            out.add(call.func.attr)
        elif isinstance(base, ast.Attribute) and base.attr == "language":
            out.add(call.func.attr)
    return out


@LINT_REGISTRY.add
class UnvectorisedKernel(Check):
    check_id = "F3.1"
    name = "unvectorised_kernel"
    severity = "warn"

    def run(self, model: ModuleModel) -> list[Finding]:
        findings = []
        for name in sorted({ls.kernel_name for ls in model.reachable_launches}):
            kernel = model.kernels.get(name)
            if kernel is None:
                continue
            calls = _tl_calls(kernel.node)
            if "arange" in calls or not calls & {"load", "store"}:
                continue
            # A loop walks the tensor one element at a time; without one the kernel is an
            # epilogue reading a few precomputed scalars. Same defect, orders apart in cost,
            # so they must not compete for the same fix.
            looped = any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(kernel.node))
            where = (
                "and it walks its input inside a Python loop, one element per iteration"
                if looped
                else "though it only reads a few scalars, so the waste is bounded"
            )
            findings.append(
                self.finding(
                    f"`{name}` never builds a range, so each program loads, computes and "
                    f"stores a single element -- the block dimension of the program is "
                    f"unused {where}. Give the kernel a `BLOCK: tl.constexpr`, index with "
                    f"`tl.arange(0, BLOCK)` and mask the tail.",
                    severity="warn" if looped else "info",
                    kernel=name,
                    looped=looped,
                    lineno=kernel.lineno,
                )
            )
        return findings
