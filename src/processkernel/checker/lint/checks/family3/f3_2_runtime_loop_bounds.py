"""F3.2 -- a loop trip count that is only known at run time.

``for kd in range(K)`` with ``K`` an ordinary argument compiles to a real loop: a counter,
a compare and a branch per iteration, with the index arithmetic recomputed each time and
every address depending on loop state. Declaring the bound ``tl.constexpr`` makes Triton
specialise per value, so the same loop unrolls into straight-line loads the scheduler can
issue together.

Reported, not instructed. ``tl.constexpr`` pays off when the bound takes few distinct
values -- a filter window -- and costs a recompile per value when it does not, and the
linter cannot see the value set. So this states the compile-time fact and leaves the
judgement, the same way F2.1 reports a cost without naming a fusion it cannot prove safe.

The tiled-reduction idiom ``for k in range(0, K, BLOCK_K)`` is excluded: the bound there is
a data extent that *must* stay dynamic, and unrolling is not the question.
"""

from __future__ import annotations

import ast

from ....core.check import Check
from ....core.model import Finding, KernelDef, ModuleModel
from .. import LINT_REGISTRY


def _uses_arange(node: ast.AST) -> bool:
    for call in ast.walk(node):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            if call.func.attr == "arange":
                return True
    return False


def _runtime_bounds(kernel: KernelDef) -> list[tuple[str, int]]:
    """Single-argument ``range(<param>)`` bounds that are not constexpr, with their line."""
    out: list[tuple[str, int]] = []
    for node in ast.walk(kernel.node):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
            continue
        call = node.iter
        if not isinstance(call.func, ast.Name) or call.func.id != "range":
            continue
        # range(0, K, BLOCK) is a tiled reduction over a data extent, not a window.
        if len(call.args) != 1:
            continue
        bound = call.args[0]
        if not isinstance(bound, ast.Name):
            continue
        role = kernel.params.get(bound.id)
        if role is None or role.is_constexpr:
            continue
        out.append((bound.id, node.lineno))
    return out


@LINT_REGISTRY.add
class RuntimeLoopBounds(Check):
    check_id = "F3.2"
    name = "runtime_loop_bounds"
    severity = "info"

    def run(self, model: ModuleModel) -> list[Finding]:
        findings = []
        for name in sorted({ls.kernel_name for ls in model.reachable_launches}):
            kernel = model.kernels.get(name)
            # A kernel with no range at all is unvectorised, which is F3.1's finding; piling
            # this on top would split one fix across two messages.
            if kernel is None or not _uses_arange(kernel.node):
                continue
            found = _runtime_bounds(kernel)
            if not found:
                continue
            bounds = sorted({n for n, _ in found})
            names = ", ".join("`%s`" % b for b in bounds)
            many = len(bounds) > 1
            findings.append(
                self.finding(
                    f"`{name}` loops over {names}, which "
                    f"{'arrive as runtime arguments' if many else 'arrives as a runtime argument'}, "
                    f"so the trip count is unknown at compile time and Triton emits a real "
                    f"loop (counter, compare and branch per iteration, addresses that cannot "
                    f"be hoisted). If {'these take' if many else 'this takes'} only a few "
                    f"distinct values, declaring {'them' if many else 'it'} `tl.constexpr` "
                    f"lets the loop unroll; if the value set is large, the recompiles cost "
                    f"more than the unrolling saves.",
                    bounds=bounds,
                    depth=len(found),
                    kernel=name,
                    lineno=min(line for _, line in found),
                )
            )
        return findings
