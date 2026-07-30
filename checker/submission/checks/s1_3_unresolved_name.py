"""S1.3 -- a known module alias is used as ``X.attr`` but never bound.

``class ModelNew(nn.Module)`` in a file that forgot ``import torch.nn as nn`` raises
``NameError`` at import. The AST is well formed, so the linter has nothing to say.

**Why an allowlist rather than "any unbound attribute root".** The general predicate was
run over the whole shipped corpus. It would flag 389 files, and its unresolved roots are:

    237 tl   196 triton   62 torch   55 nn   43 math    <- real, unambiguous module aliases
     19 self   2 bn   1 init   1 W   1 B   1 F   1 x    <- the false-positive surface

``self`` and ``x`` appearing at all proves a naive binder is wrong: it misses lambda
arguments, comprehension targets and walrus assignments, so it would mark working kernels
dirty and spend a GPU round on them. That is worse than the bug this package fixes. So the
check fires only on names that are never local variables -- the module aliases actually
evidenced -- and the general version is left to a later pass with real scope handling.

The check is file-wide rather than import-time-only: ``F.relu(x)`` inside ``forward``
fails when the benchmark calls it rather than when it loads it, but both score zero.
"""

from __future__ import annotations

import ast

from ...core.check import Check
from ...core.model import Finding, ModuleModel
from . import SUBMISSION_REGISTRY

#: Names that are module aliases by convention and never local variables.
MODULE_ALIASES = frozenset({"torch", "nn", "F", "tl", "triton", "np", "numpy", "math"})


@SUBMISSION_REGISTRY.add
class UnresolvedName(Check):
    check_id = "S1.3"
    name = "unresolved_name"
    severity = "fail"

    def run(self, model: ModuleModel) -> list[Finding]:
        if model.tree is None or model.compile_error is not None:
            return []

        bound = _bound_anywhere(model.tree)
        uses: dict[str, int] = {}
        for node in ast.walk(model.tree):
            if not isinstance(node, ast.Attribute):
                continue
            root = node.value
            if not isinstance(root, ast.Name):
                continue
            if root.id in MODULE_ALIASES and root.id not in bound:
                uses.setdefault(root.id, root.lineno)

        if not uses:
            return []

        names = sorted(uses)
        listed = ", ".join(f"`{n}`" for n in names)
        return [
            self.finding(
                f"{listed} {'is' if len(names) == 1 else 'are'} used but never imported, "
                f"so loading the file raises NameError and the solution scores zero "
                f"before any kernel runs. Add the missing import "
                f"(e.g. `import torch.nn as nn`, `import triton.language as tl`).",
                lineno=min(uses.values()),
                names=names,
            )
        ]


def _bound_anywhere(tree: ast.Module) -> set[str]:
    """Every name bound anywhere in the file, by any construct that binds one.

    Deliberately scope-blind and over-generous: a name bound in *some* scope is treated as
    bound everywhere. A missed binding is a false positive, and a false positive costs a
    working kernel a wasted repair round -- so the check errs towards silence.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            bound.update(_argument_names(node.args))
        elif isinstance(node, ast.Lambda):
            bound.update(_argument_names(node.args))
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            bound.update(node.names)
    return bound


def _argument_names(args: ast.arguments) -> set[str]:
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg:
        every.append(args.vararg)
    if args.kwarg:
        every.append(args.kwarg)
    return {a.arg for a in every}
