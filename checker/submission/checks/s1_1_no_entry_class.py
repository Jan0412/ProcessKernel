"""S1.1 -- nothing named ``ModelNew`` is bound at module level.

The evaluator does the equivalent of ``exec(src, ns); ns["ModelNew"](...)``, so the only
thing that matters is whether that name exists after the module runs. A ``class ModelNew``
binds it; so does ``ModelNew = MyKernel``, and so does importing it. A class nested inside
another class does not.

The linter's ``resolve_entry_class`` accepts ``Model`` as well, which is correct when the
file being analysed is a KernelBench *reference* -- and wrong here. A submission named
``Model`` scores zero.
"""

from __future__ import annotations

import ast

from ...core.check import Check
from ...core.model import Finding, ModuleModel
from . import SUBMISSION_REGISTRY

ENTRY = "ModelNew"


def module_level_bindings(tree: ast.Module) -> set[str]:
    """Every name this module binds at top level, by any statement that binds one."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_bound_by(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            names.update(_bound_by(node.target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _bound_by(target: ast.expr) -> set[str]:
    """`a`, `a, b` and `a = b = c` all bind; `obj.attr` and `xs[0]` do not."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {n for elt in target.elts for n in _bound_by(elt)}
    if isinstance(target, ast.Starred):
        return _bound_by(target.value)
    return set()


@SUBMISSION_REGISTRY.add
class NoEntryClass(Check):
    check_id = "S1.1"
    name = "no_entry_class"
    severity = "fail"

    def run(self, model: ModuleModel) -> list[Finding]:
        # A file that does not compile never gets far enough to bind anything. S1.0
        # reports the cause; repeating it here as a missing class reports a consequence.
        if model.compile_error is not None:
            return []

        # `tree is None` is not the same question. The front end nulls it for an empty
        # source as well as an unparseable one -- but an empty file parses and compiles
        # fine, which is why S1.0 stays silent on it and why this check is the one that
        # owns it (see SubmissionAnalyzer.build). An empty module binds nothing, so it
        # falls through to the no-class branch below (KGEN-22).
        tree = model.tree if model.tree is not None else ast.Module(body=[], type_ignores=[])

        if ENTRY in module_level_bindings(tree):
            return []

        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        if classes:
            listed = ", ".join(f"`{c}`" for c in classes)
            detail = (
                f"this file defines {listed}. The evaluator instantiates `{ENTRY}` by "
                f"name, so rename your module class to `{ENTRY}`"
            )
        else:
            detail = (
                f"this file defines no class at all. The evaluator instantiates "
                f"`{ENTRY}`, an nn.Module whose forward() runs your kernel"
            )

        return [self.finding(f"There is no `{ENTRY}` to evaluate: {detail}.")]
