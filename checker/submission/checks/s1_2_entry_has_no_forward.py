"""S1.2 -- ``ModelNew`` exists but defines no ``forward``.

The benchmark constructs the class and calls it, which is ``forward``. Without one there
is nothing to time: ``nn.Module.__call__`` reaches ``nn.Module.forward``, which raises
``NotImplementedError``.

An inherited ``forward`` counts, but only from a base defined in this file -- an external
base's method is not something a static scan can see. That is a deliberate false positive
and it is the safe direction: `class ModelNew(nn.Module): pass` really does score zero.
"""

from __future__ import annotations

import ast

from ...core.check import Check
from ...core.model import Finding, ModuleModel
from ...core.parsing import inherited_forward
from . import SUBMISSION_REGISTRY
from .s1_1_no_entry_class import ENTRY


@SUBMISSION_REGISTRY.add
class EntryHasNoForward(Check):
    check_id = "S1.2"
    name = "entry_has_no_forward"
    severity = "fail"

    def run(self, model: ModuleModel) -> list[Finding]:
        if model.tree is None or model.compile_error is not None:
            return []

        cls = _entry_class(model.tree)
        if cls is None:
            return []  # no ModelNew at all is S1.1's finding, not a second one here

        if any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "forward"
            for n in cls.body
        ):
            return []

        functions = {
            f"{c.name}.{n.name}": n
            for c in model.tree.body
            if isinstance(c, ast.ClassDef)
            for n in c.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if inherited_forward(model.tree, cls.name, functions):
            return []

        return [
            self.finding(
                f"`{ENTRY}` defines no forward(). The benchmark calls the module, which "
                f"runs forward() -- without one there is nothing to execute and the "
                f"solution scores zero. Add `def forward(self, ...)` that launches your "
                f"Triton kernel and returns its result.",
                lineno=cls.lineno,
                cls=cls.name,
            )
        ]


def _entry_class(tree: ast.Module) -> ast.ClassDef | None:
    """The class the evaluator will actually instantiate, following one alias hop."""
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    if ENTRY in classes:
        return classes[ENTRY]

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == ENTRY for t in node.targets
        ):
            if isinstance(node.value, ast.Name):
                return classes.get(node.value.id)
    return None
