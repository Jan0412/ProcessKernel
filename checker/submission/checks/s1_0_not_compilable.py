"""S1.0 -- ``compile(source, path, "exec")`` refuses the file.

The evaluator loads a submission by executing it. ``ast.parse`` succeeding is not enough
for that: ``compile`` additionally runs the symbol-table and codegen passes, which reject
source the parser accepts. Measured over the shipped corpus, 217 kernels parse but do not
compile -- 181 of them because one parameter name is repeated in a long kernel signature.

Deliberately not a list of error types. It hands the source to CPython and reports what
comes back, so a construct nobody thought of is caught the same as a duplicate argument.
"""

from __future__ import annotations

from ...core.check import Check
from ...core.model import Finding, ModuleModel
from . import SUBMISSION_REGISTRY


@SUBMISSION_REGISTRY.add
class NotCompilable(Check):
    check_id = "S1.0"
    name = "not_compilable"
    severity = "fail"

    def run(self, model: ModuleModel) -> list[Finding]:
        if model.compile_error is None:
            return []

        # Only SyntaxError and its subclasses carry a lineno. The key is omitted rather
        # than faked, because inspect_trace joins findings to source lines through it.
        lineno = model.compile_error_lineno
        data = {"lineno": lineno} if lineno else {}
        return [
            self.finding(
                f"The file is not valid Python and cannot be imported: "
                f"{model.compile_error}. The evaluator loads your solution with `exec`, "
                f"so this scores zero before anything runs. Return the complete "
                f"corrected file.",
                **data,
            )
        ]
