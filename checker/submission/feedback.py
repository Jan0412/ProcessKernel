"""Submission findings -> prompt text.

Not staged, unlike the linter's renderer: there is no ordering to negotiate, because every
S1.* finding is the same kind of thing -- the evaluator cannot load this file. It either
has something to say or it returns ``None``.

The header deliberately differs from the linter's. The model is not being told its kernel
is suboptimal; it is being told the file never ran.
"""

from __future__ import annotations

from ..core.feedback import Feedback, Renderer
from ..core.model import FileReport

_HEADER = "## Your previous solution could not be loaded"

_INSTRUCTION = (
    "Fix this first -- nothing else about the kernel was checked, because the file never "
    "ran. Return the COMPLETE corrected file in a single ```python block -- not a diff, "
    "not a fragment."
)


class BlockingRenderer(Renderer):
    def feedback(
        self, report: FileReport, previous_check_ids: set[str] | None = None
    ) -> Feedback:
        if not report.findings:
            return Feedback(None, frozenset())

        previous = previous_check_ids or set()
        lines = [
            _HEADER,
            "",
            "The evaluator loads your solution by executing the file and instantiating "
            "`ModelNew`. That failed, so the solution scored zero without running:",
            "",
        ]
        for finding in report.findings:
            repeat = (
                " **(you were told this last round and it is still here)**"
                if finding.check_id in previous
                else ""
            )
            lines.append(f"- **{finding.check_id}** {finding.message}{repeat}")
        lines.extend(["", _INSTRUCTION])
        # Nothing is staged and nothing is capped here, so what fired is what was shown.
        return Feedback(
            "\n".join(lines), frozenset(f.check_id for f in report.findings)
        )
