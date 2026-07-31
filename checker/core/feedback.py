"""Findings -> prompt text. Pure: no GPU, no model, no filesystem.

This lives in ``checker`` rather than in ``kernel_gen`` because the thing that knows
what a finding *means* is the package that raised it. A generation arm
should be able to hand a kernel to a renderer and get back text, without knowing that
F1.4 outranks F2.1.

Three decisions are load-bearing.

**Severity staging, not family staging.** Every check runs every round -- the analysis
is ~1ms, there is nothing to save by gating it. The *rendering* is what stages: if the
kernel has any ``fail``-severity finding, show only those and say nothing at all about
performance; show the ``warn``s only once there are no fails left. This gets "stop
cheating before you optimize" without spending a round on it -- a kernel with no F1
findings goes straight to F2 feedback in round 1. It is also simply correct on the
merits: advice about fusing memory traffic is worse than useless on a file that is
secretly calling ``torch.conv2d``.

**A file that does not parse is the most valuable feedback there is.** Today such a
generation is written to disk and quietly scores zero at eval. In a loop it is the
cheapest thing to fix and it gets its own block.

**The prompt stays Markov.** Base prompt + the latest kernel + the latest findings. No
transcript: it would blow the context window by round three and invite the model to
relitigate its own dead ends. The one piece of history worth carrying is "you were
told this last round and it is still here", which is what ``previous_check_ids`` marks
-- and which is also the honest signal that a round accomplished nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from .model import Finding, FileReport

#: How much of the review to show. ``severity`` is the default and the one the
#: experiment runs; the other two exist so the staging choice is ablatable rather than
#: baked in -- "we showed fails first" is a claim that should be falsifiable.
Policy = Literal["severity", "fails-only", "all"]

_HEADER = "## Automated review of your previous solution"

_INSTRUCTION = (
    "Rewrite the kernel to address the issues above. Return the COMPLETE corrected "
    "file in a single ```python block -- not a diff, not a fragment."
)

_REPEAT_MARK = " **(you were told this last round and it is still here)**"


def actionable(report: FileReport, policy: Policy = "severity") -> list[Finding]:
    """The findings this policy would actually show the model.

    ``info`` findings are never actionable: nothing is asked of the model, so letting
    one keep a slot alive would spend a GPU round saying "noted".
    """
    fails = [f for f in report.findings if f.severity == "fail"]
    warns = [f for f in report.findings if f.severity == "warn"]

    if policy == "fails-only":
        return fails
    if policy == "all":
        return fails + warns
    if policy == "severity":
        return fails if fails else warns
    raise ValueError(f"unknown feedback policy {policy!r}")


@dataclass(frozen=True)
class Feedback:
    """Prompt text, and exactly the check ids that appear in it.

    Two things downstream need to know what the model was *told*, not what *fired*: the
    repeat marker ("you were told this last round") and the readout's per-check table.
    Those differ whenever staging suppresses a warn, the cap truncates, or the submission
    gate replaces the lint text -- and only the renderer knows. So it says so, rather than
    leaving every consumer to guess from a summary that cannot express it.
    """

    text: str | None
    check_ids: frozenset[str]


class Renderer(ABC):
    """Report -> prompt text, or ``None`` when there is nothing worth another round.

    That ``None`` is the loop's early stop, so a renderer is not a formatting detail: it
    decides whether a slot spends another GPU round. There are two of them -- the linter's
    staged advice and the submission gate's blocking message -- and the critic composes
    them, which is what this base class exists to keep uniform.
    """

    @abstractmethod
    def feedback(
        self, report: FileReport, previous_check_ids: set[str] | None = None
    ) -> Feedback: ...

    def render(
        self, report: FileReport, previous_check_ids: set[str] | None = None
    ) -> str | None:
        """Text only. Derived, so text and ids cannot drift apart."""
        return self.feedback(report, previous_check_ids).text


class StagedRenderer(Renderer):
    """Severity staging: fails first, warns only once there are none, info never."""

    def __init__(self, max_findings: int = 8, policy: Policy = "severity") -> None:
        self.max_findings = max_findings
        self.policy = policy

    def feedback(
        self, report: FileReport, previous_check_ids: set[str] | None = None
    ) -> Feedback:
        if report.parse_status in ("syntax_error", "empty", "read_error"):
            # This block names no check, so it reports none -- the file never reached one.
            return Feedback(_render_broken(report), frozenset())

        findings = actionable(report, self.policy)
        if not findings:
            return Feedback(None, frozenset())

        # Selected once, then used for both the text and the ids: the cap and the staging
        # are exactly the steps every downstream reconstruction used to get wrong.
        shown = _shown(findings, self.max_findings)
        return Feedback(
            _render_findings(shown, previous_check_ids or set(), len(findings) - len(shown)),
            frozenset(f.check_id for f in shown),
        )


def render(
    report: FileReport,
    previous_check_ids: set[str] | None = None,
    max_findings: int = 8,
    policy: Policy = "severity",
) -> str | None:
    """Prompt-ready feedback, or ``None`` when there is nothing worth another round.

    That ``None`` is the loop's early stop: it is what makes a clean sample free.
    """
    return StagedRenderer(max_findings, policy).render(report, previous_check_ids)


def _render_broken(report: FileReport) -> str:
    if report.parse_status == "empty":
        detail = "Your previous answer contained no code at all."
    else:
        notes = report.summary.get("notes") or []
        reason = notes[0] if notes else "the file could not be parsed"
        detail = f"Your previous answer is not valid Python: {reason}."

    return (
        f"{_HEADER}\n\n"
        f"{detail}\n\n"
        f"Nothing else could be checked, because the file does not parse. Write the "
        f"kernel again from scratch and make sure the result is a single complete, "
        f"syntactically valid Python file inside one ```python block."
    )


def _shown(findings: list[Finding], max_findings: int) -> list[Finding]:
    """The findings the cap actually leaves in the prompt.

    Fails first, so the cap -- when it bites -- drops performance advice rather than a
    correctness violation.
    """
    fails = [f for f in findings if f.severity == "fail"][:max_findings]
    warns = [f for f in findings if f.severity == "warn"][: max_findings - len(fails)]
    return fails + warns


def _render_findings(shown: list[Finding], previous: set[str], n_hidden: int) -> str:
    fails = [f for f in shown if f.severity == "fail"]
    warns = [f for f in shown if f.severity == "warn"]

    lines = [_HEADER, ""]
    lines.append(
        "A static analyzer inspected the kernel above. It reports the following, and "
        "each one is a real defect in the code you wrote:"
    )
    lines.append("")

    if fails:
        lines.append("### Correctness -- these must be fixed")
        lines.append("")
        lines.extend(_bullets(fails, previous))
        lines.append("")

    if warns:
        # Only reachable when there are no fails (under the default policy), which is
        # the point: performance advice is meaningless on a kernel that is cheating.
        lines.append("### Performance -- these are wasted work")
        lines.append("")
        lines.extend(_bullets(warns, previous))
        lines.append("")

    if n_hidden > 0:
        lines.append(
            f"({n_hidden} further finding(s) omitted -- fix these first.)"
        )
        lines.append("")

    lines.append(_INSTRUCTION)
    return "\n".join(lines)


def _bullets(findings: list[Finding], previous: set[str]) -> list[str]:
    bullets = []
    for finding in findings:
        repeat = _REPEAT_MARK if finding.check_id in previous else ""
        bullets.append(f"- **{finding.check_id}** {finding.message}{repeat}")
    return bullets
