"""The lint critic: a kernel in, prompt-ready feedback out.

A factory returning a closure rather than a class, because there is exactly one
implementation and a base class shipping one implementation is ceremony.

``render`` is a *parameter*, not a hardcoded call, for one specific reason. The loop
optimizes the linter's own score by construction, so "findings went down" proves
nothing on its own -- the honest control is the identical loop with the lint content
stripped out of the feedback text, which would isolate "the advice helped" from "being
asked to try again helped". That control is not built here, and passing a different
``render`` is all it will take.

**Two analyzers, staged (KGEN-14).** ``clean`` ends the slot, so it has to mean the whole
answer is acceptable -- and the linter only ever answered "is this good Triton?". A file
that raises ``SyntaxError`` at import, or defines no ``ModelNew``, scores zero however
clean it lints. So the submission gate runs first and, when it has something to say, its
message *replaces* the lint feedback rather than joining it. That is the argument
``checker.core.feedback`` already makes about fails outranking warns, one level up:
advice about fusing memory traffic is worse than useless on a file Python cannot load.
"""

from __future__ import annotations

from typing import Callable

from checker import analyze_source
from checker.core.feedback import Feedback, Policy
from checker.core.feedback import feedback as feedback_findings
from checker.submission import SubmissionAnalyzer
from checker.submission.feedback import BlockingRenderer

from .model import Problem, Review


def lint_critic(
    feedback: Callable[..., Feedback] = feedback_findings,
    only: set[str] | None = None,
    policy: Policy = "severity",
    max_findings: int = 8,
) -> Callable[[Problem, str, set], Review]:
    """A critic that lints ``code`` and renders whatever it finds.

    Shapes come from the problem's own reference source, IN MEMORY. The obvious
    alternative -- ``analyze_file`` -- looks the reference up on disk under
    ``KernelBench/level{L}/``, a directory that does not exist for KernelBook at all;
    every shape-dependent F2 check would then silently lose its byte estimates and the
    performance feedback would quietly degrade to "something is wasteful". The
    reference is already in hand, so use it.
    """
    shape_cache: dict[tuple[int, int], list] = {}
    submission = SubmissionAnalyzer()
    blocking = BlockingRenderer()

    def critic(problem: Problem, code: str, previous_check_ids: set) -> Review:
        report = analyze_source(
            code,
            path="<generated>",
            only=only,
            fallback_shapes=_shapes(problem, shape_cache),
        )
        submission_report = submission.analyze(code, path="<generated>")
        blocked = blocking.feedback(submission_report, previous_check_ids)

        # ABLATION (one-off, revert after): the submission gate no longer feeds the
        # prompt, so the loop is linter-only and `clean` means "lints clean" again --
        # an unloadable file now ends its slot. `blocked` is still computed above, so
        # submission_ok and the S1 findings are recorded as before, just never shown.
        # shown = blocked if blocked.text is not None else feedback(
        shown = feedback(
            report,
            previous_check_ids=previous_check_ids,
            max_findings=max_findings,
            policy=policy,
        )
        text = shown.text
        summary = report.summary
        return Review(
            text=text or "",
            # Unchanged expression, stricter input: `clean` still means "nothing to say",
            # it is now asked of both analyzers.
            clean=text is None,
            data={
                "parse_status": report.parse_status,
                # The linter's counters stay the linter's. A submission defect must not
                # inflate n_fail, or every comparison against a historical run shifts
                # meaning; `submission_ok` is the one key that was added.
                "n_fail": summary.get("n_fail", 0),
                "n_warn": summary.get("n_warn", 0),
                "check_ids": summary.get("check_ids", []),
                "submission_ok": blocked.text is None,
                # What the prompt ACTUALLY contained, which `check_ids` above cannot say:
                # the gate replaces the lint text, severity staging hides warns behind
                # fails, and the cap truncates. Both readers of this journal -- the repeat
                # marker and the readout's per-check table -- need what was shown, not
                # what fired (KGEN-17, KGEN-18).
                "shown_check_ids": sorted(shown.check_ids),
            },
            # The full findings, carried past the summary that used to be all the loop
            # kept. Each one holds a `lineno` in its data, which is a verifier pointing
            # at the exact line it objects to -- free, exact error localization that
            # every later credit-assignment method would otherwise have to estimate.
            # It goes on `findings`, not into `data`, so `lint_loop.jsonl` stays the
            # size it was; see Review.to_dict.
            #
            # S1.* share the array with F1/F2 -- one schema for the PRM, told apart by id
            # prefix. Suppressed from the *prompt* while the file is unloadable, never
            # dropped from the record: what the linter saw is still what it saw.
            findings=[f.to_dict() for f in submission_report.findings]
            + [f.to_dict() for f in report.findings],
        )

    return critic


def _shapes(problem: Problem, cache: dict[tuple[int, int], list]) -> list:
    key = (problem.level, problem.problem_id)
    if key not in cache:
        from checker.lint.shapes import shapes_from_source

        try:
            cache[key] = shapes_from_source(problem.ref_arch_src)
        except Exception:  # noqa: BLE001 - shapes are best-effort, findings are not
            cache[key] = []
    return cache[key]
