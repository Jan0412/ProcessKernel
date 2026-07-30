"""The lint critic: a kernel in, prompt-ready feedback out.

A factory returning a closure rather than a class, because there is exactly one
implementation and a base class shipping one implementation is ceremony.

``render`` is a *parameter*, not a hardcoded call, for one specific reason. The loop
optimizes the linter's own score by construction, so "findings went down" proves
nothing on its own -- the honest control is the identical loop with the lint content
stripped out of the feedback text, which would isolate "the advice helped" from "being
asked to try again helped". That control is not built here, and passing a different
``render`` is all it will take.
"""

from __future__ import annotations

from typing import Callable

from checker import analyze_source
from checker.core.feedback import Policy
from checker.core.feedback import render as render_findings

from .model import Problem, Review


def lint_critic(
    render: Callable[..., str | None] = render_findings,
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

    def critic(problem: Problem, code: str, previous_check_ids: set) -> Review:
        report = analyze_source(
            code,
            path="<generated>",
            only=only,
            fallback_shapes=_shapes(problem, shape_cache),
        )
        text = render(
            report,
            previous_check_ids=previous_check_ids,
            max_findings=max_findings,
            policy=policy,
        )
        summary = report.summary
        return Review(
            text=text or "",
            clean=text is None,
            data={
                "parse_status": report.parse_status,
                "n_fail": summary.get("n_fail", 0),
                "n_warn": summary.get("n_warn", 0),
                "check_ids": summary.get("check_ids", []),
            },
            # The full findings, carried past the summary that used to be all the loop
            # kept. Each one holds a `lineno` in its data, which is a verifier pointing
            # at the exact line it objects to -- free, exact error localization that
            # every later credit-assignment method would otherwise have to estimate.
            # It goes on `findings`, not into `data`, so `lint_loop.jsonl` stays the
            # size it was; see Review.to_dict.
            findings=[f.to_dict() for f in report.findings],
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
