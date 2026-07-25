"""The four objects a generation arm passes around.

A :class:`Problem` is what the model is asked to solve; a :class:`Trajectory` is one
*sample slot* working on it across rounds. The slot -- not the attempt -- is the unit
of work: ``sample_id`` is assigned once and never renumbered, because it is a primary
key in the output filename and every downstream join (eval, sweep, reranker) keys on
the file stem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # numpy is not needed to define a Problem or run a dry run
    from .trace import TokenTrace


@dataclass(frozen=True)
class Problem:
    """One problem to solve, from either dataset.

    ``level`` is the integer that goes into the filename. For KernelBench it is the
    real level (1-3); for KernelBook it is the pseudo-level (5/6) -- the same number
    the conversion and eval scripts were told to use. ``ref_arch_src`` is a
    KernelBench-style reference either way (KernelBook rows are converted on load),
    which is what makes the linter, the shape inference and the prompt builder
    dataset-agnostic.
    """

    level: int
    problem_id: int
    name: str
    ref_arch_src: str


@dataclass
class Review:
    """A critic's verdict on one attempt.

    ``text`` is prompt-ready feedback. ``clean`` is the early-stop signal: nothing
    actionable was found, so this slot is done and costs nothing further.

    ``findings`` is the critic's full output -- every finding's check id, severity,
    message and line number. It is separate from ``data`` and deliberately absent
    from :meth:`to_dict`, because ``data`` is what lands in ``lint_loop.jsonl`` and that
    file is read start-to-finish by ``--skip-existing`` before every resumed run. The
    line numbers matter enormously later (they are free, verifier-supplied error
    localization) and not at all to the loop, so they go to the trace sidecar instead.
    """

    text: str
    clean: bool
    data: dict = field(default_factory=dict)
    findings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"clean": self.clean, **self.data}


@dataclass
class Attempt:
    """One generation for one slot in one round.

    ``prompt`` is the exact user turn the model saw this round -- the base prompt at
    round 0, the repair prompt (base + previous kernel + feedback) after. It is captured
    so the trace can reconstruct the whole conversation without replaying the prompt
    builders against a pinned dataset; like ``trace`` it is deliberately absent from
    :meth:`to_dict`, so ``lint_loop.jsonl`` stays byte-identical and ``--skip-existing``
    is untouched.

    ``trace`` is the token-level record when the run was started with ``--trace``, and
    ``None`` otherwise. It is never journaled -- it is arrays, and it goes to its own
    ``.npz``; see :meth:`to_dict`, which is unchanged by its presence.
    """

    round: int
    raw: str
    code: str
    review: Review | None = None
    trace: TokenTrace | None = None
    prompt: str = ""

    def to_dict(self) -> dict:
        out: dict = {"round": self.round, "n_chars": len(self.code)}
        if self.review is not None:
            out.update(self.review.to_dict())
        return out


def _rank(attempt: Attempt) -> tuple:
    """Sort key for "which attempt do we keep", lower is better.

    ``(does not parse, n_fail, n_warn, round)``. A non-parsing attempt can never beat
    a parsing one no matter how many findings the parsing one has -- a file that does
    not compile scores zero at eval. Ties break toward the earliest round, so a round
    that changed nothing measurable does not get credit for the previous round's work.
    """
    review = attempt.review
    if review is None:
        # No critic ran (or it crashed): all we can say is whether there is any code.
        return (not attempt.code.strip(), 0, 0, attempt.round)
    data = review.data
    parses = data.get("parse_status") in (None, "ok", "partial")
    return (
        not parses,
        int(data.get("n_fail", 0)),
        int(data.get("n_warn", 0)),
        attempt.round,
    )


@dataclass
class Trajectory:
    """One sample slot's history: what it generated, round by round."""

    problem: Problem
    sample_id: int
    attempts: list[Attempt] = field(default_factory=list)
    #: no further rounds for this slot -- it went clean, or it ran out of rounds
    done: bool = False

    @property
    def last(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts else None

    def final(self) -> Attempt | None:
        """The attempt to write to disk -- the loop's non-regression guard.

        The first clean attempt if there is one; otherwise the best attempt by
        :func:`_rank`. Refinement can make a kernel *worse*, and without this a run
        would silently ship round 3's regression over round 1's better answer.
        """
        if not self.attempts:
            return None
        for attempt in self.attempts:
            if attempt.review is not None and attempt.review.clean:
                return attempt
        return min(self.attempts, key=_rank)

    def to_dict(self) -> dict:
        final = self.final()
        return {
            "level": self.problem.level,
            "problem_id": self.problem.problem_id,
            "sample_id": self.sample_id,
            "problem_name": self.problem.name,
            "n_rounds": len(self.attempts),
            "final_round": final.round if final else None,
            "clean": bool(final and final.review and final.review.clean),
            "rounds": [a.to_dict() for a in self.attempts],
        }
