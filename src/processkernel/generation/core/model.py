"""The objects generation passes around.

A :class:`Problem` is what the model is asked to solve; a :class:`Trajectory` is one
*sample slot* working on it. ``sample_id`` is assigned once and never renumbered, because
it is a primary key in the output filename and every downstream join (eval, reranker)
keys on the file stem.
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
class Attempt:
    """One generation for one slot.

    ``prompt`` is the exact user turn the model saw. It is captured so the trace can
    reconstruct the conversation without replaying the prompt builders against a pinned
    dataset; like ``trace`` it is deliberately absent from :meth:`to_dict`, so the journal
    stays small.

    ``trace`` is the token-level record when the run was started with ``--trace``, and
    ``None`` otherwise. It is never journaled -- it is arrays, and it goes to its own
    ``.npz``.
    """

    raw: str
    code: str
    trace: TokenTrace | None = None
    prompt: str = ""

    def to_dict(self) -> dict:
        return {"n_chars": len(self.code)}


@dataclass
class Trajectory:
    """One sample slot and the generation it produced."""

    problem: Problem
    sample_id: int
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def last(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts else None

    def to_dict(self) -> dict:
        out = {
            "level": self.problem.level,
            "problem_id": self.problem.problem_id,
            "sample_id": self.sample_id,
            "problem_name": self.problem.name,
        }
        if self.last is not None:
            out.update(self.last.to_dict())
        return out
