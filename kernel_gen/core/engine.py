"""The refinement loop: generate, review, repair, stop.

Not an agent. The control flow is fixed -- generate, then review, then repair -- so
there is no decision for a model to make and nothing for a tool-calling framework to
choose between. A driver loop that calls the critic itself is smaller, deterministic,
and keeps the one property an agent framework would have destroyed:

**Round-major batching.** A round collects every still-active slot across every
problem into ONE :func:`generate_batch` call. The existing arms loop per problem,
which is affordable exactly once; at 10 samples x 3 rounds it is not. This is the
engine's reason to exist, and ``test_engine.py`` pins it -- a future "just loop over
the problems" refactor must fail a test, not merely be slower.

The slot, not the problem, is the unit of work. A slot goes ``done`` as soon as its
critic says clean, so round 2's batch contains only what round 1 could not fix, and
the batch shrinks as the run proceeds. Slots of the same problem stop independently.

Failure policy mirrors ``triton_lint.build_model``: one bad generation must not abort
a run that has hours of GPU time in it. A completion that will not parse, or a critic
that raises, degrades that *attempt* -- it is still recorded, still written, and the
loop moves on.
"""

from __future__ import annotations

from typing import Callable

from .backend import Backend
from .model import Attempt, Problem, Review, Trajectory
from .sampling import SamplingSpec, generate_batch
from .text import extract_code_block

#: A critic sees a problem and the code one slot just produced, and returns a verdict:
#: feedback text, plus ``clean`` when there is nothing actionable to say.
#:
#: It always returns a Review -- "nothing to report" is ``Review(clean=True)``, not
#: ``None``. ``None`` is reserved for "the critic could not run at all", and conflating
#: the two would let a crashing critic read as a clean kernel and silently pass a
#: broken file straight through to eval.
#:
#: A callable, not a base class: there is one implementation. A second one that needs
#: state can be a callable object without touching this module.
Critic = Callable[[Problem, str], Review]


def run_rounds(
    backend: Backend,
    slots: list[tuple[Problem, int]],
    build_prompt: Callable[[Problem], str],
    build_repair_prompt: Callable[[Problem, Attempt], str],
    spec: SamplingSpec,
    critic: Critic | None = None,
    rounds: int = 1,
    on_round_end: Callable[[int, list[Trajectory]], None] | None = None,
) -> list[Trajectory]:
    """Run ``rounds`` rounds over ``slots``; return one trajectory per slot.

    With ``critic=None`` and ``rounds=1`` this degenerates to a plain generation -- one
    batch, one attempt per slot -- which is what makes round 0 a baseline arm rather
    than a special case.

    ``on_round_end`` is how a caller persists intermediates without the engine ever
    learning about the filesystem.
    """
    trajectories = [Trajectory(problem=problem, sample_id=sid) for problem, sid in slots]
    last_round = rounds - 1

    for round_index in range(rounds):
        active = [t for t in trajectories if not t.done]
        if not active:
            print(f"round {round_index}: every slot is done, stopping early")
            break

        print(
            f"\n=== round {round_index}/{last_round}: {len(active)} slots "
            f"over {len({t.problem.problem_id for t in active})} problems ==="
        )

        prompts = [
            build_prompt(t.problem)
            if t.last is None
            else build_repair_prompt(t.problem, t.last)
            for t in active
        ]
        raws = generate_batch(backend, prompts, spec)

        if len(raws) != len(active):  # a backend that reorders or drops is a hard bug
            raise RuntimeError(
                f"backend returned {len(raws)} completions for {len(active)} prompts"
            )

        n_clean = 0
        for traj, raw in zip(active, raws):
            attempt = _review(traj.problem, raw, round_index, critic)
            traj.attempts.append(attempt)

            clean = attempt.review is not None and attempt.review.clean
            n_clean += clean
            # A slot only carries into the next round if there is something to say to
            # it. No critic, or a critic that crashed, means no feedback -- and a repair
            # prompt with no feedback in it is just a resample, which is a different
            # experiment than the one this run is making a claim about.
            if clean or attempt.review is None or round_index == last_round:
                traj.done = True

        print(
            f"round {round_index}: {n_clean}/{len(active)} clean, "
            f"{sum(1 for t in trajectories if not t.done)} carry into the next round"
        )
        if on_round_end is not None:
            on_round_end(round_index, active)

    return trajectories


def _review(problem: Problem, raw: str, round_index: int, critic: Critic | None) -> Attempt:
    """One completion -> one Attempt, degrading rather than raising."""
    try:
        code = extract_code_block(raw)
    except Exception as exc:  # noqa: BLE001 - defensive by design
        print(f"[WARN] could not extract code for problem {problem.problem_id}: {exc}")
        code = ""

    attempt = Attempt(round=round_index, raw=raw, code=code)
    if critic is None:
        return attempt

    try:
        attempt.review = critic(problem, code)
    except Exception as exc:  # noqa: BLE001 - a critic must never abort a GPU run
        print(f"[WARN] critic raised on problem {problem.problem_id}: {exc}")
        attempt.review = None

    return attempt
