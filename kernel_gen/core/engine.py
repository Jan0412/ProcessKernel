"""The refinement loop: generate, review, repair, stop.

Not an agent. The control flow is fixed -- generate, then review, then repair -- so
there is no decision for a model to make and nothing for a tool-calling framework to
choose between. A driver loop that calls the critic itself is smaller, deterministic,
and keeps the one property an agent framework would have destroyed:

**Round-major batching.** A round collects every still-active slot across every
problem into ONE :func:`generate_batch_traced` call. The existing arms loop per problem,
which is affordable exactly once; at 10 samples x 3 rounds it is not. This is the
engine's reason to exist, and ``test_engine.py`` pins it -- a future "just loop over
the problems" refactor must fail a test, not merely be slower.

The slot, not the problem, is the unit of work. A slot goes ``done`` as soon as its
critic says clean, so round 2's batch contains only what round 1 could not fix, and
the batch shrinks as the run proceeds. Slots of the same problem stop independently.

Failure policy mirrors ``checker.build_model``: one bad generation must not abort
a run that has hours of GPU time in it. A completion that will not parse, or a critic
that raises, degrades that *attempt* -- it is still recorded, still written, and the
loop moves on.
"""

from __future__ import annotations

from typing import Callable

from .backend import Backend
from .model import Attempt, Problem, Review, Trajectory
from .sampling import SamplingSpec, TracedCompletion, generate_batch_traced
from .text import extract_code_block

#: A critic sees a problem, the code one slot just produced, and what it complained
#: about last round; it returns a verdict: feedback text, plus ``clean`` when there is
#: nothing actionable to say.
#:
#: The third argument is the only history that crosses a round boundary. The prompt
#: itself is Markov by design, but "you were told this last round and it is still
#: here" is worth saying -- and the engine is the only thing that knows it, because the
#: critic is stateless and sees one attempt at a time.
#:
#: It always returns a Review -- "nothing to report" is ``Review(clean=True)``, not
#: ``None``. ``None`` is reserved for "the critic could not run at all", and conflating
#: the two would let a crashing critic read as a clean kernel and silently pass a
#: broken file straight through to eval.
#:
#: A callable, not a base class: there is one implementation. A second one that needs
#: state can be a callable object without touching this module.
Critic = Callable[[Problem, str, set], Review]


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
        completions = generate_batch_traced(backend, prompts, spec)

        if len(completions) != len(active):  # a backend that reorders or drops is a hard bug
            raise RuntimeError(
                f"backend returned {len(completions)} completions for {len(active)} prompts"
            )

        n_clean = 0
        for traj, completion, prompt in zip(active, completions, prompts):
            attempt = _review(traj, completion, round_index, critic)
            # The exact user turn this slot saw this round, carried onto the attempt so a
            # trace can reconstruct the conversation. `prompts` was built from `active` in
            # the same order just above, so the zip is aligned.
            attempt.prompt = prompt
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


def _review(
    traj: Trajectory, completion: TracedCompletion, round_index: int, critic: Critic | None
) -> Attempt:
    """One completion -> one Attempt, degrading rather than raising."""
    problem = traj.problem
    raw = completion.text
    try:
        code = extract_code_block(raw)
    except Exception as exc:  # noqa: BLE001 - defensive by design
        print(f"[WARN] could not extract code for problem {problem.problem_id}: {exc}")
        code = ""

    # completion.trace is already None when tracing is off or the backend had nothing to
    # give -- carried through as-is, so an untraced run reaches exactly the code an
    # untraced run reached before.
    attempt = Attempt(round=round_index, raw=raw, code=code, trace=completion.trace)
    if critic is None:
        return attempt

    try:
        attempt.review = critic(problem, code, _previous_check_ids(traj))
    except Exception as exc:  # noqa: BLE001 - a critic must never abort a GPU run
        print(f"[WARN] critic raised on problem {problem.problem_id}: {exc}")
        attempt.review = None

    return attempt


def _previous_check_ids(traj: Trajectory) -> set[str]:
    """What the critic complained about in this slot's previous round, if any."""
    last = traj.last
    if last is None or last.review is None:
        return set()
    return set(last.review.data.get("check_ids") or [])
