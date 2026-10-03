"""One generation per sample slot, all slots in one batch.

**One batch.** Every slot across every problem goes into ONE :func:`generate_batch_traced`
call. Looping per problem would pay vLLM's scheduling cost once per problem instead of
once per run; ``test_engine.py`` pins it.

Failure policy mirrors ``checker.build_model``: one bad generation must not abort a run
that has hours of GPU time in it. A completion whose code cannot be extracted is still
recorded, with empty code, and still written.
"""

from __future__ import annotations

from typing import Callable

from .backend import Backend
from .model import Attempt, Problem, Trajectory
from .sampling import SamplingSpec, TracedCompletion, generate_batch_traced
from .text import extract_code_block


def generate(
    backend: Backend,
    slots: list[tuple[Problem, int]],
    build_prompt: Callable[[Problem], str],
    spec: SamplingSpec,
) -> list[Trajectory]:
    """One trajectory per slot, each holding its single attempt."""
    trajectories = [Trajectory(problem=problem, sample_id=sid) for problem, sid in slots]
    if not trajectories:
        return trajectories

    print(
        f"\n=== {len(trajectories)} slots over "
        f"{len({t.problem.problem_id for t in trajectories})} problems ==="
    )
    prompts = [build_prompt(t.problem) for t in trajectories]
    completions = generate_batch_traced(backend, prompts, spec)

    if len(completions) != len(trajectories):  # a backend that reorders or drops is a hard bug
        raise RuntimeError(
            f"backend returned {len(completions)} completions for {len(trajectories)} prompts"
        )

    for traj, completion, prompt in zip(trajectories, completions, prompts):
        attempt = _attempt(traj.problem, completion)
        attempt.prompt = prompt
        traj.attempts.append(attempt)
    return trajectories


def _attempt(problem: Problem, completion: TracedCompletion) -> Attempt:
    """One completion -> one Attempt, degrading rather than raising."""
    raw = completion.text
    try:
        code = extract_code_block(raw)
    except Exception as exc:  # noqa: BLE001 - defensive by design
        print(f"[WARN] could not extract code for problem {problem.problem_id}: {exc}")
        code = ""
    # completion.trace is already None when tracing is off or the backend had nothing to give.
    return Attempt(raw=raw, code=code, trace=completion.trace)
