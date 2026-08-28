"""The text the model actually sees: system message, round-0 prompt, repair prompt.

The round-0 prompt is deliberately identical to what ``generate_kernels_samples.py``
sends -- same ``get_prompt_for_backend`` call, same system message -- so that round 0
of a refinement run is a valid *unrefined* baseline and not a subtly different arm.
``--dry-run`` exists to diff the two.

The repair prompt is MARKOV: base prompt + the latest kernel + the latest findings.
No transcript. Carrying the history would blow ``max_model_len`` after two rounds and,
worse, invites the model to argue with its own dead ends instead of rewriting. The one
piece of history worth keeping is "you were told this last round and it is still
there", and that is the renderer's job, not the prompt's.
"""

from __future__ import annotations

from .model import Attempt, Problem

#: Verbatim from ``generate_kernels_samples.py``. The "plan first" paragraph is what
#: the two-pass sampler's ``## Plan`` prefill continues -- change one and the other
#: stops making sense.
SYSTEM_PROMPT = (
    "You write custom kernels to replace the pytorch operators in the given "
    "architecture to get speedups.\n\n"
    "You have complete freedom to choose the set of operators you want to replace. "
    "You may replace some operators with custom kernels and leave others unchanged.\n\n"
    "Before writing any code, first think through and lay out a plan. Identify which "
    "operators are the most promising to replace, explain why, and describe the kernel "
    "strategy you intend to use. Keep this planning section concise.\n\n"
    "After you have written out the plan, implement it. You need to provide the "
    "complete Python code wrapped in a Python code block that starts with ```python "
    "and ends with ```."
)


def build_base_prompt(
    problem: Problem,
    backend: str = "triton",
    option: str = "one_shot",
    include_hardware: bool = False,
    gpu_name: str | None = None,
    deltas: frozenset[str] = frozenset(),
) -> str:
    """KernelBench's own prompt constructor, plus any enabled additive deltas."""
    from kernelbench.prompt_constructor_toml import get_prompt_for_backend

    from .prompt_deltas import apply_deltas

    hardware = include_hardware or "hardware" in deltas
    prompt = get_prompt_for_backend(
        ref_arch_src=problem.ref_arch_src,
        backend=backend,
        option=option,
        include_hardware=hardware,
        gpu_name=gpu_name if hardware else None,
    )
    return apply_deltas(prompt, problem, deltas)


def build_repair_prompt(base_prompt: str, attempt: Attempt) -> str:
    """Ask for a full rewrite, given the previous kernel and what a critic said.

    A full-file regeneration rather than a patch: the model is bad at emitting diffs,
    the file is small, and vLLM's prefix cache makes the shared base prompt nearly
    free to re-send. This is the shape ``generate_kernels_feedback.py`` already runs.
    """
    review = attempt.review
    feedback = review.text if review is not None else ""
    return (
        f"{base_prompt}\n\n"
        f"## Your previous solution\n\n"
        f"```python\n{attempt.code}\n```\n\n"
        f"{feedback}"
    )
