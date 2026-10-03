"""The text the model sees: our system message and KernelBench's one-shot Triton prompt.

Both are exactly what the paper's appendix prints under "Generation Prompt". The PRM's
training runs also appended two blocks (``prompt_deltas``); by default nothing is added.
"""

from __future__ import annotations

from .model import Problem
from .prompt_deltas import apply_deltas

#: The "plan first" paragraph is what the two-pass sampler's ``## Plan`` prefill
#: continues -- change one and the other stops making sense.
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

BACKEND = "triton"
OPTION = "one_shot"


def build_base_prompt(problem: Problem, deltas: frozenset[str] = frozenset()) -> str:
    """KernelBench's own one-shot Triton prompt for ``problem``, plus any ``deltas``."""
    from kernelbench.prompt_constructor_toml import get_prompt_for_backend

    prompt = get_prompt_for_backend(
        ref_arch_src=problem.ref_arch_src,
        backend=BACKEND,
        option=OPTION,
        include_hardware=False,
        gpu_name=None,
    )
    return apply_deltas(prompt, problem, deltas)
