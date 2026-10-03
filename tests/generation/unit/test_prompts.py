"""``processkernel.generation.core.prompts``: the system message and the one-shot Triton prompt.

Both are what the paper's appendix prints under "Generation Prompt". The user prompt must
stay KernelBench's own, byte for byte.
"""

from __future__ import annotations

from processkernel.generation.core.model import Problem
from processkernel.generation.core.prompts import BACKEND, OPTION, SYSTEM_PROMPT, build_base_prompt

REF = "import torch\n\n\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x\n"


def test_base_prompt_embeds_the_reference_architecture():
    # The one thing that must hold regardless: the problem's reference reaches the model.
    problem = Problem(level=1, problem_id=1, name="1_Id.py", ref_arch_src=REF)

    out = build_base_prompt(problem)
    assert isinstance(out, str) and out
    assert "class Model" in out  # the reference is present, not dropped


def test_base_prompt_is_kernelbenchs_one_shot_triton_prompt_byte_for_byte():
    from kernelbench.prompt_constructor_toml import get_prompt_for_backend

    problem = Problem(level=1, problem_id=1, name="1_X.py", ref_arch_src=REF)
    stock = get_prompt_for_backend(ref_arch_src=REF, backend="triton", option="one_shot")
    assert (BACKEND, OPTION) == ("triton", "one_shot")
    assert build_base_prompt(problem) == stock


def test_deltas_are_appended_to_kernelbenchs_prompt_and_nothing_else_changes():
    from processkernel.generation.core.prompt_deltas import apply_deltas

    problem = Problem(level=6, problem_id=1, name="1_X.py", ref_arch_src=REF)
    both = frozenset({"contract", "precision"})
    out = build_base_prompt(problem, both)
    assert out == apply_deltas(build_base_prompt(problem), problem, both)
    assert out.startswith(build_base_prompt(problem).rstrip("\n"))
    assert "## Numerical precision" in out


# -- the system prompt -----------------------------------------------------


def test_system_prompt_asks_for_a_plan_then_a_fenced_block():
    # The two-pass sampler prefills "## Plan" to continue this paragraph and stops at the
    # fence this paragraph promises. Change one without the other and the split breaks.
    assert "plan" in SYSTEM_PROMPT.lower()
    assert "```python" in SYSTEM_PROMPT
