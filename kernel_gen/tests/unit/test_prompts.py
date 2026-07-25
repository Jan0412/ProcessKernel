"""``kernel_gen.core.prompts``: the system message and the repair prompt.

The repair prompt is what the whole lint-feedback loop rests on -- if it drops the
previous kernel or the findings, every round after round 0 is just a resample, a
different experiment than the one the run claims to make. It is pure and Markov, so it
is fully testable here.
"""

from __future__ import annotations

from kernel_gen.core.model import Attempt, Problem, Review
from kernel_gen.core.prompts import SYSTEM_PROMPT, build_base_prompt, build_repair_prompt


def _attempt(code: str, feedback: str) -> Attempt:
    return Attempt(round=0, raw=code, code=code, review=Review(text=feedback, clean=False))


# -- the base prompt (round 0) ---------------------------------------------


def test_base_prompt_embeds_the_reference_architecture():
    # build_base_prompt delegates to KernelBench's own constructor; the one thing that
    # must hold regardless is that the problem's reference source reaches the model.
    ref = "import torch\n\n\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x\n"
    problem = Problem(level=1, problem_id=1, name="1_Id.py", ref_arch_src=ref)

    out = build_base_prompt(problem, backend="triton")
    assert isinstance(out, str) and out
    assert "class Model" in out  # the reference is present, not dropped


# -- the system prompt -----------------------------------------------------


def test_system_prompt_asks_for_a_plan_then_a_fenced_block():
    # The two-pass sampler prefills "## Plan" to continue this paragraph and stops at the
    # fence this paragraph promises. Change one without the other and the split breaks.
    assert "plan" in SYSTEM_PROMPT.lower()
    assert "```python" in SYSTEM_PROMPT


# -- the repair prompt -----------------------------------------------------


def test_repair_prompt_carries_the_previous_kernel_and_the_findings():
    base = "SOLVE THIS PROBLEM"
    attempt = _attempt("import torch  # my kernel", "F1.2: you never launched the kernel")
    out = build_repair_prompt(base, attempt)

    assert base in out
    assert "import torch  # my kernel" in out  # the previous solution, quoted back
    assert "F1.2: you never launched the kernel" in out  # the findings
    assert "## Your previous solution" in out  # the marker the engine/tests key on


def test_repair_prompt_fences_the_previous_kernel_so_it_reads_as_code():
    out = build_repair_prompt("base", _attempt("x = 1", "fix it"))
    assert "```python\nx = 1\n```" in out


def test_repair_prompt_tolerates_a_missing_review():
    # A critic that crashed leaves review=None; the prompt must still build (empty
    # feedback) rather than raise inside a loop holding hours of GPU time.
    attempt = Attempt(round=0, raw="code", code="code", review=None)
    out = build_repair_prompt("base", attempt)
    assert "code" in out
    assert out.endswith("\n\n")  # feedback slot is empty, nothing after it
