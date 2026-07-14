"""How a batch of prompts becomes a batch of completions, including the think prefill.

This is a free function over :class:`~kernel_gen.core.backend.Backend`, NOT a method
on it. If the two-pass logic lived inside ``VLLMBackend``, then ``FakeBackend`` -- the
thing every test runs against -- would bypass exactly the code most likely to be
wrong. Here, the fake exercises it.

The two-pass trick: instruct-tuned models ignore "plan first" and emit the code fence
immediately. Prefilling the assistant turn with a ``## Plan`` heading forces them to
start in prose, which is then generated at a *higher* temperature than the code that
follows -- diverse strategies, careful implementations. Pass 1 stops at the fence;
pass 2 continues from it; the two halves are stitched back into one completion that
still looks like a normal fenced answer to ``extract_code_block``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import Backend
from .prompts import SYSTEM_PROMPT

CODE_FENCE = "```python"
PLAN_PREFIX = "## Plan\n"


@dataclass
class SamplingSpec:
    """Everything :func:`generate_batch` needs besides the prompts themselves.

    ``system`` lives here because the later reranked port needs a different system
    message with an otherwise identical sampler -- one field, not a refactor.
    """

    system: str = SYSTEM_PROMPT
    temperature: float = 0.8
    max_new_tokens: int = 2048
    #: enables the two-pass plan/code split; ``None`` = single pass
    think_temperature: float | None = None


def generate_batch(backend: Backend, prompts: list[str], spec: SamplingSpec) -> list[str]:
    """One completion per user prompt, in order. Always ``n=1``.

    The whole batch goes to the backend in one call (two, with the think path), which
    is the property that makes a round over every active slot across every problem
    affordable. Callers must not loop this per problem.
    """
    if not prompts:
        return []

    rendered = [backend.render_chat(spec.system, p) for p in prompts]

    if spec.think_temperature is None:
        return backend.complete(
            rendered, temperature=spec.temperature, max_tokens=spec.max_new_tokens
        )

    plan_prompts = [r + PLAN_PREFIX for r in rendered]
    plans = backend.complete(
        plan_prompts,
        temperature=spec.think_temperature,
        max_tokens=spec.max_new_tokens,
        stop=[CODE_FENCE],
    )
    if plans:
        print(f"  plan lengths (chars): {[len(p) for p in plans[:8]]}")

    continuations = [pp + plan + CODE_FENCE for pp, plan in zip(plan_prompts, plans)]
    codes = backend.complete(
        continuations, temperature=spec.temperature, max_tokens=spec.max_new_tokens
    )
    return [
        PLAN_PREFIX + plan + CODE_FENCE + code for plan, code in zip(plans, codes)
    ]
