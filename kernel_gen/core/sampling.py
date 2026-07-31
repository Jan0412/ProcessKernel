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

The same stitch is where token traces are hardest, and for a reason worth stating
plainly: **the assembled string contains characters nobody generated.** ``PLAN_PREFIX``
is a prefill written into pass 1's prompt, ``CODE_FENCE`` is written into pass 2's, and
neither has a token, a logprob or a confidence value. Meanwhile pass 1's *token* ids run
past its truncated text, because vLLM keeps the tokens that spelled the stop string. So
the token arrays and the character offsets disagree at both ends of the seam, in
opposite directions, and neither is wrong -- they are answers to different questions.
:func:`generate_batch_traced` records both and reconciles neither, because the honest
reconciliation needs a tokenizer and would throw away the model's own decision to stop
planning and start coding.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import Backend, Completion
from .prompts import SYSTEM_PROMPT
from .trace import SEG_CODE, SEG_PLAN, TokenTrace, concat_passes, pack

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
    #: top-K alternatives to record per token; ``None`` = no trace, the old behaviour
    trace_topk: int | None = None


@dataclass
class TracedCompletion:
    """One slot's assembled completion, plus the trace behind it.

    ``trace`` is ``None`` whenever the backend had nothing to give -- an untraced run,
    or a backend that ignored ``trace_topk``. Callers treat that as "no trace", never
    as an error, so turning tracing off is a configuration change and not a code path.
    """

    text: str
    trace: TokenTrace | None = None


def generate_batch(backend: Backend, prompts: list[str], spec: SamplingSpec) -> list[str]:
    """One completion per user prompt, in order. Always ``n=1``.

    The whole batch goes to the backend in one call (two, with the think path), which
    is the property that makes a round over every active slot across every problem
    affordable. Callers must not loop this per problem.
    """
    return [out.text for out in generate_batch_traced(backend, prompts, spec)]


def generate_batch_traced(
    backend: Backend, prompts: list[str], spec: SamplingSpec
) -> list[TracedCompletion]:
    """:func:`generate_batch`, keeping the token trace behind each completion."""
    if not prompts:
        return []

    rendered = [backend.render_chat(spec.system, p) for p in prompts]
    k = spec.trace_topk

    if spec.think_temperature is None:
        singles = backend.complete_traced(
            rendered,
            temperature=spec.temperature,
            max_tokens=spec.max_new_tokens,
            logprobs=k,
        )
        return [_single_pass(single, k, spec) for single in singles]

    plan_prompts = [r + PLAN_PREFIX for r in rendered]
    plans = backend.complete_traced(
        plan_prompts,
        temperature=spec.think_temperature,
        max_tokens=spec.max_new_tokens,
        stop=[CODE_FENCE],
        logprobs=k,
    )
    if plans:
        print(f"  plan lengths (chars): {[len(p.text) for p in plans[:8]]}")
        _warn_on_truncated_plans(plans)

    continuations = [pp + plan.text + CODE_FENCE for pp, plan in zip(plan_prompts, plans)]
    codes = backend.complete_traced(
        continuations,
        temperature=spec.temperature,
        max_tokens=spec.max_new_tokens,
        logprobs=k,
    )
    return [_two_pass(plan, code, k, spec) for plan, code in zip(plans, codes)]


def _single_pass(single: Completion, k: int | None, spec: SamplingSpec) -> TracedCompletion:
    """``--think-temperature 0``: one call, one temperature, no seam to get wrong.

    Every token is labelled :data:`SEG_CODE` even though the completion still opens with
    prose. ``seg`` marks *which pass and temperature produced this token*, not what the
    text looks like -- and here there is one of each. A consumer that wants prose vs code
    in this mode has the fence position in the text and can find it there.
    """
    if not _traceable(single):
        return TracedCompletion(text=single.text)
    trace = pack(
        single.token_ids,
        single.topk,
        k=k or 0,
        seg=SEG_CODE,
        meta={
            "passes": 1,
            "n_plan_tokens": 0,
            "n_code_tokens": len(single.token_ids),
            "code_char_start": 0,
            "code_char_end": len(single.text),
            "code_temperature": spec.temperature,
            "code_finish_reason": single.finish_reason,
        },
    )
    return TracedCompletion(text=single.text, trace=trace)


def _two_pass(
    plan: Completion, code: Completion, k: int | None, spec: SamplingSpec
) -> TracedCompletion:
    """Stitch one slot's two halves, and write down exactly where the seams fell.

    The offsets below index the *assembled* string, which is what gets written to disk
    and what a line number from the linter refers to. They are the only way to get from
    "this finding is on line 40" back to "these tokens produced it", so they are
    recorded even for the two spans that have no tokens at all.
    """
    # CODE_FENCE carries no newline of its own, so when pass 2 opens mid-line the seam
    # reads ```pythonimport torch, no fence matches, and the import sharing that line is
    # dropped (KGEN-21). Offsets below come off `seam`, never CODE_FENCE.
    seam = CODE_FENCE if code.text.startswith("\n") else CODE_FENCE + "\n"
    text = PLAN_PREFIX + plan.text + seam + code.text
    if not (_traceable(plan) and _traceable(code)):
        return TracedCompletion(text=text)

    plan_start = len(PLAN_PREFIX)
    plan_end = plan_start + len(plan.text)
    code_start = plan_end + len(seam)

    trace = concat_passes(
        pack(plan.token_ids, plan.topk, k=k or 0, seg=SEG_PLAN),
        pack(code.token_ids, code.topk, k=k or 0, seg=SEG_CODE),
        meta={
            "passes": 2,
            # Character spans of the two generated halves within `text`. [0, plan_start)
            # is the PLAN_PREFIX prefill and [plan_end, code_start) is the CODE_FENCE --
            # both are prompt text, and no token in this trace corresponds to either.
            "plan_char_start": plan_start,
            "plan_char_end": plan_end,
            "code_char_start": code_start,
            "code_char_end": code_start + len(code.text),
            "plan_temperature": spec.think_temperature,
            "code_temperature": spec.temperature,
            # "stop" means the model reached the fence itself; "length" means it was cut
            # off mid-plan and a kernel was written from a truncated plan anyway. That
            # has always happened silently -- this is what makes it countable.
            "plan_finish_reason": plan.finish_reason,
            "code_finish_reason": code.finish_reason,
            "plan_stop_reason": plan.stop_reason,
            # Pass 1's token ids run past plan_char_end: vLLM keeps the tokens that
            # spelled the stop string even though it trims them out of the text. Those
            # trailing plan tokens are the model committing to start coding, which is
            # why they are kept rather than trimmed -- but they do NOT map into
            # [plan_char_start, plan_char_end), and anything aligning tokens to
            # characters has to detokenize to find the boundary.
            "plan_tokens_overrun_text": True,
        },
    )
    return TracedCompletion(text=text, trace=trace)


def _traceable(completion: Completion) -> bool:
    return bool(completion.token_ids)


def _warn_on_truncated_plans(plans: list[Completion]) -> None:
    """A plan that hit the token cap still gets a fence appended and a kernel written.

    Silently, until now. It is not an error -- the kernel may well be fine -- but it is
    a different experiment from the one the run claims to be doing, so it is counted.
    """
    truncated = sum(1 for p in plans if p.finish_reason == "length")
    if truncated:
        print(
            f"  [WARN] {truncated}/{len(plans)} plans hit --max-new-tokens before the "
            f"code fence; those kernels are written from a cut-off plan"
        )
