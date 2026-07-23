"""The two-pass trace seam -- the alignment nobody downstream would catch being wrong.

``test_sampling.py`` pins that the two halves of the *text* are stitched correctly.
This file pins the same thing for the *arrays*, which is harder for one reason: the
assembled string contains two spans that no token produced (the ``## Plan`` prefill and
the ``` ```python ``` fence), while pass 1 returns tokens that no kept character
accounts for (the ones that spelled the fence). Both facts are recorded rather than
patched over, so both get a test that says so out loud.
"""

from __future__ import annotations

from kernel_gen.core.backend import FakeBackend
from kernel_gen.core.sampling import (
    CODE_FENCE,
    PLAN_PREFIX,
    SamplingSpec,
    generate_batch,
    generate_batch_traced,
)
from kernel_gen.core.trace import SEG_CODE, SEG_PLAN, derive_scalars

PLAN = "fuse the elementwise ops\n"
CODE = "\nimport torch\n```\n"
TOPK = 8


def _model() -> FakeBackend:
    return FakeBackend(rules=[(PLAN, CODE)], default=PLAN + CODE_FENCE + CODE)


def _traced(spec_kwargs: dict | None = None):
    spec = SamplingSpec(think_temperature=1.0, temperature=0.3, trace_topk=TOPK)
    for key, value in (spec_kwargs or {}).items():
        setattr(spec, key, value)
    return generate_batch_traced(_model(), ["solve this"], spec)[0]


# ------------------------------------------------------------------ non-regression


def test_generate_batch_still_returns_plain_strings():
    # Every existing caller goes through this. It must be the old function in behaviour.
    out = generate_batch(_model(), ["solve this"], SamplingSpec(think_temperature=1.0))
    assert out == [PLAN_PREFIX + PLAN + CODE_FENCE + CODE]


def test_tracing_off_changes_nothing_about_the_text():
    off = generate_batch_traced(_model(), ["a", "b"], SamplingSpec(think_temperature=1.0))
    on = generate_batch_traced(
        _model(), ["a", "b"], SamplingSpec(think_temperature=1.0, trace_topk=TOPK)
    )

    assert [c.text for c in off] == [c.text for c in on]
    assert off[0].trace is not None  # ids are free even with no top-K requested
    assert off[0].trace.k == 0
    assert on[0].trace.k == TOPK


def test_the_think_path_is_still_exactly_two_backend_calls():
    backend = _model()
    generate_batch_traced(
        backend, [f"p{i}" for i in range(5)], SamplingSpec(think_temperature=1.0, trace_topk=4)
    )
    assert [len(b) for b in backend.batches] == [5, 5]


# ------------------------------------------------------------------------ the seam


def test_seg_flips_exactly_at_the_plan_code_boundary():
    trace = _traced().trace
    n_plan = trace.meta["n_plan_tokens"]

    assert trace.seg[:n_plan].tolist() == [SEG_PLAN] * n_plan
    assert set(trace.seg[n_plan:].tolist()) == {SEG_CODE}
    assert n_plan + trace.meta["n_code_tokens"] == len(trace)


def test_every_array_has_one_entry_per_token():
    trace = _traced().trace
    lengths = {
        array.shape[0]
        for array in (
            trace.token_ids,
            trace.topk_ids,
            trace.topk_lp,
            trace.sampled_lp,
            trace.sampled_rank,
            trace.seg,
        )
    }
    assert lengths == {len(trace)}


def test_the_char_offsets_reslice_the_completion_into_its_two_halves():
    # The join that later credit assignment depends on: a linter finding carries a line
    # number into THIS string, and these offsets are what map it back to a segment.
    out = _traced()
    meta = out.trace.meta

    assert out.text[meta["plan_char_start"] : meta["plan_char_end"]] == PLAN
    assert out.text[meta["code_char_start"] : meta["code_char_end"]] == CODE


def test_the_spans_no_token_produced_are_identifiable_from_the_offsets():
    # [0, plan_char_start) is the prefill and [plan_char_end, code_char_start) is the
    # fence. Both are prompt text. A consumer that assumed every character had a token
    # behind it would silently shift every offset by the length of these two strings.
    out = _traced()
    meta = out.trace.meta

    assert out.text[: meta["plan_char_start"]] == PLAN_PREFIX
    assert out.text[meta["plan_char_end"] : meta["code_char_start"]] == CODE_FENCE


def test_the_plan_half_has_more_tokens_than_its_text_and_says_so():
    # vLLM keeps the tokens that spelled the stop string. Those trailing tokens are the
    # model committing to start coding; they are kept, and the flag is what tells a
    # consumer not to expect a clean character mapping at the end of the plan segment.
    trace = _traced().trace
    text_only = FakeBackend(default=PLAN).complete_traced(
        ["x"], temperature=1.0, max_tokens=64
    )[0]

    assert trace.meta["plan_tokens_overrun_text"] is True
    assert trace.meta["n_plan_tokens"] > len(text_only.token_ids)


def test_both_temperatures_are_recorded_so_the_halves_stay_interpretable():
    # Confidence is captured pre-temperature (raw_logprobs), but the sampled TOKEN was
    # drawn post-temperature. Without both values on the trace, nobody can tell later
    # which regime a given token was drawn under.
    meta = _traced().trace.meta
    assert (meta["plan_temperature"], meta["code_temperature"]) == (1.0, 0.3)


def test_finish_reasons_are_recorded_for_both_passes():
    meta = _traced().trace.meta
    assert meta["plan_finish_reason"] == "stop"
    assert meta["plan_stop_reason"] == CODE_FENCE
    assert "code_finish_reason" in meta


def test_scalars_derive_cleanly_over_the_stitched_trace():
    trace = _traced().trace
    scalars = derive_scalars(trace.topk_lp, trace.sampled_lp, vocab_size=FakeBackend().vocab_size)

    assert {v.shape for v in scalars.values()} == {(len(trace),)}
    for name, values in scalars.items():
        assert values[trace.seg == SEG_PLAN].size > 0, name
        assert values[trace.seg == SEG_CODE].size > 0, name


# ------------------------------------------------------------------- the single pass


def test_single_pass_produces_one_segment_and_no_seam():
    out = generate_batch_traced(
        _model(), ["solve this"], SamplingSpec(think_temperature=None, trace_topk=TOPK)
    )[0]

    assert out.text == PLAN + CODE_FENCE + CODE
    assert out.trace.meta["passes"] == 1
    assert out.trace.meta["n_plan_tokens"] == 0
    assert set(out.trace.seg.tolist()) == {SEG_CODE}
    assert out.text[out.trace.meta["code_char_start"] : out.trace.meta["code_char_end"]] == out.text


# ---------------------------------------------------------------------- degradation


def test_a_backend_with_no_internals_yields_text_and_no_trace():
    from kernel_gen.core.backend import Backend

    class TextOnly(Backend):
        def render_chat(self, system, user):
            return user

        def complete(self, prompts, *, temperature, max_tokens, stop=None):
            return [PLAN] * len(prompts)

    out = generate_batch_traced(
        TextOnly(), ["x"], SamplingSpec(think_temperature=1.0, trace_topk=TOPK)
    )[0]

    assert out.trace is None  # "no trace", never an exception
    assert out.text == PLAN_PREFIX + PLAN + CODE_FENCE + PLAN


def test_empty_prompt_list_never_reaches_the_backend():
    backend = _model()
    assert generate_batch_traced(backend, [], SamplingSpec(trace_topk=TOPK)) == []
    assert backend.batches == []
