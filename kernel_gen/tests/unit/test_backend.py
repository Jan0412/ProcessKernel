"""The traced backend seam: ``complete_traced`` and what the fake has to reproduce.

``FakeBackend`` is the only backend the test suite can run, so anything it does not
model is untested in this repo. These tests pin the two vLLM behaviours the fake exists
to imitate -- token ids that outrun the truncated text, and logprob rows that are not
sorted -- because a fake that quietly gets either one right would hide a real bug.
"""

from __future__ import annotations

import pytest

from kernel_gen.core.backend import Backend, Completion, FakeBackend
from kernel_gen.core.trace import pack

PLAN = "fuse the elementwise ops\n"
FENCE = "```python"
CODE = "\nimport torch\n```\n"


def _model() -> FakeBackend:
    return FakeBackend(rules=[(PLAN, CODE)], default=PLAN + FENCE + CODE)


def test_complete_still_returns_plain_strings():
    # The old interface is the one every existing caller uses; it must not have moved.
    assert _model().complete(["x"], temperature=0.3, max_tokens=64) == [PLAN + FENCE + CODE]


def test_complete_and_complete_traced_agree_on_the_text():
    backend = _model()
    plain = backend.complete(["x"], temperature=0.3, max_tokens=64, stop=[FENCE])
    traced = backend.complete_traced(["x"], temperature=0.3, max_tokens=64, stop=[FENCE])

    assert plain == [traced[0].text] == [PLAN]


def test_the_base_class_degrades_to_text_only_rather_than_raising():
    # A backend with no internals to offer implements nothing and still works.
    class TextOnly(Backend):
        def complete(self, prompts, *, temperature, max_tokens, stop=None):
            return ["hello"] * len(prompts)

    out = TextOnly().complete_traced(["a", "b"], temperature=0.0, max_tokens=8)

    assert [c.text for c in out] == ["hello", "hello"]
    assert out[0].token_ids is None and out[0].topk is None


def test_no_logprobs_requested_means_no_logprobs_returned():
    out = _model().complete_traced(["x"], temperature=0.3, max_tokens=64)

    assert out[0].token_ids  # ids are free
    assert out[0].topk is None  # the expensive part is opt-in


def test_token_ids_cover_the_stop_string_that_the_text_does_not():
    # vLLM's detokenizer excludes the stop token from `text` and then re-appends it to
    # `token_ids`. So the plan pass returns MORE tokens than its text accounts for --
    # the ones that spelled the fence. Concatenating passes as if this were not true is
    # the misalignment the whole seam design is about.
    out = _model().complete_traced(["x"], temperature=1.0, max_tokens=64, stop=[FENCE])[0]

    assert out.text == PLAN
    assert out.stop_reason == FENCE

    # The count is the whole point: tokenizing the text we were GIVEN would produce
    # fewer tokens than we were HANDED, and the difference is the fence.
    text_only = FakeBackend(default=PLAN).complete_traced(["x"], temperature=1.0, max_tokens=64)[0]
    assert len(out.token_ids) > len(text_only.token_ids)


def test_logprob_rows_are_sampled_token_first_not_best_first():
    # The vLLM quirk that pack() has to undo. If the fake sorted its rows, the sort in
    # pack() would be dead code that no test could ever have exercised.
    out = _model().complete_traced(["x"], temperature=1.0, max_tokens=64, logprobs=8)[0]

    assert [row[0][0] for row in out.topk] == out.token_ids
    assert any(row[0][1] < row[1][1] for row in out.topk), "no row was out of order"


def test_some_rows_are_wider_than_k_because_the_sample_fell_outside():
    out = _model().complete_traced(["x"], temperature=1.0, max_tokens=64, logprobs=4)[0]

    widths = {len(row) for row in out.topk}
    assert widths <= {4, 5}
    assert 5 in widths, "the sampled-outside-top-K case never occurred"


def test_a_ragged_row_still_packs_rectangularly_with_its_sample_recorded():
    out = _model().complete_traced(["x"], temperature=1.0, max_tokens=64, logprobs=4)[0]
    trace = pack(out.token_ids, out.topk, k=4)

    assert trace.topk_ids.shape == (len(out.token_ids), 4)
    # Every token's own logprob survived, including the ones truncation dropped.
    assert (trace.sampled_rank > 0).all()
    assert (trace.sampled_lp < 0).all()


def test_the_fake_is_deterministic_so_fixtures_do_not_drift():
    first = _model().complete_traced(["x"], temperature=1.0, max_tokens=64, logprobs=4)[0]
    second = _model().complete_traced(["x"], temperature=1.0, max_tokens=64, logprobs=4)[0]

    assert first.token_ids == second.token_ids
    assert first.topk == second.topk


def test_one_completion_per_prompt_in_order():
    backend = FakeBackend(rules=[("alpha", "A"), ("beta", "B")], default="?")
    out = backend.complete_traced(
        ["alpha", "beta", "gamma"], temperature=0.3, max_tokens=8, logprobs=2
    )

    assert [c.text for c in out] == ["A", "B", "?"]
    assert len(backend.batches) == 1


def test_completion_defaults_leave_every_internal_absent():
    bare = Completion(text="hi")
    assert (bare.token_ids, bare.topk, bare.finish_reason, bare.stop_reason) == (
        None,
        None,
        None,
        None,
    )


# -- VLLMBackend without a GPU ---------------------------------------------
#
# VLLMBackend was the least-covered module in core (70%): everything inside it needs a
# loaded model. But the parts most worth pinning are pure wiring -- WHICH kwargs reach
# LLM(), and how a CompletionOutput is mapped -- and those are testable by injecting a
# fake `vllm` module before the lazy `from vllm import ...` inside each method. The
# kwargs assertions are the real value: logprobs_mode is the setting that silently
# corrupts every trace if it regresses, and there is no other test that would catch it.

import sys
import types


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeCompletionOutput:
    def __init__(self, text="out", token_ids=(1, 2), logprobs=None,
                 finish_reason="stop", stop_reason=None):
        self.text = text
        self.token_ids = list(token_ids)
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason


class _FakeRequestOutput:
    def __init__(self, completion):
        self.outputs = [completion]


class _FakeLLM:
    """Records the kwargs it was constructed with; returns scripted outputs."""

    last_kwargs: dict = {}

    def __init__(self, model=None, **kwargs):
        _FakeLLM.last_kwargs = {"model": model, **kwargs}
        self.generate_calls = []
        self.llm_engine = types.SimpleNamespace(
            model_config=types.SimpleNamespace(get_vocab_size=lambda: 151936)
        )

    def get_tokenizer(self):
        return types.SimpleNamespace(
            apply_chat_template=lambda msgs, tokenize, add_generation_prompt: (
                f"<chat>{msgs[0]['content']}|{msgs[1]['content']}"
            )
        )

    def generate(self, prompts, params):
        self.generate_calls.append((list(prompts), params))
        return [_FakeRequestOutput(_FakeCompletionOutput(text=f"reply::{p}")) for p in prompts]


@pytest.fixture
def vllm_backend(monkeypatch):
    """A VLLMBackend built against a fake vllm module -- no GPU, no real import."""
    fake = types.ModuleType("vllm")
    fake.LLM = _FakeLLM
    fake.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    from kernel_gen.core.backend import VLLMBackend

    return VLLMBackend("some/model", max_model_len=4096, max_num_seqs=8, max_logprobs=12)


def test_vllm_backend_pins_the_settings_that_would_silently_corrupt_traces(vllm_backend):
    kw = _FakeLLM.last_kwargs
    # THE one that matters: processed_logprobs would bake the sampling temperature into
    # every recorded logprob, making the plan/code halves incomparable forever.
    assert kw["logprobs_mode"] == "raw_logprobs"
    assert kw["max_logprobs"] == 12
    # VL checkpoints die at cublasCreate during vision-tower profiling without this.
    assert kw["language_model_only"] is True
    # NCCL is always correct; the custom IPC all-reduce fails on H100s without NVLink.
    assert kw["disable_custom_all_reduce"] is True
    assert kw["gdn_prefill_backend"] == "flashinfer"


def test_tensor_parallel_size_follows_cuda_visible_devices(vllm_backend):
    assert _FakeLLM.last_kwargs["tensor_parallel_size"] == 2  # "0,1"
    assert _FakeLLM.last_kwargs["max_model_len"] == 4096
    assert _FakeLLM.last_kwargs["max_num_seqs"] == 8


def test_vocab_size_is_read_from_the_model_not_the_tokenizer(vllm_backend):
    # self-certainty is a KL against the uniform over the SOFTMAX width; the tokenizer's
    # length differs by the padding rows most checkpoints carry.
    assert vllm_backend.vocab_size == 151936


def test_render_chat_delegates_to_the_models_own_template(vllm_backend):
    assert vllm_backend.render_chat("SYS", "USER") == "<chat>SYS|USER"


def test_complete_traced_requests_flat_logprobs_only_when_tracing(vllm_backend):
    vllm_backend.complete_traced(["p"], temperature=0.6, max_tokens=32, logprobs=20)
    params = vllm_backend.llm.generate_calls[-1][1]
    assert params.kwargs["logprobs"] == 20
    assert params.kwargs["flat_logprobs"] is True  # avoids ~84M dataclasses at scale
    assert params.kwargs["n"] == 1
    assert params.kwargs["include_stop_str_in_output"] is False

    vllm_backend.complete_traced(["p"], temperature=0.6, max_tokens=32)
    untraced = vllm_backend.llm.generate_calls[-1][1]
    assert untraced.kwargs["logprobs"] is None
    assert untraced.kwargs["flat_logprobs"] is False  # no-op when not tracing


def test_complete_returns_text_and_complete_traced_returns_completions(vllm_backend):
    assert vllm_backend.complete(["a"], temperature=0.3, max_tokens=8) == ["reply::a"]
    traced = vllm_backend.complete_traced(["a"], temperature=0.3, max_tokens=8)
    assert traced[0].text == "reply::a"
    assert traced[0].token_ids == [1, 2]
    assert traced[0].finish_reason == "stop"


def test_an_empty_prompt_list_never_reaches_the_model(vllm_backend):
    before = len(vllm_backend.llm.generate_calls)
    assert vllm_backend.complete_traced([], temperature=0.3, max_tokens=8) == []
    assert len(vllm_backend.llm.generate_calls) == before


# -- the two logprob container shapes, in-process --------------------------


def test_topk_reads_the_legacy_dict_container():
    from kernel_gen.core.backend import _topk

    lp = types.SimpleNamespace  # a stand-in for vllm.Logprob
    position = {7: lp(logprob=-0.2), 3: lp(logprob=-1.2)}
    assert _topk([position]) == [[(7, -0.2), (3, -1.2)]]


def test_topk_reads_the_flat_container_by_slicing_its_lists():
    from kernel_gen.core.backend import _topk

    flat = types.SimpleNamespace(
        start_indices=[0, 2], end_indices=[2, 3],
        token_ids=[7, 3, 9], logprobs=[-0.2, -1.2, -0.5],
    )
    assert _topk(flat) == [[(7, -0.2), (3, -1.2)], [(9, -0.5)]]


def test_a_none_position_becomes_an_empty_row_not_a_crash():
    from kernel_gen.core.backend import _topk

    # A position is typed LogprobsOnePosition | None; losing one token's alternatives
    # must not lose the run.
    assert _topk([None]) == [[]]


# -- the base class contract -----------------------------------------------


def test_the_base_backend_requires_its_two_methods():
    bare = Backend()
    with pytest.raises(NotImplementedError):
        bare.render_chat("s", "u")
    with pytest.raises(NotImplementedError):
        bare.complete(["p"], temperature=0.1, max_tokens=1)


def test_load_in_4bit_switches_the_weight_loader(monkeypatch):
    fake = types.ModuleType("vllm")
    fake.LLM = _FakeLLM
    fake.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake)

    from kernel_gen.core.backend import VLLMBackend

    VLLMBackend("m", load_in_4bit=True)
    assert _FakeLLM.last_kwargs["quantization"] == "bitsandbytes"
    assert _FakeLLM.last_kwargs["load_format"] == "bitsandbytes"
