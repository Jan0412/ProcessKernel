"""The one seam in this package: something that turns prompts into completions.

This is the repo's first base class, and it earns that because it has two real
implementations dispatched over at runtime -- :class:`VLLMBackend`, which holds a
loaded model and needs a GPU, and :class:`FakeBackend`, which does not. Everything
above this line (the engine, the sampler, the critics, the artifact writer) is then
testable on a login node. Nothing else in the design cleared that bar, so nothing
else got a base class.

Plain ``NotImplementedError``, not ``abc.ABC``: nothing else in the repo imports
``abc``, and the only caller is one function in this package.

The interface is deliberately two methods and no ``n``. **One prompt per sample slot,
n=1, always.** A slot must be able to stop on its own round, which a single prompt
with ``n=10`` cannot express -- and it is already the shape the two-pass think path
uses, so nothing is lost.

:meth:`Backend.complete_traced` is the third method, added rather than substituted: it
returns the same completions plus the token-level internals a process reward model
needs. ``complete`` stays because most callers want a string and should not have to
know what a logprob is; ``complete_traced`` has a working default in terms of it, so a
backend that has no internals to offer implements nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Completion:
    """One completion, with whatever the backend could tell us about how it got there.

    Every field past ``text`` is optional because the base implementation cannot supply
    them, and *those fields are the entire reason this class exists*: until now
    ``complete`` returned ``out.outputs[0].text`` and dropped the rest of vLLM's
    ``CompletionOutput`` on the floor, which is the single line that made model
    confidence unobservable in this pipeline.

    ``topk[t]`` is ``[(token_id, logprob), ...]`` for step ``t``, in whatever order the
    backend produced -- :func:`kernel_gen.core.trace.pack` sorts it, because vLLM's is
    not the order it looks like.

    ``finish_reason`` is worth as much as the logprobs on the plan pass: ``"stop"``
    means the model reached the code fence on its own, ``"length"`` means it was cut off
    mid-plan and the sampler went on to append a fence and generate a kernel from a
    truncated plan anyway. That has always happened silently; recording it makes it
    countable.
    """

    text: str
    token_ids: list[int] | None = None
    topk: list[list[tuple[int, float]]] | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None


class Backend:
    def render_chat(self, system: str, user: str) -> str:
        """Apply the model's chat template, leaving the assistant turn open."""
        raise NotImplementedError

    def complete(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None = None,
    ) -> list[str]:
        """One completion per prompt, in order. Never batches across calls."""
        raise NotImplementedError

    def complete_traced(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None = None,
        logprobs: int | None = None,
    ) -> list[Completion]:
        """:meth:`complete`, plus token internals where the backend has them.

        The default degrades to text only, so ``logprobs`` is a request and never a
        requirement -- a backend that ignores it is still correct, and the trace writer
        treats a completion with no ``token_ids`` as an untraced one rather than an
        error.
        """
        return [
            Completion(text=text)
            for text in self.complete(
                prompts, temperature=temperature, max_tokens=max_tokens, stop=stop
            )
        ]


class VLLMBackend(Backend):
    """vLLM offline inference. Loads the model in ``__init__``."""

    def __init__(
        self,
        model_id: str,
        load_in_4bit: bool = False,
        gpu_memory_utilization: float = 0.92,
        # A one-shot prompt can reach ~16.4k tokens; 16384 left no room for output. 40960
        # fits the longest prompt plus a full 16384-token generation. KV cache is on
        # demand, so this does not inflate memory. See core/cli.py --max-model-len.
        max_model_len: int = 40960,
        trust_remote_code: bool = False,
        max_num_seqs: int = 32,
        max_logprobs: int = 20,
    ):
        # Disabled as a precaution while debugging Qwen3.6 GDN crashes; not
        # independently proven causal (the crash-relevant lever turned out to be
        # max_num_seqs), but harmless and left in place. setdefault so the shell can
        # still override.
        os.environ.setdefault("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", "0")

        from vllm import LLM

        # gpu_memory_utilization is the fraction of *total* VRAM vLLM may use for
        # weights + activations + KV cache. Large models need this high or the KV
        # cache budget goes negative and startup fails outright. Lower it when a
        # second process (e.g. a reranker) shares the GPU.
        #
        # max_num_seqs caps CUDA-graph capture sizes and the sampler warmup batch;
        # the 1024 default OOMs tight-memory models before a single token is emitted.
        num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
        kwargs = dict(
            dtype="auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=num_gpus,
            trust_remote_code=trust_remote_code,
            max_num_seqs=max_num_seqs,
            # The custom CUDA IPC all-reduce kernel fails with 'invalid argument' on
            # H100 nodes without NVLink IPC support. NCCL is always correct.
            disable_custom_all_reduce=True,
            # We only ever send text. On a VL checkpoint (e.g. Qwen3.6-27B, which vLLM
            # resolves to Qwen3_5ForConditionalGeneration) startup otherwise pushes a
            # dummy max-size image through the vision tower to size the encoder cache,
            # and the first GEMM in that path dies on an 80 GiB H100:
            #
            #     CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasCreate`
            #
            # cuBLAS allocates its handle workspace outside torch's caching allocator,
            # and the dummy encoder forward leaves it nothing. Zeroing every modality
            # limit drops the encoder profiling entirely. No-op on text-only models.
            language_model_only=True,
            # vLLM's default is "auto", which on Hopper resolves to flashinfer. We used
            # to pin triton here to skip flashinfer's ~9-minute cold JIT; that trade was
            # wrong. The Triton/FLA chunk kernel faults with 'unspecified launch failure'
            # on batches that mix prefills with peeled decodes -- the crash surfaces in
            # qwen_gdn_linear_attn.py's _forward_core, on the torch.cat stitching the
            # decode outputs onto chunk_gated_delta_rule's, which is the kernel this
            # setting picks. It killed jobs 2339959 and 2339985 hours into a run, always
            # in a repair round, where long prefills and decodes batch together.
            #
            # So this is now vLLM's own choice, stated explicitly. The JIT has had an
            # nvcc since 23fa6e6 and its result is cached after the first run. Falls
            # back to triton on non-Hopper. Ignored by models with no GDN layers.
            gdn_prefill_backend="flashinfer",
            # CUDA-graph capture is left at vLLM's default (FULL_AND_PIECEWISE). We used
            # to force cudagraph_mode=NONE here to dodge a vLLM 0.24 startup crash: the
            # "Profiling CUDA graph memory" phase ran dummy decode batches against a
            # minimal 64-block KV cache and a Qwen3.6 GDN kernel read past it, killing
            # the CUDA context ('unspecified launch failure', vllm-project/vllm#35743).
            # That is fixed as of 0.25.1 -- graph capture now completes cleanly for
            # Qwen3.6 -- so we take the decode speedup back. If a future vLLM regresses
            # the profiling phase for hybrid GDN models, restore cudagraph_mode=NONE.
            #
            # THE setting that would silently corrupt every trace if left implicit.
            # "raw" means the logprobs are read off the logits BEFORE the sampler's
            # processors -- before temperature, top-k and top-p. That matters here more
            # than in most pipelines, because the two-pass sampler deliberately runs the
            # plan at --think-temperature 1.0 and the code at --temperature 0.3-0.6.
            # Under "processed_logprobs" the code half would come back roughly twice as
            # peaked as the plan half no matter what the model actually knew, and every
            # plan-vs-code confidence comparison would be measuring a CLI flag. vLLM
            # already defaults to raw, but a default is not a contract; it is pinned
            # here and recorded into each trace's meta so a future flip is detectable in
            # data that has already been written.
            logprobs_mode="raw_logprobs",
            # Caps SamplingParams.logprobs. vLLM's default is also 20; stated so that
            # raising --trace-topk past it fails at startup rather than per request.
            max_logprobs=max_logprobs,
        )
        self.max_logprobs = max_logprobs
        if load_in_4bit:
            kwargs["quantization"] = "bitsandbytes"
            kwargs["load_format"] = "bitsandbytes"

        print(f"Loading model {model_id} with vLLM …")
        self.llm = LLM(model=model_id, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        # The model's output dimension, not the tokenizer's -- self-certainty is a KL
        # against the uniform distribution over what the softmax actually spans, and the
        # two differ by the padding rows most checkpoints carry.
        self.vocab_size = self.llm.llm_engine.model_config.get_vocab_size()
        print(f"Model loaded (vocab {self.vocab_size}, max_logprobs {max_logprobs}).")

    def render_chat(self, system: str, user: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def complete(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None = None,
    ) -> list[str]:
        return [
            completion.text
            for completion in self.complete_traced(
                prompts, temperature=temperature, max_tokens=max_tokens, stop=stop
            )
        ]

    def complete_traced(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None = None,
        logprobs: int | None = None,
    ) -> list[Completion]:
        """The real generate call. ``logprobs=None`` costs exactly what it used to.

        Note what ``token_ids`` contains when ``stop`` fires. vLLM truncates ``text`` at
        the stop string but *keeps every generated token id* -- ``detokenizer.update``
        re-appends the skipped stop token after excluding it from the text. So the plan
        pass returns a few more tokens than its text accounts for: the ones that spelled
        the fence the sampler is about to re-insert as a literal. They are real sampled
        tokens with real confidence and are kept, not trimmed. Trimming would need the
        tokenizer and would discard the model's own commitment to start coding, which is
        one of the more interesting tokens in the trace.
        """
        from vllm import SamplingParams

        if not prompts:
            return []
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=stop,
            include_stop_str_in_output=False,
            logprobs=logprobs,
        )
        outputs = self.llm.generate(prompts, params)
        return [_completion(out.outputs[0]) for out in outputs]


def _completion(output) -> Completion:
    """vLLM's ``CompletionOutput`` -> :class:`Completion`, flattening the logprob dicts.

    Each position arrives as ``{token_id: Logprob(logprob, rank, decoded_token)}``. Only
    the id and the logprob are carried across: ``rank`` is recoverable by sorting, and
    ``decoded_token`` is a string per token per position -- at 20 alternatives over
    thousands of tokens it is the largest thing in the object and the tokenizer can
    reproduce all of it from the ids on demand.

    A position is typed ``LogprobsOnePosition | None`` and becomes an empty row rather
    than a crash: this runs once per completion inside a job holding hours of GPU time,
    and losing one token's alternatives is not worth losing the run over.
    """
    return Completion(
        text=output.text,
        token_ids=list(output.token_ids),
        topk=(
            [
                [(token_id, lp.logprob) for token_id, lp in position.items()]
                if position
                else []
                for position in output.logprobs
            ]
            if output.logprobs is not None
            else None
        ),
        finish_reason=output.finish_reason,
        stop_reason=output.stop_reason,
    )


#: The fake's stand-in tokenizer: a fixed number of characters per "token". Nothing
#: depends on the value except how many tokens a fixture produces.
FAKE_CHARS_PER_TOKEN = 4
FAKE_VOCAB = 151936  # Qwen3's, so self-certainty comes out on the scale we will see


class FakeBackend(Backend):
    """A scripted backend for tests: prompt substring -> completion.

    Keyed on *what it was asked*, never on call order, so batch ordering stays an
    engine implementation detail that no test accidentally pins. ``stop`` is honored
    (the completion is truncated at the first stop string) so the two-pass sampler's
    reassembly is exercised for real rather than stubbed around.

    It also synthesizes token ids and top-K logprobs, which is not a convenience: the
    two-pass trace seam is the code most likely to be wrong and the least likely to be
    caught downstream, and ``sampling.py``'s whole design rests on the fake exercising
    that logic rather than tiptoeing around it. So the fake reproduces the two vLLM
    behaviours that make the seam hard, deterministically and without a GPU:

    * ``token_ids`` covers the stop string even though ``text`` is truncated before it,
      exactly as vLLM's detokenizer does;
    * each logprob row is ordered *sampled token first*, and is one entry wider than K
      whenever the sampled token ranked outside the top-K.
    """

    def __init__(self, rules: list[tuple[str, str]] | None = None, default: str = ""):
        self.rules = list(rules or [])
        self.default = default
        self.vocab_size = FAKE_VOCAB
        #: one entry per complete() call -- the batch it was given. Tests assert on
        #: len(batches) (round-major batching) and on the prompts within.
        self.batches: list[list[str]] = []

    def render_chat(self, system: str, user: str) -> str:
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

    def complete(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None = None,
    ) -> list[str]:
        return [
            completion.text
            for completion in self.complete_traced(
                prompts, temperature=temperature, max_tokens=max_tokens, stop=stop
            )
        ]

    def complete_traced(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None = None,
        logprobs: int | None = None,
    ) -> list[Completion]:
        self.batches.append(list(prompts))
        completions = []
        for prompt in prompts:
            text = self.default
            for needle, completion in self.rules:
                if needle in prompt:
                    text = completion
                    break

            emitted, stop_reason = text, None
            for stop_str in stop or []:
                head, sep, _ = text.partition(stop_str)
                if sep:
                    # Tokens run through the stop string; the text stops short of it.
                    text, emitted, stop_reason = head, head + stop_str, stop_str
                    break

            token_ids, topk = _fake_tokens(emitted, logprobs)
            completions.append(
                Completion(
                    text=text,
                    token_ids=token_ids,
                    topk=topk,
                    finish_reason="stop",
                    stop_reason=stop_reason,
                )
            )
        return completions


def _fake_tokens(
    text: str, k: int | None
) -> tuple[list[int], list[list[tuple[int, float]]] | None]:
    """Deterministic token ids and logprob rows for a string. Same text, same numbers."""
    import zlib

    chunks = [
        text[i : i + FAKE_CHARS_PER_TOKEN] for i in range(0, len(text), FAKE_CHARS_PER_TOKEN)
    ]
    token_ids = [zlib.crc32(chunk.encode()) % FAKE_VOCAB for chunk in chunks]
    if not k:
        return token_ids, None

    rows = []
    for chunk, sampled_id in zip(chunks, token_ids):
        seed = zlib.crc32(chunk.encode())
        alternatives = [((seed + 7919 * j) % FAKE_VOCAB, -0.1 - 0.7 * j) for j in range(k)]
        # seed % (k + 1) == k is the "sampled token fell outside the top-K" case, which
        # is the one that makes rows ragged and the sampled logprob unrecoverable from
        # the truncated array -- so the fake produces it about one token in K+1.
        index = seed % (k + 1)
        if index < k:
            alternatives[index] = (sampled_id, alternatives[index][1])
            row = [alternatives[index]] + [a for i, a in enumerate(alternatives) if i != index]
        else:
            row = [(sampled_id, -0.1 - 0.7 * k)] + alternatives
        rows.append(row)
    return token_ids, rows
