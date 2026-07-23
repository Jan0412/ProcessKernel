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
"""

from __future__ import annotations

import os


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
        )
        if load_in_4bit:
            kwargs["quantization"] = "bitsandbytes"
            kwargs["load_format"] = "bitsandbytes"

        print(f"Loading model {model_id} with vLLM …")
        self.llm = LLM(model=model_id, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        print("Model loaded.")

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
        from vllm import SamplingParams

        if not prompts:
            return []
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=stop,
            include_stop_str_in_output=False,
        )
        outputs = self.llm.generate(prompts, params)
        return [out.outputs[0].text for out in outputs]


class FakeBackend(Backend):
    """A scripted backend for tests: prompt substring -> completion.

    Keyed on *what it was asked*, never on call order, so batch ordering stays an
    engine implementation detail that no test accidentally pins. ``stop`` is honored
    (the completion is truncated at the first stop string) so the two-pass sampler's
    reassembly is exercised for real rather than stubbed around.
    """

    def __init__(self, rules: list[tuple[str, str]] | None = None, default: str = ""):
        self.rules = list(rules or [])
        self.default = default
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
        self.batches.append(list(prompts))
        completions = []
        for prompt in prompts:
            text = self.default
            for needle, completion in self.rules:
                if needle in prompt:
                    text = completion
                    break
            for stop_str in stop or []:
                text = text.split(stop_str, 1)[0]
            completions.append(text)
        return completions
