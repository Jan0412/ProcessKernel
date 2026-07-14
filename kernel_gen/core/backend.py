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
        max_model_len: int = 16384,
        trust_remote_code: bool = False,
        max_num_seqs: int = 32,
    ):
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
