"""Single-config speculative-decoding benchmark worker (run as a subprocess).

Builds one vLLM target model with the requested speculative config, runs the
``generate_samples`` workload from ``kernel_gen/generate_kernels_samples.py`` over
a set of pre-built prompts, and writes throughput metrics as JSON.

Run in its own process (the notebook launches it via ``subprocess``) so all GPU
memory is reclaimed when it exits — that lets us benchmark several configs back to
back without fighting vLLM's in-process teardown.

Usage:
    python spec_bench_worker.py --config-file cfg.json --out-file out.json

The config JSON carries every parameter (see the notebook's ``make_configs``).
``spec_method`` is one of: "none" (baseline), "draft" (draft model), "ngram".
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def build_llm(cfg: dict):
    """Construct the target LLM with the requested speculative config.

    Prefers the modern ``speculative_config=`` API (as in the linked vLLM docs);
    falls back to the legacy top-level ``speculative_model=`` kwargs on older vLLM.
    """
    from vllm import LLM

    base_kwargs = dict(
        model=cfg["target_model"],
        dtype="auto",
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        enforce_eager=cfg.get("enforce_eager", False),
        # Keep stats on so vLLM logs "Speculative metrics: Draft acceptance rate ...".
        disable_log_stats=False,
    )

    method = cfg["spec_method"]
    k = cfg["num_speculative_tokens"]

    if method == "none":
        return LLM(**base_kwargs)

    if method == "draft":
        modern = {"model": cfg["draft_model"], "num_speculative_tokens": k}
        legacy = {"speculative_model": cfg["draft_model"], "num_speculative_tokens": k}
    elif method == "ngram":
        modern = {
            "method": "ngram",
            "num_speculative_tokens": k,
            "prompt_lookup_max": cfg["prompt_lookup_max"],
            "prompt_lookup_min": cfg["prompt_lookup_min"],
        }
        legacy = {
            "speculative_model": "[ngram]",
            "num_speculative_tokens": k,
            "ngram_prompt_lookup_max": cfg["prompt_lookup_max"],
            "ngram_prompt_lookup_min": cfg["prompt_lookup_min"],
        }
    else:
        raise ValueError(f"unknown spec_method: {method}")

    try:
        return LLM(speculative_config=modern, **base_kwargs)
    except TypeError as e:
        print(f"[worker] speculative_config API rejected ({e}); trying legacy kwargs", flush=True)
        return LLM(**base_kwargs, **legacy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--out-file", required=True)
    args = ap.parse_args()

    with open(args.config_file) as f:
        cfg = json.load(f)

    # Make generate_kernels_samples importable.
    sys.path.insert(0, cfg["gen_script_dir"])
    from generate_kernels_samples import generate_samples

    with open(cfg["prompts_file"]) as f:
        prompts = json.load(f)

    temperature = cfg["temperature"]
    num_samples = cfg["num_samples"]
    max_new = cfg["max_new_tokens"]

    t_load0 = time.perf_counter()
    llm = build_llm(cfg)
    load_s = time.perf_counter() - t_load0
    tok = llm.get_tokenizer()

    def count_output_tokens(texts):
        # Re-tokenize completions as a throughput proxy. Consistent across configs,
        # so comparisons are fair even if it differs from the exact generated count.
        return sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)

    # Warmup (untimed): triggers CUDA-graph capture / draft warmup so it doesn't
    # pollute the first timed run.
    print("[worker] warmup …", flush=True)
    _ = generate_samples(llm, prompts[0], max_new, num_samples, temperature)

    per_run = []
    for r in range(cfg["repeats"]):
        t0 = time.perf_counter()
        total_tokens = 0
        for prompt in prompts:
            texts = generate_samples(llm, prompt, max_new, num_samples, temperature)
            total_tokens += count_output_tokens(texts)
        dt = time.perf_counter() - t0
        per_run.append(
            {"wall_s": dt, "output_tokens": total_tokens, "tokens_per_s": total_tokens / dt}
        )
        print(f"[worker] repeat {r}: {total_tokens} tok in {dt:.1f}s = {total_tokens / dt:.1f} tok/s", flush=True)

    n = len(per_run)
    agg = {
        "label": cfg["label"],
        "spec_method": cfg["spec_method"],
        "draft_model": cfg.get("draft_model"),
        "num_speculative_tokens": cfg.get("num_speculative_tokens"),
        "temperature": temperature,
        "num_samples": num_samples,
        "max_new_tokens": max_new,
        "num_prompts": len(prompts),
        "repeats": n,
        "load_s": load_s,
        "wall_s_mean": sum(x["wall_s"] for x in per_run) / n,
        "output_tokens_mean": sum(x["output_tokens"] for x in per_run) / n,
        "tokens_per_s_mean": sum(x["tokens_per_s"] for x in per_run) / n,
        "per_run": per_run,
    }
    with open(args.out_file, "w") as f:
        json.dump(agg, f, indent=2)
    print("WORKER_DONE " + json.dumps({"label": agg["label"], "tokens_per_s_mean": round(agg["tokens_per_s_mean"], 1)}))


if __name__ == "__main__":
    main()
