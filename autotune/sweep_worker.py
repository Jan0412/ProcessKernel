"""Evaluate one (kernel, config) pair and print the result as JSON. Runs as its own process.

This is deliberately a separate process per evaluation, not a function call in a pool:

  * a bad launch config can raise an illegal memory access, which poisons the CUDA context
    for every subsequent evaluation in that process -- one wrong tile size would corrupt
    the rest of the kernel's sweep and we would never notice;
  * a Triton compile can hang, and a hung thread is not killable from Python. A process is.

The cost is a torch import per evaluation (~8s), which is the price of trustworthy numbers.

The caller masks CUDA_VISIBLE_DEVICES to a single GPU before spawning us, so cuda:0 here is
whichever physical device the scheduler assigned. eval_kernel_against_ref sets
CUDA_VISIBLE_DEVICES itself for the triton backend; inside our already-masked environment
that write is a no-op.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-file", required=True)
    ap.add_argument("--kernel-file", required=True, help="the PATCHED kernel source")
    ap.add_argument("--num-correct-trials", type=int, required=True)
    ap.add_argument("--num-perf-trials", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch
        from kernelbench.eval import eval_kernel_against_ref

        with open(args.ref_file) as f:
            ref_src = f.read()
        with open(args.kernel_file) as f:
            kernel_src = f.read()

        result = eval_kernel_against_ref(
            original_model_src=ref_src,
            custom_model_src=kernel_src,
            seed_num=args.seed,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            measure_performance=True,
            timing_method="cuda_event",
            backend="triton",
            precision=torch.float32,
            device=torch.device("cuda:0"),
            verbose=False,
            # We compare configs of one kernel against each other, and the reference time is
            # already known from the baseline JSON. Re-timing it on every config would double
            # the sweep's cost for nothing.
            check_for_excessive_speedup=False,
        )
    except Exception as e:
        print(json.dumps({
            "compiled": False, "correct": False, "runtime_ms": None,
            "error": f"{type(e).__name__}: {e}", "error_kind": "worker_crash",
            "traceback": traceback.format_exc()[-2000:],
        }))
        return 0  # a crash is data, not a driver failure -- the parent records and moves on

    if result is None:  # eval signals "retry" (compile lock contention) this way
        print(json.dumps({
            "compiled": False, "correct": False, "runtime_ms": None,
            "error": "eval returned None (lock contention)", "error_kind": "retry",
        }))
        return 0

    meta = result.metadata or {}
    print(json.dumps({
        "compiled": bool(result.compiled),
        "correct": bool(result.correctness),
        "runtime_ms": float(result.runtime) if result.runtime and result.runtime > 0 else None,
        "runtime_stats": result.runtime_stats or {},
        "error": meta.get("runtime_error") or meta.get("compilation_error"),
        "error_kind": _classify(result, meta),
        "correctness_trials": meta.get("correctness_trials"),
        "max_difference": meta.get("max_difference"),
        "hardware": (result.runtime_stats or {}).get("hardware"),
    }))
    return 0


def _classify(result, meta: dict) -> str | None:
    """Why did this config fail? The distinction matters downstream.

    'wrong_result' is the interesting one: the kernel compiled and ran but produced the
    wrong answer at this block size, which means its correctness silently depended on the
    constant the model happened to pick. That is a latent bug we feed back to the model.
    """
    if result.correctness:
        return None
    if not result.compiled:
        return "compile_error"
    if meta.get("runtime_error"):
        return "runtime_error"
    return "wrong_result"


if __name__ == "__main__":
    sys.exit(main())
