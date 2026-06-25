"""Build the labeled reranker dataset from KernelBench evaluation runs.

Reads, for each configured run directory:
  - eval_results.json          {problem_id: [{sample_id, compiled, correctness, runtime, ...}]}
  - staged kernel sources      level_{L}_problem_{P}_sample_{S}_kernel.py
and joins them with:
  - the reference architecture source (from the KernelBench dataset)
  - the per-problem PyTorch baseline runtime (`data.baseline_timing_json`) to
    compute a `speedup = baseline / kernel_runtime` for each correct kernel
    (KernelBench `fast_p` style; None when the kernel is wrong or has no baseline).

Label: positive (1) iff compiled AND correct; negative (0) otherwise.

Emits one JSONL row per evaluated kernel candidate to `data.dataset_jsonl`
(fields include `runtime`, `runtime_min`, `runtime_std`, `speedup`, `label`).

Usage:
    bash scripts/build_dataset.sh [configs/default.yaml]
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

from reranker.src.config import PROJECT_ROOT, RerankerConfig, _resolve, load_config
from reranker.src.data.labels import compute_label


def _add_kernelbench_to_path(kernelbench_dir: str) -> None:
    src = os.path.join(_resolve(kernelbench_dir), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_baseline_times(path: str) -> dict[int, dict[int, float]]:
    """Load the PyTorch-eager baseline runtimes as ``{level: {problem_id: time_ms}}``.

    ``time_ms`` is the baseline's mean wall-clock runtime. The KernelBench timing
    JSON is keyed by ``"level1"`` ... and, within each level, by problem filename
    (``"1_Square_matrix_multiplication_.py"``) whose integer prefix is the problem
    id. Returns ``{}`` (and warns) if the file is missing, so ``speedup`` simply
    becomes ``None`` everywhere.
    """
    if not os.path.isfile(path):
        print(f"[WARN] baseline timing file not found: {path} — speedup will be None for all rows")
        return {}
    raw = _load_json(path)
    out: dict[int, dict[int, float]] = {}
    for level_key, problems in raw.items():
        m = re.match(r"level(\d+)$", str(level_key))
        if not m or not isinstance(problems, dict):
            continue
        per_problem: dict[int, float] = {}
        for fname, stats in problems.items():
            pm = re.match(r"(\d+)_", str(fname))
            if pm and isinstance(stats, dict) and stats.get("mean") is not None:
                per_problem[int(pm.group(1))] = float(stats["mean"])
        out[int(m.group(1))] = per_problem
    return out


def _staged_kernel_path(run_dir: str, level: int, problem_id: int, sample_id: int) -> str:
    return os.path.join(
        run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
    )


def build_dataset(cfg: RerankerConfig) -> str:
    """Build the JSONL dataset and return its path."""
    _add_kernelbench_to_path(cfg.data.kernelbench_dir)
    from kernelbench.dataset import construct_kernelbench_dataset, fetch_ref_arch_from_dataset

    data_cfg = cfg.data
    if data_cfg.negative_mode not in ("all_negative", "compiled_wrong"):
        raise ValueError(
            f"data.negative_mode must be 'all_negative' or 'compiled_wrong', got {data_cfg.negative_mode}"
        )
    levels = data_cfg.levels_for_run_dirs()
    kb_base = os.path.join(_resolve(data_cfg.kernelbench_dir), "KernelBench")

    out_path = _resolve(data_cfg.dataset_jsonl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Per-problem PyTorch baseline runtimes for the speedup grade (listwise).
    baseline_times = _load_baseline_times(_resolve(data_cfg.baseline_timing_json))

    # Cache KernelBench datasets per level (run_dirs may share a level).
    kb_datasets: dict[int, object] = {}

    reason_counts: Counter = Counter()
    level_counts: Counter = Counter()
    rows_written = 0
    missing_kernel = 0
    dropped_compile_fail = 0
    correct_with_speedup = 0
    correct_without_baseline = 0

    with open(out_path, "w") as out_f:
        for run_dir_rel, level in zip(data_cfg.run_dirs, levels):
            run_dir = _resolve(run_dir_rel)
            run_name = os.path.basename(run_dir.rstrip("/"))
            eval_path = os.path.join(run_dir, "eval_results.json")
            if not os.path.isfile(eval_path):
                print(f"[WARN] no eval_results.json in {run_dir}, skipping run")
                continue

            if level not in kb_datasets:
                kb_datasets[level] = construct_kernelbench_dataset(
                    level=level, source="local", base_path=kb_base
                )
            kb_dataset = kb_datasets[level]

            eval_results = _load_json(eval_path)
            print(f"\n[run] {run_name}  (level {level})  — {len(eval_results)} problems")

            for problem_id_str, samples in eval_results.items():
                problem_id = int(problem_id_str)
                try:
                    _, problem_name, ref_arch_src = fetch_ref_arch_from_dataset(
                        kb_dataset, problem_id
                    )
                except ValueError:
                    print(f"  [WARN] problem {problem_id} not in KernelBench level {level}")
                    continue

                baseline_time = baseline_times.get(level, {}).get(problem_id)

                for sample in samples:
                    sample_id = int(sample["sample_id"])
                    kernel_path = _staged_kernel_path(
                        run_dir, level, problem_id, sample_id
                    )
                    if not os.path.isfile(kernel_path):
                        missing_kernel += 1
                        continue
                    with open(kernel_path) as kf:
                        kernel_src = kf.read()

                    compiled = bool(sample.get("compiled", False))
                    correct = bool(sample.get("correctness", False))
                    runtime = sample.get("runtime")
                    runtime = float(runtime) if runtime is not None else None

                    rstats = sample.get("runtime_stats") or {}
                    runtime_min = float(rstats["min"]) if rstats.get("min") is not None else None
                    runtime_std = float(rstats["std"]) if rstats.get("std") is not None else None

                    # Speedup over the per-problem PyTorch baseline (KernelBench
                    # fast_p style). Only meaningful for correct kernels with a valid
                    # runtime and a known baseline; otherwise None (the listwise
                    # builder drops such candidates rather than guessing a grade).
                    speedup = None
                    if correct and runtime is not None and runtime > 0:
                        if baseline_time is not None:
                            speedup = baseline_time / runtime
                            correct_with_speedup += 1
                        else:
                            correct_without_baseline += 1

                    lr = compute_label(compiled=compiled, correct=correct)

                    # In compiled_wrong mode, keep positives and compiled-but-wrong
                    # negatives only; drop compile-failure negatives entirely.
                    if (data_cfg.negative_mode == "compiled_wrong"
                            and lr.label == 0 and not compiled):
                        dropped_compile_fail += 1
                        continue

                    reason_counts[lr.reason] += 1
                    level_counts[level] += 1
                    rows_written += 1

                    row = {
                        "run_name": run_name,
                        "level": level,
                        "problem_id": problem_id,
                        "problem_name": problem_name,
                        "sample_id": sample_id,
                        "ref_arch_src": ref_arch_src,
                        "kernel_src": kernel_src,
                        "compiled": compiled,
                        "correct": correct,
                        "runtime": runtime,
                        "runtime_min": runtime_min,
                        "runtime_std": runtime_std,
                        "speedup": speedup,
                        "label": lr.label,
                    }
                    out_f.write(json.dumps(row) + "\n")

    n_correct = reason_counts["compiled_and_correct"]
    print("\n" + "=" * 60)
    print(f"Dataset written: {out_path}")
    print(f"  negative_mode  : {data_cfg.negative_mode}")
    print(f"  rows           : {rows_written}")
    print(f"  positives      : {n_correct}  ({100 * n_correct / max(rows_written, 1):.1f}%)")
    print(f"  negatives      : {rows_written - n_correct}")
    if data_cfg.negative_mode == "compiled_wrong":
        print(f"  dropped (compile-fail negatives): {dropped_compile_fail}")
    print(f"  speedup        : {correct_with_speedup}/{n_correct} correct have a baseline "
          f"({correct_without_baseline} correct lack a baseline -> speedup None)")
    print(f"  missing kernels: {missing_kernel} (eval entry but no staged .py)")
    print(f"  per-level rows : {dict(level_counts)}")
    print("  label reasons  :")
    for reason, count in reason_counts.most_common():
        print(f"      {reason:22s} {count}")
    if rows_written == 0:
        print("[ERROR] 0 rows written.")
        sys.exit(1)
    print("=" * 60)
    return out_path


def main() -> None:
    cfg = load_config()
    build_dataset(cfg)


if __name__ == "__main__":
    main()
