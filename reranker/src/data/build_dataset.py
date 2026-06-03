"""Build the labeled reranker dataset from KernelBench evaluation runs.

Reads, for each configured run directory:
  - eval_results.json          {problem_id: [{sample_id, compiled, correctness, runtime, ...}]}
  - staged kernel sources      level_{L}_problem_{P}_sample_{S}_kernel.py
and joins them with:
  - the reference architecture source (from the KernelBench dataset)

Label: positive (1) iff compiled AND correct; negative (0) otherwise.

Emits one JSONL row per evaluated kernel candidate to `data.dataset_jsonl`.

Usage:
    bash scripts/build_dataset.sh [configs/default.yaml]
"""

from __future__ import annotations

import json
import os
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


def _staged_kernel_path(run_dir: str, level: int, problem_id: int, sample_id: int) -> str:
    return os.path.join(
        run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
    )


def build_dataset(cfg: RerankerConfig) -> str:
    """Build the JSONL dataset and return its path."""
    _add_kernelbench_to_path(cfg.data.kernelbench_dir)
    from kernelbench.dataset import construct_kernelbench_dataset, fetch_ref_arch_from_dataset

    data_cfg = cfg.data
    levels = data_cfg.levels_for_run_dirs()
    kb_base = os.path.join(_resolve(data_cfg.kernelbench_dir), "KernelBench")

    out_path = _resolve(data_cfg.dataset_jsonl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Cache KernelBench datasets per level (run_dirs may share a level).
    kb_datasets: dict[int, object] = {}

    reason_counts: Counter = Counter()
    level_counts: Counter = Counter()
    rows_written = 0
    missing_kernel = 0

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

                    lr = compute_label(compiled=compiled, correct=correct)
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
                        "label": lr.label,
                    }
                    out_f.write(json.dumps(row) + "\n")

    pos = reason_counts["compiled_and_correct"]
    print("\n" + "=" * 60)
    print(f"Dataset written: {out_path}")
    print(f"  rows           : {rows_written}")
    print(f"  positives      : {pos}  ({100 * pos / max(rows_written, 1):.1f}%)")
    print(f"  negatives      : {rows_written - pos}")
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
