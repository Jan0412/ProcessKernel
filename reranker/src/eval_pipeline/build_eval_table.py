"""Stage 1a — build the reranker-independent eval table.

Walks each KernelBench run directory, joins every staged kernel with its entry in
``eval_results.json`` and with the per-problem PyTorch baseline runtime, and
writes one JSONL row per kernel to ``--out``. The table carries outcomes
(compiled / correct), kernel runtimes (mean/min/std) and baseline runtimes
(mean/min) — everything needed to compute a speedup under either statistic — but
**no** reranker scores. It is built once and reused by every reranker (Stage 1b
adds the scores keyed by ``(run_name, kernel_file)``).

Example:
    python -m reranker.src.eval_pipeline.build_eval_table \\
        --runs /path/runs/gpt-oss-120b_kernelbook_level5_triton \\
        --timing /path/results/timing/A100/baseline_time_torch.json \\
        --kernelbench-dir /path/KernelBench \\
        --out reranker/data/eval/eval_table.jsonl
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from collections import Counter
from pathlib import Path

from reranker.src.eval_pipeline.common import (
    EVAL_TABLE_FIELDS,
    RefArchIndex,
    _load_baseline_times,
    git_sha,
    iter_run_kernels,
    load_eval_results,
    run_name_of,
)


def _level_of_run(run_dir: str, override: int | None) -> int | None:
    """A run dir mixes exactly one level; take it from the first kernel filename."""
    if override is not None:
        return override
    for _path, level, _pid, _sid in iter_run_kernels(run_dir):
        return level
    return None


def build_eval_table(
    run_dirs: list[str],
    timing_json: str,
    kernelbench_dir: str,
    out_path: str,
    levels: list[int] | None = None,
) -> str:
    baseline_times = _load_baseline_times(timing_json)
    ref_index = RefArchIndex(kernelbench_dir)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    rows_written = 0
    missing_eval = 0
    missing_baseline = 0
    per_run: dict[str, dict] = {}
    reason = Counter()

    with open(out_path, "w") as out_f:
        for i, run_dir in enumerate(run_dirs):
            run_dir = os.path.abspath(run_dir)
            run_name = run_name_of(run_dir)
            eval_results = load_eval_results(run_dir)
            if eval_results is None:
                continue
            level_override = levels[i] if levels else None

            n_run = n_compiled = n_correct = 0
            for path, level, pid, sid in iter_run_kernels(run_dir):
                if level_override is not None:
                    level = level_override
                entry = next(
                    (e for e in eval_results.get(str(pid), []) if e.get("sample_id") == sid),
                    None,
                )
                if entry is None:
                    missing_eval += 1
                    continue

                compiled = bool(entry.get("compiled", False))
                correct = bool(entry.get("correctness", False))
                runtime = entry.get("runtime")
                runtime_mean = float(runtime) if runtime is not None else None
                rstats = entry.get("runtime_stats") or {}
                runtime_min = float(rstats["min"]) if rstats.get("min") is not None else None
                runtime_std = float(rstats["std"]) if rstats.get("std") is not None else None

                bl = baseline_times.get(level, {}).get(pid)
                baseline_mean = bl.get("mean") if bl else None
                baseline_min = bl.get("min") if bl else None
                if bl is None:
                    missing_baseline += 1

                row = {
                    "run_name": run_name,
                    "level": level,
                    "problem_id": pid,
                    "problem_name": ref_index.problem_name(level, pid),
                    "sample_id": sid,
                    "kernel_file": path.name,
                    "compiled": compiled,
                    "correct": correct,
                    "runtime_mean": runtime_mean,
                    "runtime_min": runtime_min,
                    "runtime_std": runtime_std,
                    "baseline_mean": baseline_mean,
                    "baseline_min": baseline_min,
                }
                out_f.write(json.dumps(row) + "\n")
                rows_written += 1
                n_run += 1
                n_compiled += int(compiled)
                n_correct += int(compiled and correct)
                reason["compiled" if compiled else "not_compiled"] += 1

            per_run[run_name] = {
                "level": _level_of_run(run_dir, level_override),
                "kernels": n_run,
                "compiled": n_compiled,
                "correct": n_correct,
            }
            print(f"[run] {run_name}: {n_run} kernels, {n_compiled} compiled, {n_correct} correct")

    meta = {
        "out": os.path.abspath(out_path),
        "runs": [os.path.abspath(r) for r in run_dirs],
        "timing_json": os.path.abspath(timing_json),
        "kernelbench_dir": os.path.abspath(kernelbench_dir),
        "levels_override": levels,
        "rows": rows_written,
        "missing_eval_entry": missing_eval,
        "missing_baseline": missing_baseline,
        "per_run": per_run,
        "git_sha": git_sha(kernelbench_dir),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "fields": EVAL_TABLE_FIELDS,
    }
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print(f"eval table written : {out_path}  ({rows_written} rows)")
    print(f"  sidecar meta     : {meta_path}")
    print(f"  missing eval     : {missing_eval} (kernel file w/o eval entry)")
    print(f"  missing baseline : {missing_baseline} (no baseline timing -> speedup uncomputable)")
    print("=" * 60)
    if rows_written == 0:
        raise SystemExit("[ERROR] 0 rows written — check --runs / naming.")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the reranker-independent eval table (Stage 1a)")
    ap.add_argument("--runs", nargs="+", required=True, help="One or more KernelBench run dirs")
    ap.add_argument("--timing", required=True, help="Baseline timing JSON (baseline_time_torch.json)")
    ap.add_argument("--kernelbench-dir", required=True,
                    help="KernelBench checkout whose KernelBench/level{L}/ holds reference archs")
    ap.add_argument("--out", required=True, help="Output eval_table.jsonl path")
    ap.add_argument("--levels", nargs="+", type=int, default=None,
                    help="Optional per-run level override (one int per --runs); "
                         "default: inferred from kernel filenames")
    args = ap.parse_args()
    if args.levels is not None and len(args.levels) != len(args.runs):
        ap.error(f"--levels has {len(args.levels)} entries but there are {len(args.runs)} --runs")
    build_eval_table(args.runs, args.timing, args.kernelbench_dir, args.out, args.levels)


if __name__ == "__main__":
    main()
