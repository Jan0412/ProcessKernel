"""Evaluate every generated sample in a run and write eval_results.json. Produces arm A1.

    uv run python -m autotune.eval_run --run-dir runs/<run> --level 1 --gpus 0,1

This is the round-1 evaluation, and it is also what A3 and A4 need before their kernels can
be swept. It exists rather than calling KernelBench's own scripts/eval_from_generations.py
for two reasons:

  * that script lives in the upstream repo, not in the installed `kernelbench` package. The
    repo's wrapper expects a clone at ~/KernelBench which is not present on this machine, so
    the only copy is inside a uv cache directory that is content-addressed and disposable.
    Depending on it would make the experiment unreproducible.
  * its resume check is per *problem*: once any sample of a problem has been written it skips
    that problem's remaining samples. Under a 1-day wall limit these jobs will be requeued,
    and that granularity would silently drop the samples we had not reached yet.

The measurement itself is identical -- the same kernelbench.eval.eval_kernel_against_ref, the
same 5 correctness trials and 100 timed runs, the same cuda_event timing. We reuse the sweep's
worker process and scheduler, so a kernel evaluated here and the same kernel's identity config
evaluated in the sweep go through exactly the same code path. That is what makes the tuning
gain a like-for-like ratio instead of an artifact of two different harnesses.

Output schema matches KernelBench's exactly, so the existing notebooks keep working:

    {"<problem_id>": [{"sample_id", "compiled", "correctness", "metadata",
                       "runtime", "runtime_stats"}, ...], ...}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from autotune.sweep import KERNEL_RE, Task, find_ref, run_tasks


def build_tasks(run_dir: Path, dataset_dir: Path, level: int) -> tuple[list[Task], dict]:
    """One task per generated sample, at its own constants (config 0)."""
    tasks: list[Task] = []
    skipped: dict[str, int] = defaultdict(int)
    for kernel in sorted(run_dir.glob(f"level_{level}_problem_*_sample_*_kernel.py")):
        match = KERNEL_RE.search(kernel.name)
        if not match:
            continue
        _, pid, sid = (int(g) for g in match.groups())
        ref = find_ref(dataset_dir, pid)
        if ref is None:
            skipped["no_reference_problem"] += 1
            continue
        tasks.append(Task(kernel, ref, pid, sid, config_id=0, config={}, phase="eval"))
    return tasks, dict(skipped)


def write_eval_results(run_dir: Path, out_path: Path) -> dict:
    """Fold the JSONL into KernelBench's eval_results.json schema."""
    jsonl = run_dir / "eval" / "results.jsonl"
    by_problem: dict[int, dict[int, dict]] = defaultdict(dict)
    for line in jsonl.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("phase") != "eval":
            continue
        metadata = {}
        if rec.get("error"):
            metadata["runtime_error"] = rec["error"]
        if rec.get("error_kind"):
            metadata["error_kind"] = rec["error_kind"]
        if rec.get("correctness_trials"):
            metadata["correctness_trials"] = rec["correctness_trials"]
        if rec.get("hardware"):
            metadata["hardware"] = rec["hardware"]
        # A later record for the same sample supersedes an earlier one (requeue re-ran it).
        by_problem[rec["problem_id"]][rec["sample_id"]] = {
            "sample_id": rec["sample_id"],
            "compiled": bool(rec.get("compiled")),
            "correctness": bool(rec.get("correct")),
            "metadata": metadata,
            "runtime": float(rec["runtime_ms"]) if rec.get("runtime_ms") else -1.0,
            "runtime_stats": rec.get("runtime_stats") or {},
        }

    results = {
        str(pid): [samples[sid] for sid in sorted(samples)]
        for pid, samples in sorted(by_problem.items())
    }
    out_path.write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--dataset-dir", default=None, help="default: KernelBench/level<level>")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--num-correct-trials", type=int, default=5)
    ap.add_argument("--num-perf-trials", type=int, default=100)
    args = ap.parse_args()
    if args.dataset_dir is None:
        args.dataset_dir = f"KernelBench/level{args.level}"

    # run_task() reads the trial counts off the "final" phase fields.
    args.final_correct_trials = args.num_correct_trials
    args.final_perf_trials = args.num_perf_trials
    args.search_correct_trials = args.num_correct_trials
    args.search_perf_trials = args.num_perf_trials

    run_dir = Path(args.run_dir)
    tasks, skipped = build_tasks(run_dir, Path(args.dataset_dir), args.level)
    if skipped:
        print(f"skipped: {skipped}")
    run_tasks(tasks, args, run_dir / "eval" / "results.jsonl", "eval")

    out_path = run_dir / "eval_results.json"
    results = write_eval_results(run_dir, out_path)

    n = sum(len(v) for v in results.values())
    compiled = sum(1 for v in results.values() for s in v if s["compiled"])
    correct = sum(1 for v in results.values() for s in v if s["correctness"])
    print(f"\nwrote {out_path}")
    print(f"  {len(results):,} problems, {n:,} samples")
    print(f"  compiled  {compiled:,} ({100 * compiled / max(n, 1):.1f}%)")
    print(f"  correct   {correct:,} ({100 * correct / max(n, 1):.1f}%)   <- arm A1")


if __name__ == "__main__":
    main()
