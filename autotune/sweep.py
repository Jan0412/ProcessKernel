"""Sweep launch configs over generated kernels. Produces arm A2 and the A4 feedback payload.

    uv run python -m autotune.sweep --run-dir runs/<run> --level 1 --top-k 2 --gpus 0,1
    uv run python -m autotune.sweep --run-dir runs/<run> --level 1 --finalize

Two phases, because a full-fidelity timing run over the whole grid would cost 5x what it
needs to. The search phase ranks configs cheaply (3 correctness trials, 20 timed runs); the
finalize phase re-measures only the winner -- and the kernel's own identity config -- at the
same 5/100 fidelity KernelBench uses, so the reported tuning gain is a ratio of two
full-fidelity numbers taken by the same harness.

That last point is the one that matters. If we compared a sweep-measured best config against
the runtime already sitting in eval_results.json, any systematic difference between the two
measurement setups -- different process, different warmup, different cache state -- would show
up as tuning gain. Re-measuring the identity config here makes the ratio honest.

Output is an append-only JSONL: the 1-day wall limit on the cluster means these jobs will be
requeued, and resuming has to be free. We do not reuse eval_from_generations.py's
eval_results.json for this: its resume check is per-problem, so once any sample of a problem
is written it skips the rest -- the wrong granularity for per-config work.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

from autotune.grids import at_grid_edge, build_grid
from autotune.knobs import analyze
from autotune.patcher import Unpatchable, patch_source

KERNEL_RE = re.compile(r"level_(\d+)_problem_(\d+)_sample_(\d+)_kernel\.py$")


@dataclass(frozen=True)
class Task:
    kernel: Path
    ref: Path
    problem_id: int
    sample_id: int
    config_id: int
    config: dict
    phase: str  # "search" | "final"

    @property
    def key(self) -> str:
        return f"{self.kernel.stem}|{self.phase}|{self.config_id}"


def find_ref(dataset_dir: Path, problem_id: int) -> Path | None:
    """KernelBench problem files are named <id>_<Name>.py; the id is the filename prefix."""
    for path in dataset_dir.glob(f"{problem_id}_*.py"):
        return path
    return None


def select_kernels(run_dir: Path, top_k: int) -> dict[int, list[tuple[int, float]]]:
    """The top-k *correct* samples per problem, by as-generated runtime.

    Sweeping every sample would be ~5x the cost for a baseline that is already best-of-k;
    sweeping only the fastest one would make A2 weaker than it should be, and A2 is the
    baseline the whole experiment has to beat honestly.
    """
    results = json.loads((run_dir / "eval_results.json").read_text())
    out: dict[int, list[tuple[int, float]]] = {}
    for pid, samples in results.items():
        correct = [
            (s["sample_id"], float(s["runtime"]))
            for s in samples
            if s.get("correctness") and s.get("runtime", -1) > 0
        ]
        if correct:
            correct.sort(key=lambda t: t[1])
            out[int(pid)] = correct[:top_k]
    return out


def build_tasks(args, phase: str, summary: dict | None = None) -> tuple[list[Task], dict]:
    run_dir, dataset_dir = Path(args.run_dir), Path(args.dataset_dir)
    selected = select_kernels(run_dir, args.top_k)
    tasks: list[Task] = []
    skipped: dict[str, int] = defaultdict(int)
    tunability_dir = run_dir / "sweep" / "tunability"
    tunability_dir.mkdir(parents=True, exist_ok=True)

    for pid, samples in sorted(selected.items()):
        ref = find_ref(dataset_dir, pid)
        if ref is None:
            skipped["no_reference_problem"] += 1
            continue
        for sid, _ in samples:
            kernel = run_dir / f"level_{args.level}_problem_{pid}_sample_{sid}_kernel.py"
            if not kernel.exists():
                skipped["kernel_file_missing"] += 1
                continue
            rep = analyze(kernel.read_text())
            (tunability_dir / f"{kernel.stem}.json").write_text(json.dumps({
                "knobs": [{"name": k.name, "kind": k.kind, "current": k.current} for k in rep.knobs],
                "excluded": rep.excluded, "ndim_class": rep.ndim_class,
                "has_loop": rep.has_loop, "n_jit_kernels": rep.n_jit_kernels,
                "parse_error": rep.parse_error,
            }, indent=2))
            if rep.parse_error:
                skipped["parse_error"] += 1
                continue

            if phase == "search":
                configs = list(enumerate(build_grid(rep)))
            else:
                entry = (summary or {}).get(kernel.stem)
                if not entry or entry.get("best_config") is None:
                    skipped["no_search_winner"] += 1
                    continue
                # Re-measure the winner and the identity config at full fidelity: these two
                # numbers are the numerator and denominator of the reported tuning gain.
                configs = [(0, {}), (entry["best_config_id"], entry["best_config"])]
                if entry["best_config_id"] == 0:
                    configs = [(0, {})]

            for cid, cfg in configs:
                tasks.append(Task(kernel, ref, pid, sid, cid, cfg, phase))

    return tasks, dict(skipped)


def run_task(task: Task, gpu: str, args) -> dict:
    """Patch, then evaluate in a throwaway process pinned to one GPU."""
    src = task.kernel.read_text()
    try:
        patched = patch_source(src, task.config)
    except Unpatchable as e:
        return {"compiled": False, "correct": False, "runtime_ms": None,
                "error": str(e), "error_kind": "unpatchable"}

    correct_trials = args.final_correct_trials if task.phase == "final" else args.search_correct_trials
    perf_trials = args.final_perf_trials if task.phase == "final" else args.search_perf_trials

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(patched)
        patched_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "autotune.sweep_worker",
             "--ref-file", str(task.ref), "--kernel-file", patched_path,
             "--num-correct-trials", str(correct_trials),
             "--num-perf-trials", str(perf_trials)],
            capture_output=True, text=True, timeout=args.timeout,
            env={**_env(), "CUDA_VISIBLE_DEVICES": gpu},
        )
    except subprocess.TimeoutExpired:
        # The process is SIGKILLed by subprocess. A hung Triton compile or a spinning kernel
        # is only recoverable at this level -- this is why each eval gets its own process.
        return {"compiled": False, "correct": False, "runtime_ms": None,
                "error": f"timeout after {args.timeout}s", "error_kind": "timeout"}
    finally:
        Path(patched_path).unlink(missing_ok=True)

    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"compiled": False, "correct": False, "runtime_ms": None,
                "error": f"worker produced no JSON (rc={proc.returncode}): {proc.stderr[-500:]}",
                "error_kind": "worker_crash"}


def _env() -> dict:
    import os
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def sweep(args, phase: str, summary: dict | None = None) -> None:
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "results.jsonl"

    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                rec = json.loads(line)
                done.add(f"{rec['kernel']}|{rec['phase']}|{rec['config_id']}")
            except (json.JSONDecodeError, KeyError):
                continue  # a torn last line from a killed job; ignore it

    tasks, skipped = build_tasks(args, phase, summary)
    todo = [t for t in tasks if t.key not in done]
    print(f"[{phase}] {len(tasks):,} tasks, {len(tasks) - len(todo):,} already done, "
          f"{len(todo):,} to run" + (f"   skipped: {skipped}" if skipped else ""))
    if not todo:
        return

    # All configs of one kernel go to the same GPU, so their timings are comparable.
    gpus = args.gpus.split(",")
    queues: dict[str, Queue] = {g: Queue() for g in gpus}
    by_kernel: dict[str, list[Task]] = defaultdict(list)
    for t in todo:
        by_kernel[t.kernel.stem].append(t)
    for i, (_, group) in enumerate(sorted(by_kernel.items())):
        for t in group:
            queues[gpus[i % len(gpus)]].put(t)

    lock = threading.Lock()
    counter = {"n": 0}
    total = len(todo)

    def worker(gpu: str) -> None:
        while not queues[gpu].empty():
            task = queues[gpu].get()
            res = run_task(task, gpu, args)
            rec = {
                "kernel": task.kernel.stem, "problem_id": task.problem_id,
                "sample_id": task.sample_id, "config_id": task.config_id,
                "config": task.config, "phase": task.phase, **res,
            }
            rec.pop("traceback", None)
            with lock:
                with jsonl.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                counter["n"] += 1
                if counter["n"] % 25 == 0 or counter["n"] == total:
                    print(f"  {counter['n']:,}/{total:,}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def summarize(run_dir: Path, phase: str) -> dict:
    """Fold the JSONL into one record per kernel."""
    jsonl = run_dir / "sweep" / "results.jsonl"
    rows: dict[str, list[dict]] = defaultdict(list)
    for line in jsonl.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec["phase"] == phase:
            rows[rec["kernel"]].append(rec)

    summary = {}
    for kernel, recs in rows.items():
        recs = {r["config_id"]: r for r in recs}.values()  # last write wins on requeue
        correct = [r for r in recs if r.get("correct") and r.get("runtime_ms")]
        identity = next((r for r in recs if r["config_id"] == 0), None)
        identity_ms = identity["runtime_ms"] if identity and identity.get("correct") else None
        best = min(correct, key=lambda r: r["runtime_ms"]) if correct else None

        rep = analyze((run_dir / f"{kernel}.py").read_text())
        entry = {
            "problem_id": next(iter(recs))["problem_id"],
            "sample_id": next(iter(recs))["sample_id"],
            "ndim_class": rep.ndim_class,
            "knobs": [k.name for k in rep.knobs],
            # The model's own constants. Shown to it as config 0 in the A4 table, so the
            # table is a comparison rather than a list of numbers with a hole in it.
            "identity_config": rep.identity_config(),
            "identity_ms": identity_ms,
            "best_ms": best["runtime_ms"] if best else None,
            "best_config": best["config"] if best else None,
            "best_config_id": best["config_id"] if best else None,
            "tuning_gain": (identity_ms / best["runtime_ms"]) if (best and identity_ms) else None,
            "at_grid_edge": at_grid_edge(best["config"], rep) if best else None,
            "n_configs": len(recs),
            "n_correct": len(correct),
            "n_wrong_result": sum(1 for r in recs if r.get("error_kind") == "wrong_result"),
            "n_compile_error": sum(1 for r in recs if r.get("error_kind") == "compile_error"),
            "n_timeout": sum(1 for r in recs if r.get("error_kind") == "timeout"),
            "table": sorted(
                [{"config": r["config"], "runtime_ms": r.get("runtime_ms"),
                  "config_id": r["config_id"],
                  "status": "ok" if r.get("correct") else (r.get("error_kind") or "failed")}
                 for r in recs],
                key=lambda t: (t["status"] != "ok", t["runtime_ms"] or float("inf")),
            ),
        }
        summary[kernel] = entry
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--dataset-dir", default=None, help="default: KernelBench/level<level>")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--search-correct-trials", type=int, default=3)
    ap.add_argument("--search-perf-trials", type=int, default=20)
    ap.add_argument("--final-correct-trials", type=int, default=5)
    ap.add_argument("--final-perf-trials", type=int, default=100)
    ap.add_argument("--finalize", action="store_true",
                    help="re-time each kernel's winner and identity config at full fidelity")
    ap.add_argument("--limit-problems", type=int, default=None, help="smoke tests")
    args = ap.parse_args()
    if args.dataset_dir is None:
        args.dataset_dir = f"KernelBench/level{args.level}"

    run_dir = Path(args.run_dir)
    if args.limit_problems:
        _patch_selection_limit(args.limit_problems)

    if not args.finalize:
        sweep(args, "search")
        search = summarize(run_dir, "search")
        (run_dir / "sweep" / "search_summary.json").write_text(json.dumps(search, indent=2))
        print(f"\nwrote {run_dir}/sweep/search_summary.json ({len(search)} kernels)")
        print("now run again with --finalize")
        return

    search = json.loads((run_dir / "sweep" / "search_summary.json").read_text())
    sweep(args, "final", search)
    final = summarize(run_dir, "final")
    # Carry the search table across: it is the A4 feedback payload, and finalize only
    # re-measures two of its rows.
    for kernel, entry in final.items():
        if kernel in search:
            entry["table"] = search[kernel]["table"]
            entry["n_configs"] = search[kernel]["n_configs"]
            entry["n_correct"] = search[kernel]["n_correct"]
            entry["n_wrong_result"] = search[kernel]["n_wrong_result"]
            entry["n_compile_error"] = search[kernel]["n_compile_error"]
            entry["n_timeout"] = search[kernel]["n_timeout"]
    (run_dir / "sweep" / "sweep_summary.json").write_text(json.dumps(final, indent=2))
    print(f"\nwrote {run_dir}/sweep/sweep_summary.json ({len(final)} kernels)")


def _patch_selection_limit(n: int) -> None:
    """Smoke-test hook: only sweep the first n problems."""
    original = globals()["select_kernels"]

    def limited(run_dir: Path, top_k: int):
        sel = original(run_dir, top_k)
        return {k: sel[k] for k in sorted(sel)[:n]}

    globals()["select_kernels"] = limited


if __name__ == "__main__":
    main()
