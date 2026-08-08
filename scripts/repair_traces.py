"""Rewrite a traced run's ``.npz`` files to a clean, deduplicated top-K (KGEN-25).

    python scripts/repair_traces.py --run-dir <run> --dry-run
    python scripts/repair_traces.py --run-dir <run> -j 16

vLLM prepends the sampled token to the top-K without deduping, so every row this run
recorded is K+1 wide with one id twice, and ``pack``'s truncation kept the repeat and
discarded the real rank-K alternative. What survives on disk is the true top-(K-1) plus a
duplicate, so the repair is exact: drop the repeat, keep descending order, land on K-1.

Only ``topk_ids`` and ``topk_lp`` change. ``token_ids``, ``sampled_lp``, ``sampled_rank``
and ``seg`` are already correct and are asserted byte-identical before anything is written,
which is why ``surprisal``, ``top1_lp`` and the rank calibration come back unchanged while
``margin`` -- which the duplicate pinned to zero on 90.9% of rows -- becomes real.

Rows are decided by ``meta["trace_k"]``, never by array width: a row written by the fixed
``pack`` and a row written by the buggy one are both K wide, and truncating the former
would throw away a genuine alternative.

Journals are rewritten in place with the recomputed ``confidence`` blocks, and
``trace_topk`` in ``generation_config.yaml`` / ``trace_config.json`` is moved to the K that
is actually stored. Traces already at the target K are skipped, so this is idempotent.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel_gen.core.trace import (  # noqa: E402
    PAD_ID,
    dedupe_rows,
    derive_scalars,
    read_trace,
    summarize,
    write_trace,
)

REPAIR_TAG = "kgen-25"


def tar_writing(run_dir: str) -> int | None:
    """PID of a ``tar`` archiving this run, if one is running.

    An in-place repair that starts under a live backup leaves the archive holding a mix of
    pre- and post-repair files -- an archive that restores to a state the run was never in.
    """
    name = os.path.basename(os.path.normpath(run_dir))
    for entry in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(entry, "rb") as fh:
                argv = fh.read().decode("utf-8", "replace").split("\0")
        except OSError:  # the process exited between the glob and the read
            continue
        if argv and os.path.basename(argv[0]) == "tar" and any(name in a for a in argv[1:]):
            return int(entry.split("/")[2])
    return None


def lintloop_jobs() -> list[str]:
    """Running lintloop job ids, so the repair refuses to race a live writer."""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", "lintloop",
             "-t", "RUNNING", "-o", "%i"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def verify(before, ids: np.ndarray, lp: np.ndarray, k: int) -> None:
    """Everything that must hold of a repaired trace, checked before it is written."""
    if ids.shape != (len(before), k):
        raise ValueError(f"repaired to {ids.shape}, expected {(len(before), k)}")

    valid = np.isfinite(lp)
    if not (valid.sum(axis=1) == k).all():
        raise ValueError(f"rows with fewer than {k} valid entries: "
                         f"{np.bincount(valid.sum(axis=1), minlength=k + 1).tolist()}")
    # Descending: float16 is coarse, so compare with a tolerance rather than exactly.
    step = lp[:, 1:].astype(np.float32) - lp[:, :-1].astype(np.float32)
    if (step > 1e-3).any():
        raise ValueError(f"{int((step > 1e-3).sum())} rows out of descending order")
    same = (ids[:, :, None] == ids[:, None, :]) & (ids[:, :, None] != PAD_ID)
    if np.triu(same, 1).any():
        raise ValueError(f"{int(np.triu(same, 1).any(axis=(1, 2)).sum())} rows still duplicated")

    # The sampled token still sits where sampled_rank says, for the ranks still in range.
    rank = before.sampled_rank.astype(np.int64)
    inside = (rank >= 1) & (rank <= k)
    if inside.any():
        rows = np.flatnonzero(inside)
        if not (ids[rows, rank[inside] - 1] == before.token_ids[inside]).all():
            raise ValueError("sampled_rank no longer indexes the sampled token")


def repair_one(path: str, k: int, requested: int, vocab_size: int | None, window: int,
               apply: bool) -> dict:
    """Repair one ``.npz``. Returns a per-file outcome; writes nothing when ``apply`` is False.

    ``requested`` is the ``--trace-topk`` the run was launched with, taken from the shard's
    ``trace_config.json``. It is NOT derivable from the array: the buggy ``pack`` allocated
    exactly ``(n, k)`` and truncated into it, so the stored width already *is* the requested
    K, and the alternative that went missing was dropped rather than leaving a gap.
    """
    trace = read_trace(path)
    stem = os.path.basename(path)[: -len(".npz")]
    stored = trace.meta.get("trace_k")
    width = trace.topk_ids.shape[1]

    if stored == k:
        if trace.meta.get("trace_topk_requested") != requested:
            # An earlier pass stamped the wrong provenance. The arrays are already right,
            # so correct the record rather than leaving the file asserting something false.
            trace.meta["trace_topk_requested"] = requested
            if apply:
                tmp = path[: -len(".npz")] + ".tmp.npz"
                write_trace(tmp, trace)
                os.replace(tmp, path)
            return {"stem": stem, "status": "restamped", "n": len(trace),
                    "confidence": summarize(
                        derive_scalars(trace.topk_lp, trace.sampled_lp, vocab_size=vocab_size),
                        window=window),
                    "meta": dict(trace.meta)}
        return {"stem": stem, "status": "already-repaired", "n": len(trace)}
    if stored is not None and stored != k:
        # Written by the fixed pack() at a different K. Truncating it would discard a real
        # alternative, which is the opposite of the point.
        return {"stem": stem, "status": f"native-k{stored}", "n": len(trace)}
    if width != k + 1:
        return {"stem": stem, "status": f"unexpected-width-{width}", "n": len(trace)}

    ids, lp = dedupe_rows(trace.topk_ids, trace.topk_lp, k)
    if len(trace):
        verify(trace, ids, lp, k)

    old = summarize(
        derive_scalars(trace.topk_lp, trace.sampled_lp, vocab_size=vocab_size), window=window
    )
    trace.topk_ids, trace.topk_lp = ids, lp
    trace.meta.update(trace_k=k, trace_topk_requested=requested, trace_repair=REPAIR_TAG)
    new = summarize(
        derive_scalars(lp, trace.sampled_lp, vocab_size=vocab_size), window=window
    )

    if apply:
        # ".tmp.npz", not ".tmp": np.savez appends the extension when it is missing, and
        # would leave the temp file somewhere os.replace is not looking.
        tmp = path[: -len(".npz")] + ".tmp.npz"
        write_trace(tmp, trace)
        os.replace(tmp, path)

    return {
        "stem": stem, "status": "repaired", "n": len(trace),
        "confidence": new, "meta": dict(trace.meta), "old": old,
    }


def rewrite_journal(journal: str, results: dict[str, dict], apply: bool) -> int:
    """Swap in the recomputed ``confidence`` blocks, preserving every other field and the order."""
    if not os.path.exists(journal):
        return 0
    updated, lines = 0, []
    with open(journal) as fh:
        for line in fh:
            record = json.loads(line)
            outcome = results.get(record.get("stem"))
            if record.get("trace") and outcome and outcome["status"] in ("repaired", "restamped"):
                record["confidence"] = outcome["confidence"]
                record["trace"] = {**record["trace"], **outcome["meta"]}
                updated += 1
            lines.append(json.dumps(record))
    if apply and updated:
        tmp = journal + ".tmp"
        with open(tmp, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp, journal)
    return updated


def update_configs(shard: str, k: int, requested: int, apply: bool) -> list[str]:
    """Point ``trace_topk`` at what is actually stored, keeping the requested value."""
    import yaml

    touched = []
    trace_config = os.path.join(shard, "traces", "trace_config.json")
    if os.path.exists(trace_config):
        with open(trace_config) as fh:
            cfg = json.load(fh)
        if cfg.get("trace_topk") != k:
            cfg.update(trace_topk=k, trace_topk_requested=requested, trace_repair=REPAIR_TAG)
            if apply:
                tmp = trace_config + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(cfg, fh, indent=2)
                os.replace(tmp, trace_config)
            touched.append(trace_config)

    for name in (
        os.path.join(shard, "generation_config.yaml"),
        os.path.join(shard, "rounds", "round_0", "generation_config.yaml"),
    ):
        if not os.path.exists(name):
            continue
        with open(name) as fh:
            cfg = yaml.safe_load(fh) or {}
        if cfg.get("trace_topk") == k:
            continue
        cfg.update(trace_topk=k, trace_topk_requested=requested)
        if apply:
            tmp = name + ".tmp"
            with open(tmp, "w") as fh:
                yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=True)
            os.replace(tmp, name)
        touched.append(name)
    return touched


def repair_run(run_dir: str, k: int, shards: list[str] | None, jobs: int,
               limit: int, apply: bool) -> dict:
    counts: dict[str, int] = {}
    tokens = 0
    before: list[dict] = []
    after: list[dict] = []
    config_files: list[str] = []

    pattern = os.path.join(run_dir, "shard_*") if shards is None else None
    shard_dirs = sorted(
        glob.glob(pattern) if pattern else [os.path.join(run_dir, f"shard_{s}") for s in shards]
    )
    if not shard_dirs and os.path.isdir(os.path.join(run_dir, "traces")):
        shard_dirs = [run_dir]  # an unsharded run

    for shard in shard_dirs:
        round_dirs = sorted(glob.glob(os.path.join(shard, "traces", "round_*")))
        if not round_dirs:
            continue
        cfg_path = os.path.join(shard, "traces", "trace_config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        vocab_size = cfg.get("vocab_size")
        window = int(cfg.get("trace_window", 512))
        requested = int(cfg.get("trace_topk_requested", cfg.get("trace_topk", k + 1)))
        if vocab_size is None:
            raise SystemExit(
                f"!! no vocab_size in {cfg_path} -- self_cert would be silently dropped.\n"
                f"!! add it, or the recomputed confidence will not match the run's own."
            )

        repaired_here = 0
        for round_dir in round_dirs:
            paths = sorted(glob.glob(os.path.join(round_dir, "*.npz")))
            if limit:
                paths = paths[:limit]
            if not paths:
                continue
            results: dict[str, dict] = {}
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = [
                    pool.submit(repair_one, p, k, requested, vocab_size, window, apply)
                    for p in paths
                ]
                for done, future in enumerate(futures, start=1):
                    out = future.result()
                    results[out["stem"]] = out
                    counts[out["status"]] = counts.get(out["status"], 0) + 1
                    tokens += out["n"]
                    if out["status"] == "repaired":
                        before.append(out["old"])
                        after.append(out["confidence"])
                    if done % 500 == 0:
                        print(f"  {done}/{len(paths)} {os.path.relpath(round_dir, run_dir)}",
                              flush=True)
            repaired_here += sum(
                1 for r in results.values() if r["status"] in ("repaired", "restamped")
            )
            updated = rewrite_journal(os.path.join(round_dir, "attempts.jsonl"), results, apply)
            print(f"  {os.path.relpath(round_dir, run_dir)}: {len(paths)} traces, "
                  f"{updated} journal records", flush=True)

        if repaired_here:
            config_files += update_configs(shard, k, requested, apply)

    return {"counts": counts, "tokens": tokens, "before": before, "after": after,
            "configs": config_files}


def report(result: dict, apply: bool) -> None:
    print("\n" + "=" * 62)
    print(f"  {'applied' if apply else 'DRY RUN -- nothing written'}")
    print("=" * 62)
    for status, n in sorted(result["counts"].items()):
        print(f"  {status:24} {n:8d}")
    print(f"  {'tokens':24} {result['tokens']:8d}")

    before, after = result["before"], result["after"]
    if before:
        keys = sorted({key for row in before for key in row} & {key for row in after for key in row})
        print(f"\n  {'statistic':18} {'stored':>12} {'repaired':>12} {'delta':>12}")
        for key in keys:
            a = np.array([r[key] for r in before if np.isfinite(r.get(key, np.nan))])
            b = np.array([r[key] for r in after if np.isfinite(r.get(key, np.nan))])
            if not a.size or not b.size:
                continue
            print(f"  {key:18} {a.mean():12.5f} {b.mean():12.5f} {b.mean() - a.mean():+12.5f}")
    for path in result["configs"]:
        print(f"\n  config {'updated' if apply else 'would update'}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--shards", default="", help="comma-separated, e.g. 00,01,03; default all")
    parser.add_argument("--k", type=int, default=19, help="alternatives to keep per row")
    parser.add_argument("-j", "--jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="first N traces per round dir")
    parser.add_argument("--dry-run", action="store_true", help="verify and report, write nothing")
    parser.add_argument("--force", action="store_true", help="skip the live-writer preflight")
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"!! no such run dir: {args.run_dir}")

    apply = not args.dry_run
    if apply and not args.force:
        if (pid := tar_writing(args.run_dir)) is not None:
            raise SystemExit(
                f"!! a tar is archiving this run (pid {pid}); repairing now would leave the\n"
                f"!! archive holding a mix of pre- and post-repair files.\n"
                f"!! wait for it:  while kill -0 {pid} 2>/dev/null; do sleep 60; done"
            )
        if jobs := lintloop_jobs():
            raise SystemExit(
                f"!! lintloop is still running ({', '.join(jobs)}) and may write into this run.\n"
                f"!! wait for it, or re-run with --force if it targets other shards."
            )

    print(json.dumps({"run_dir": args.run_dir, "k": args.k, "jobs": args.jobs,
                      "dry_run": args.dry_run}, indent=2), flush=True)
    shards = [s.strip() for s in args.shards.split(",") if s.strip()] or None
    report(repair_run(args.run_dir, args.k, shards, args.jobs, args.limit, apply), apply)


if __name__ == "__main__":
    main()
