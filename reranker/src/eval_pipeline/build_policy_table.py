"""Stage 1a-bis — build the reranker-independent POLICY table.

Every *non-ORM* selection signal that already exists on disk, in one JSONL keyed the
same way as the eval table (``run_name``, ``kernel_file``): the generator's own
DeepConf confidence from the trace sidecar, and the lint-loop verdict. The notebook
joins it to ask whether a trained reranker beats "just pick the sample the generator
was most confident about" — a policy that costs nothing at inference time.

Nothing is recomputed here. ``kernel_gen`` already wrote a ``confidence`` dict per
generated kernel into ``<run>/shard_NN/traces/round_R/attempts.jsonl``; this walks
those files and flattens them. Run names come from ``expand_run_dirs``, the same
function the training build uses, so the rows line up with the eval table.

Two modes, mirroring ``build_dataset``:
    --rounds 0 1 2   one row per (sample, round), read from that round's attempts
    (omitted)        one row per sample, the kernel it finished on -- the round comes
                     from ``shard_NN/lint_loop.jsonl::final_round``

Example:
    python -m reranker.src.eval_pipeline.build_policy_table \\
        --runs /path/runs/DeepSeek-V4-Flash_level1_lintloop_triton_v5_traced \\
               /path/runs/gpt-oss-120b_level1_lintloop_triton_v5_traced \\
        --levels 1 1 --rounds 0 1 2 \\
        --out reranker/data/eval/policy_table_l12.jsonl
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from collections import Counter
from pathlib import Path

from reranker.src.config import _resolve
from reranker.src.data.build_dataset import expand_run_dirs
from reranker.src.eval_pipeline.common import git_sha

# Kept bare (not prefixed) so the notebook can name a policy by its own column.
CONF_FIELDS = [
    "c_least", "c_bottom10", "c_tail",
    "mean_deepconf_c", "mean_entropy", "mean_top1_lp", "mean_margin",
    "mean_surprisal", "mean_self_cert", "mean_tail_mass", "n_tokens",
]
POLICY_TABLE_FIELDS = [
    "run_name", "generator", "level", "problem_id", "sample_id", "kernel_file", "round",
    "clean", "n_findings", *CONF_FIELDS,
]


def _trace_dir(leaf_dir: str, rounds: list[int] | None) -> str | None:
    """Map a kernel dir from expand_run_dirs to its trace sidecar dir."""
    if rounds is None:
        return None  # shard root: the round differs per sample, resolved via lint_loop
    parent, rnd = os.path.split(leaf_dir.rstrip("/"))
    shard = os.path.dirname(parent)  # .../shard_NN/rounds/round_R -> .../shard_NN
    return os.path.join(shard, "traces", rnd)


def _read_attempts(path: str) -> dict[tuple, dict]:
    """``{(level, problem_id, sample_id): record}`` for one attempts.jsonl."""
    out: dict[tuple, dict] = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[(int(r["level"]), int(r["problem_id"]), int(r["sample_id"]))] = r
    return out


def _final_rounds(shard_dir: str) -> dict[tuple, int]:
    """``{(level, problem_id, sample_id): final_round}`` from lint_loop.jsonl."""
    path = os.path.join(shard_dir, "lint_loop.jsonl")
    out: dict[tuple, int] = {}
    if not os.path.isfile(path):
        print(f"[WARN] no lint_loop.jsonl in {shard_dir}; final-kernel mode needs it")
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[(int(r["level"]), int(r["problem_id"]), int(r["sample_id"]))] = int(
                r.get("final_round", 0)
            )
    return out


def _row(rec: dict, run_name: str, generator: str, level: int, rnd: int) -> dict:
    conf = rec.get("confidence") or {}
    findings = rec.get("findings") or []
    row = {
        "run_name": run_name,
        "generator": generator,
        "level": level,
        "problem_id": int(rec["problem_id"]),
        "sample_id": int(rec["sample_id"]),
        "kernel_file": f"{rec['stem']}.py",
        "round": rnd,
        "clean": bool(rec.get("clean", False)),
        "n_findings": len(findings),
    }
    row.update({k: conf.get(k) for k in CONF_FIELDS})
    return row


def build_policy_table(
    run_dirs: list[str],
    out_path: str,
    levels: list[int] | None = None,
    rounds: list[int] | None = None,
) -> str:
    lvls = levels or [None] * len(run_dirs)
    leaves = expand_run_dirs(run_dirs, lvls, rounds)
    gen_of = {os.path.abspath(_resolve(r)): os.path.basename(os.path.abspath(_resolve(r)))
              for r in run_dirs}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    rows_written = 0
    missing_conf = 0
    no_attempts: list[str] = []
    per_run: dict[str, int] = {}
    clean_count = Counter()

    with open(out_path, "w") as out_f:
        for leaf_dir, run_name, level in leaves:
            leaf_dir = os.path.abspath(leaf_dir)
            generator = next(
                (g for root, g in gen_of.items()
                 if leaf_dir == root or leaf_dir.startswith(root + os.sep)),
                os.path.basename(leaf_dir),
            )
            if rounds is not None:
                tdir = _trace_dir(leaf_dir, rounds)
                rnd = int(os.path.basename(leaf_dir.rstrip("/")).split("_")[-1])
                recs = _read_attempts(os.path.join(tdir, "attempts.jsonl"))
                if not recs:
                    no_attempts.append(str(tdir))
                    continue
                items = [(rec, rnd) for rec in recs.values()]
            else:
                # shard root: each sample's kernel came from its own final round.
                shard_dir = leaf_dir
                finals = _final_rounds(shard_dir)
                by_round: dict[int, dict[tuple, dict]] = {}
                items = []
                for key, fr in finals.items():
                    if fr not in by_round:
                        by_round[fr] = _read_attempts(
                            os.path.join(shard_dir, "traces", f"round_{fr}", "attempts.jsonl")
                        )
                    rec = by_round[fr].get(key)
                    if rec is None:
                        continue
                    items.append((rec, fr))
                if not items:
                    no_attempts.append(str(shard_dir))
                    continue

            n_run = 0
            for rec, rnd in items:
                if not (rec.get("confidence") or {}):
                    missing_conf += 1
                row = _row(rec, run_name, generator,
                           level if level is not None else int(rec["level"]), rnd)
                out_f.write(json.dumps(row) + "\n")
                rows_written += 1
                n_run += 1
                clean_count[bool(row["clean"])] += 1
            per_run[run_name] = n_run
            print(f"[leaf] {run_name}: {n_run} rows")

    n_clean = clean_count[True]
    meta = {
        "out": os.path.abspath(out_path),
        "runs": [os.path.abspath(r) for r in run_dirs],
        "levels": levels,
        "rounds": rounds,
        "rows": rows_written,
        "missing_confidence": missing_conf,
        "leaves_without_attempts": no_attempts,
        "per_run": per_run,
        "lint_clean": n_clean,
        "lint_clean_rate": n_clean / rows_written if rows_written else None,
        "git_sha": git_sha(os.path.dirname(os.path.abspath(out_path))),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "fields": POLICY_TABLE_FIELDS,
    }
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print(f"policy table written : {out_path}  ({rows_written} rows)")
    print(f"  sidecar meta       : {meta_path}")
    print(f"  lint-clean         : {n_clean} ({n_clean / max(rows_written, 1):.1%})")
    print(f"  missing confidence : {missing_conf}")
    if no_attempts:
        print(f"  [WARN] {len(no_attempts)} leaf dir(s) had no attempts.jsonl "
              f"(run generated without --trace?): {no_attempts[:3]}")
    print("=" * 60)
    if rows_written == 0:
        raise SystemExit("[ERROR] 0 rows written — check --runs / --rounds.")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the reranker-independent policy table (confidence + lint verdict)"
    )
    ap.add_argument("--runs", nargs="+", required=True, help="KernelBench run roots")
    ap.add_argument("--out", required=True, help="Output policy_table.jsonl path")
    ap.add_argument("--levels", nargs="+", type=int, default=None,
                    help="Per-run level (one int per --runs); default: inferred")
    ap.add_argument("--rounds", nargs="+", type=int, default=None,
                    help="Lint-loop rounds to read; omit for the final kernel per sample")
    args = ap.parse_args()
    if args.levels is not None and len(args.levels) != len(args.runs):
        ap.error(f"--levels has {len(args.levels)} entries but there are {len(args.runs)} --runs")
    build_policy_table(args.runs, args.out, args.levels, args.rounds)


if __name__ == "__main__":
    main()
