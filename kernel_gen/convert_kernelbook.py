"""Convert a subset of GPUMODE/KernelBook into a KernelBench local-dataset level.

Writes one KernelBench-style problem file per selected row, named
``{row_idx}_{module_name}.py`` (leading int = the dataset row index, which becomes
the canonical problem-id used by generation and eval), plus a ``manifest.json``
mapping each row_idx to its KernelBook provenance.

Point ``--out`` at the KernelBench local level directory you intend to evaluate
against, e.g. ``${HOME}/KernelBench/KernelBench/level5``. Then run the generation
scripts with the same ``--rows`` and evaluate with ``dataset_src=local level=5``.

KernelBook ships every shape as the placeholder value ``4`` (batch and feature dims
alike), which is far too small to be meaningful on an A100/H100 — kernel-launch
overhead dominates. By default this converter up-scales those placeholder dims to
GPU-meaningful sizes (``--scale uniform``), rewriting ``get_inputs`` and
``get_init_inputs`` together so the shapes stay mutually consistent, while leaving
genuinely meaningful dims (RGB ``3``, spatial ``64``, ``784``) untouched. With
``--smoke-test`` it walks a fallback ladder (full → halved → quartered → un-scaled)
and keeps the largest size whose Model actually runs; the chosen scale is recorded
per row in the manifest. Pass ``--scale off`` for the original tiny-input behaviour.

"Actually runs" is judged against *eval* conditions, not an empty card. Two guards
make the smoke test reject a scale that would ship a reference every sample OOMs on:

  * ``--smoke-mem-gb`` caps the CUDA allocator to the memory a sample really gets at
    eval (it shares the GPU with the candidate model and other processes). Without
    it, a reference needing e.g. 32 GiB passes on an idle 80GB card, then OOMs for
    all 10 samples at eval — labelling every candidate wrong regardless of quality.
  * ``--max-numel-ratio`` rejects "coupled init-dim" blowups, where a placeholder dim
    that becomes an *output axis* (e.g. ``Linear(1, dimension)``) is up-scaled along
    with the input, turning an ``S**2`` input into an ``S**3`` output.

A rejected rung sends the row one step down the ladder (it is shrunk and re-tested),
so such rows are kept at a smaller GPU-meaningful size rather than dropped. Every
rejected rung and its reason is recorded per row in the manifest.

Example:
    python convert_kernelbook.py \\
        --rows 0-499 \\
        --scale uniform \\
        --smoke-test \\
        --smoke-mem-gb 12 \\
        --max-numel-ratio 50 \\
        --out ~/KernelBench/KernelBench/level6
"""

import argparse
import json
import os

from kernelbook_convert import (
    DEFAULT_SCALE_BY_RANK,
    ConversionError,
    ScaleConfig,
    SmokeRunner,
    convert_row,
)

KERNELBOOK_SPLIT = "train"  # KernelBook ships a single split


def parse_rows(spec: str) -> list[int]:
    """Parse '0-499', '1,5,10', or '23' into a list of ints."""
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Convert KernelBook rows into a KernelBench local-dataset level"
    )
    parser.add_argument(
        "--rows",
        default=None,
        help="Row indices: single '23', range '0-499', or comma-list '1,5,10'. Omit for --all.",
    )
    parser.add_argument("--all", action="store_true", help="Convert every row in the dataset")
    parser.add_argument("--out", required=True, help="Output level directory (created if missing)")
    parser.add_argument("--dataset-name", default="GPUMODE/KernelBook")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Only keep rows whose converted Model runs one eager forward (needs a GPU)",
    )
    parser.add_argument(
        "--smoke-device",
        default="cuda",
        help="Device for --smoke-test (default: cuda; use 'cpu' to dry-run without a GPU)",
    )
    parser.add_argument(
        "--smoke-timeout",
        type=float,
        default=300.0,
        help="Per-row time budget in seconds for --smoke-test build+forward; "
        "rows exceeding it are rejected (default: 300; 0 disables)",
    )
    parser.add_argument(
        "--smoke-mem-gb",
        type=float,
        default=12.0,
        help="Per-row GPU memory budget in GiB for --smoke-test (default: 12; 0 "
        "disables). The smoke test runs on an *empty* card, but at eval time a "
        "sample shares the GPU with the candidate model and other processes — so "
        "a reference validated against the full 80GB can still OOM every sample "
        "at eval. Capping the allocator here makes such a scale fail the smoke "
        "test, so --scale-fallback down-scales the row instead of shipping a "
        "reference that poisons all its labels.",
    )
    parser.add_argument(
        "--max-numel-ratio",
        type=float,
        default=50.0,
        help="Reject a scaled row whose forward output element count exceeds this "
        "multiple of its input element count (default: 50; 0 disables). Catches "
        "'coupled init-dim' blowups, where a placeholder dim that becomes an "
        "output axis (e.g. Linear(1, dimension)) is up-scaled alongside the input, "
        "turning an S**2 input into an S**3 output — absurd even when it fits.",
    )
    parser.add_argument(
        "--max-src-chars",
        type=int,
        default=24000,
        help="Skip rows whose python_code exceeds this many chars (token-budget guard)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing problem files")
    parser.add_argument(
        "--scale",
        choices=["uniform", "batch-only", "off"],
        default="uniform",
        help="Up-scale KernelBook's placeholder '4' dims to GPU-meaningful sizes. "
        "'uniform' scales batch+coupled feature dims together (rank-aware, default); "
        "'batch-only' scales just the leading dim; 'off' keeps the original tiny inputs.",
    )
    parser.add_argument(
        "--placeholder-dim",
        type=int,
        default=4,
        help="The placeholder dim value KernelBook uses for every shape (default: 4)",
    )
    parser.add_argument(
        "--batch-target",
        type=int,
        default=2048,
        help="Leading-dim target for --scale batch-only (default: 2048)",
    )
    parser.add_argument(
        "--scale-by-rank",
        default=None,
        help="Override per-rank targets as 'rank:size,...' (e.g. '2:1024,4:32'); "
        f"defaults to {DEFAULT_SCALE_BY_RANK}",
    )
    parser.add_argument(
        "--scale-fallback",
        default="1.0,0.5,0.25",
        help="Comma list of scale factors to try in order before falling back to "
        "un-scaled inputs, when --smoke-test rejects the up-scaled Model (default: 1.0,0.5,0.25)",
    )
    args = parser.parse_args()

    scale_by_rank = dict(DEFAULT_SCALE_BY_RANK)
    if args.scale_by_rank:
        for part in args.scale_by_rank.split(","):
            rank, size = part.split(":")
            scale_by_rank[int(rank)] = int(size)

    def make_scale(factor: float) -> ScaleConfig:
        return ScaleConfig(
            placeholder=args.placeholder_dim,
            mode=args.scale,
            scale_by_rank=scale_by_rank,
            batch_target=args.batch_target,
            factor=factor,
        )

    fallback_factors = [float(x) for x in args.scale_fallback.split(",")] if args.scale else []

    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset

    print(f"Loading dataset {args.dataset_name} split={KERNELBOOK_SPLIT} …")
    dataset = load_dataset(args.dataset_name, split=KERNELBOOK_SPLIT)

    if args.all:
        row_ids = list(range(len(dataset)))
    elif args.rows:
        row_ids = parse_rows(args.rows)
    else:
        raise ValueError("Provide --rows or --all.")

    print(f"Output level dir : {args.out}")
    print(f"Rows to convert  : {len(row_ids)}")
    print(
        f"Smoke test       : {args.smoke_test} (device={args.smoke_device}, "
        f"timeout={args.smoke_timeout:g}s, mem_budget={args.smoke_mem_gb:g}GiB, "
        f"max_numel_ratio={args.max_numel_ratio:g}x)"
    )

    manifest_path = os.path.join(args.out, "manifest.json")
    manifest: dict = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    stats = {
        "written": 0,
        "downscaled": 0,
        "skipped_size": 0,
        "skipped_convert": 0,
        "skipped_smoke": 0,
        "skipped_exists": 0,
        # Rows where at least one scale rung was rejected by the new guards, i.e.
        # a reference that would have OOM'd (or been absurd) at eval and poisoned
        # every one of its labels. These are the rows the guards actually saved.
        "guard_oom": 0,
        "guard_blowup": 0,
    }

    # Each row's smoke test runs in an isolated, restartable child process so a
    # pathological row (OOM, segfault, CUDA abort, or hang) only kills the child
    # and is recorded as a skip — never the whole run. Reused so CUDA stays warm.
    runner = (
        SmokeRunner(
            device=args.smoke_device,
            timeout=args.smoke_timeout,
            mem_budget_gb=args.smoke_mem_gb,
            max_numel_ratio=args.max_numel_ratio,
        )
        if args.smoke_test
        else None
    )

    for row_id in row_ids:
        try:
            row = dataset[row_id]
        except IndexError:
            print(f"[WARN] row {row_id} out of range, skipping")
            continue

        module_name = row.get("module_name") or row.get("entry_point") or ""
        python_code = row.get("python_code") or ""

        if len(python_code) > args.max_src_chars:
            stats["skipped_size"] += 1
            print(f"[SKIP size] row {row_id} ({module_name}): {len(python_code)} > {args.max_src_chars}")
            continue

        fname = f"{row_id}_{module_name}.py"
        fpath = os.path.join(args.out, fname)
        if os.path.exists(fpath) and not args.overwrite:
            stats["skipped_exists"] += 1
            print(f"[SKIP exists] row {row_id} ({module_name}): {fname}")
            continue

        # Candidate scale configs, tried in order. Without --smoke-test we can't
        # validate, so we take the first (most-aggressive) candidate as-is. With it,
        # we walk the ladder down to un-scaled inputs and keep the first that runs.
        if args.scale == "off":
            candidates = [("off", make_scale(1.0))]
        else:
            candidates = [(f"{args.scale}x{f:g}", make_scale(f)) for f in fallback_factors]
            candidates.append(("off", ScaleConfig(placeholder=args.placeholder_dim, mode="off")))

        converted = None
        applied = None
        last_reason = ""
        # Every rung the ladder had to reject, kept for the manifest so it is
        # auditable *why* a row landed at the scale it did (memory budget, output
        # blowup, plain crash) rather than just recording the surviving scale.
        rejected: list[dict] = []
        for label, cfg in candidates:
            try:
                cand = convert_row(python_code, module_name, scale=cfg)
            except ConversionError as e:
                last_reason = str(e)
                rejected.append({"scale": label, "reason": last_reason})
                continue
            if not args.smoke_test:
                converted, applied = cand, label
                break
            ok, reason = runner.run(cand)
            if ok:
                converted, applied = cand, label
                break
            last_reason = reason
            rejected.append({"scale": label, "reason": reason})

        # Did the guards (not a plain crash) force this row down the ladder? These
        # are exactly the references that would have OOM'd every sample at eval.
        blowup_hit = any("footprint blowup" in r["reason"] for r in rejected)
        oom_hit = any(
            "OutOfMemory" in r["reason"] or "out-of-memory" in r["reason"]
            for r in rejected
        )
        if blowup_hit:
            stats["guard_blowup"] += 1
        if oom_hit:
            stats["guard_oom"] += 1

        if converted is None:
            if args.smoke_test:
                stats["skipped_smoke"] += 1
                print(f"[SKIP smoke] row {row_id} ({module_name}): {last_reason}")
            else:
                stats["skipped_convert"] += 1
                print(f"[SKIP convert] row {row_id} ({module_name}): {last_reason}")
            continue

        with open(fpath, "w") as f:
            f.write(converted)
        manifest[str(row_id)] = {
            "uuid": row.get("uuid"),
            "module_name": module_name,
            "entry_point": row.get("entry_point"),
            "repo_link": row.get("repo_link"),
            "file": fname,
            "scale": applied,
            "rejected_scales": rejected,
        }
        stats["written"] += 1
        if applied not in ("off", f"{args.scale}x1"):
            stats["downscaled"] += 1
        print(f"[OK] row {row_id} → {fname} (scale={applied})")

    if runner is not None:
        runner.close()

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    stats_meta = {
        "dataset_name": args.dataset_name,
        "rows_requested": len(row_ids),
        "max_src_chars": args.max_src_chars,
        "smoke_test": args.smoke_test,
        "smoke_device": args.smoke_device,
        "smoke_mem_gb": args.smoke_mem_gb,
        "max_numel_ratio": args.max_numel_ratio,
        "overwrite": args.overwrite,
        **stats,
    }
    stats_path = os.path.join(args.out, "conversion_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_meta, f, indent=2)

    print("\nDone.")
    print(
        f"  written={stats['written']} (downscaled={stats['downscaled']})  "
        f"skipped_convert={stats['skipped_convert']}  "
        f"skipped_smoke={stats['skipped_smoke']}  skipped_size={stats['skipped_size']}  "
        f"skipped_exists={stats['skipped_exists']}"
    )
    print(
        f"  guards   : mem-budget caught {stats['guard_oom']} rows, "
        f"numel-ratio caught {stats['guard_blowup']} rows "
        f"(these would have OOM'd at eval and mislabeled every sample)"
    )
    print(f"  manifest : {manifest_path}")
    print(f"  stats    : {stats_path}")


if __name__ == "__main__":
    main()
