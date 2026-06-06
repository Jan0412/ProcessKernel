"""Convert a subset of GPUMODE/KernelBook into a KernelBench local-dataset level.

Writes one KernelBench-style problem file per selected row, named
``{row_idx}_{module_name}.py`` (leading int = the dataset row index, which becomes
the canonical problem-id used by generation and eval), plus a ``manifest.json``
mapping each row_idx to its KernelBook provenance.

Point ``--out`` at the KernelBench local level directory you intend to evaluate
against, e.g. ``${HOME}/KernelBench/KernelBench/level5``. Then run the generation
scripts with the same ``--rows`` and evaluate with ``dataset_src=local level=5``.

Example:
    python convert_kernelbook.py \\
        --rows 0-499 \\
        --smoke-test \\
        --out ~/KernelBench/KernelBench/level5
"""

import argparse
import json
import os

from kernelbook_convert import ConversionError, convert_row, smoke_test

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
        "--max-src-chars",
        type=int,
        default=24000,
        help="Skip rows whose python_code exceeds this many chars (token-budget guard)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing problem files")
    args = parser.parse_args()

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
    print(f"Smoke test       : {args.smoke_test} (device={args.smoke_device})")

    manifest_path = os.path.join(args.out, "manifest.json")
    manifest: dict = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    stats = {"written": 0, "skipped_size": 0, "skipped_convert": 0, "skipped_smoke": 0, "skipped_exists": 0}

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
            continue

        fname = f"{row_id}_{module_name}.py"
        fpath = os.path.join(args.out, fname)
        if os.path.exists(fpath) and not args.overwrite:
            stats["skipped_exists"] += 1
            continue

        try:
            converted = convert_row(python_code, module_name)
        except ConversionError as e:
            stats["skipped_convert"] += 1
            print(f"[SKIP convert] row {row_id} ({module_name}): {e}")
            continue

        if args.smoke_test:
            ok, reason = smoke_test(converted, device=args.smoke_device)
            if not ok:
                stats["skipped_smoke"] += 1
                print(f"[SKIP smoke] row {row_id} ({module_name}): {reason}")
                continue

        with open(fpath, "w") as f:
            f.write(converted)
        manifest[str(row_id)] = {
            "uuid": row.get("uuid"),
            "module_name": module_name,
            "entry_point": row.get("entry_point"),
            "repo_link": row.get("repo_link"),
            "file": fname,
        }
        stats["written"] += 1
        print(f"[OK] row {row_id} → {fname}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone.")
    print(
        f"  written={stats['written']}  skipped_convert={stats['skipped_convert']}  "
        f"skipped_smoke={stats['skipped_smoke']}  skipped_size={stats['skipped_size']}  "
        f"skipped_exists={stats['skipped_exists']}"
    )
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
