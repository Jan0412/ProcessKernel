"""Emit one FileReport per shipped kernel, so a refactor can be proven behaviour-neutral.

The `checker/` refactor restructures the analyzer without changing what it detects. That
claim is only worth as much as its evidence, so this dumps `analyze_source(...).to_json()`
for every kernel kernel_gen has ever written and the refactor is required to reproduce the
file byte for byte:

    python scripts/verify_report_parity.py -o baseline.jsonl     # on the branch point
    ...refactor...
    python scripts/verify_report_parity.py -o new.jsonl && diff baseline.jsonl new.jsonl

`to_json` carries check_id, severity, message and the whole data payload, so an empty diff
means no predicate, message, severity or payload moved on real inputs.

Imports `checker` if it exists and `triton_lint` otherwise, so the same script runs on both
sides of the rename.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    from checker import analyze_source
except ModuleNotFoundError:
    from triton_lint import analyze_source

# The kernel_gen corpus: lint-loop runs plus the two single-shot runs. NOT the KernelBook
# runs -- those are 994k files, and a gate that costs 16 minutes stops being run after every
# converted check, which is the only thing that makes it useful. Pass --runs '*' with the
# traced-run root on --runs-dir for the wider sweep the final acceptance gate wants.
DEFAULT_RUNS = ("*lintloop*", "Qwen3.6-27B_level1_triton", "Qwen3.6-27B_level2_triton")

_ONLY: set[str] | None = None


def corpus(runs_dirs: list[str], patterns: tuple[str, ...]) -> list[str]:
    """Every kernel in the matching runs, top level and per round, sorted."""
    import fnmatch

    paths = []
    for runs_dir in runs_dirs:
        if not os.path.isdir(runs_dir):
            continue
        for entry in sorted(os.listdir(runs_dir)):
            if not any(fnmatch.fnmatch(entry, p) for p in patterns):
                continue
            for dirpath, _, filenames in os.walk(os.path.join(runs_dir, entry)):
                for name in filenames:
                    if name.startswith("level_") and name.endswith("_kernel.py"):
                        paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def _init(only: set[str] | None) -> None:
    global _ONLY
    _ONLY = only


def _one(path: str) -> str:
    try:
        source = open(path, encoding="utf-8", errors="replace").read()
    except OSError as exc:
        return f'{{"path": {os.path.relpath(path, REPO_ROOT)!r}, "read_error": "{exc}"}}'
    report = analyze_source(source, os.path.relpath(path, REPO_ROOT), only=_ONLY)
    return report.to_json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir", default=os.path.join(REPO_ROOT, "runs"), help="comma-separated roots"
    )
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="glob patterns")
    parser.add_argument("--only", default=None, help="restrict to these check ids")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 8)
    args = parser.parse_args()

    paths = corpus(args.runs_dir.split(","), tuple(args.runs.split(",")))
    only = set(args.only.split(",")) if args.only else None
    print(f"{len(paths)} kernels", file=sys.stderr)

    with mp.get_context("fork").Pool(args.jobs, initializer=_init, initargs=(only,)) as pool:
        lines = pool.map(_one, paths, chunksize=64)

    with open(args.out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
