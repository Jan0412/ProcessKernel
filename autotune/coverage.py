"""How many generated kernels can we actually tune?

Everything downstream rests on this: if the patcher can only reach a small fraction of the
corpus, the sweep measures a biased subsample and A2 is not a fair baseline. So we measure
it up front, on CPU, before spending any GPU time -- over the kernels already sitting in
runs/, which is the same code distribution the experiment will generate.

    uv run python -m autotune.coverage --runs-dir runs --patch-sample 2000

Reports two numbers:
  * discovery coverage -- share of kernels with at least one tunable knob;
  * patch coverage -- share of (kernel, config) pairs from a random sample that patch,
    re-parse, and verify. This is the one that can hide bugs, so it is checked by actually
    applying every config in the kernel's grid.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from autotune.grids import build_grid
from autotune.knobs import analyze
from autotune.patcher import Unpatchable, patch_source


def correct_kernels(runs_dir: Path) -> list[Path]:
    """The kernels the sweep would actually touch: the correct ones.

    Enumerated from each run's eval_results.json rather than by globbing. A glob over the
    ~175k generated files takes tens of minutes on GPFS, and the incorrect ones are not
    swept anyway -- there is nothing to tune about a kernel that does not run.
    """
    out = []
    for results_path in sorted(runs_dir.glob("*/eval_results.json")):
        run_dir = results_path.parent
        match = re.search(r"level[_-]?(\d+)", run_dir.name, re.IGNORECASE)
        if not match:
            continue
        level = int(match.group(1))
        for pid, samples in json.loads(results_path.read_text()).items():
            for s in samples:
                if s.get("correctness") and s.get("runtime", -1) > 0:
                    out.append(
                        run_dir / f"level_{level}_problem_{pid}_sample_{s['sample_id']}_kernel.py"
                    )
    return out


def _discover(path: Path) -> tuple[str, str, int]:
    """(outcome, ndim_class, n_configs) for one kernel file."""
    try:
        src = path.read_text()
    except OSError as e:
        return f"unreadable: {type(e).__name__}", "none", 0
    rep = analyze(src)
    if rep.parse_error:
        return "parse_error", "none", 0
    if not rep.n_jit_kernels:
        return "no_triton_kernel", "none", 0
    if not rep.n_launches:
        # A @triton.jit kernel that is defined but never called: ModelNew's forward is plain
        # PyTorch. These can still pass KernelBench's correctness check and be "fast" -- they
        # are just not Triton kernels, which is worth knowing about the corpus.
        return "kernel_never_launched", "none", 0
    if not rep.knobs:
        return "no_tunable_knob", rep.ndim_class, len(build_grid(rep))
    return "tunable", rep.ndim_class, len(build_grid(rep))


def _patch_check(path: Path) -> tuple[int, int, str | None]:
    """(configs_ok, configs_total, first_failure) -- apply the whole grid to one kernel."""
    try:
        src = path.read_text()
    except OSError:
        return 0, 0, "unreadable"
    rep = analyze(src)
    if rep.parse_error or not rep.n_jit_kernels:
        return 0, 0, None  # not a patcher failure; counted in discovery instead
    ok, fail = 0, None
    grid = build_grid(rep)
    for cfg in grid:
        try:
            patch_source(src, cfg, rep)
            ok += 1
        except Unpatchable as e:
            if fail is None:
                fail = f"{cfg}: {e}"
        except Exception as e:  # a patcher crash is a bug, not a skip -- surface it
            if fail is None:
                fail = f"{cfg}: CRASH {type(e).__name__}: {e}"
    return ok, len(grid), fail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--patch-sample", type=int, default=2000,
                    help="kernels to fully patch-verify (0 = all; slow)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kernels = correct_kernels(Path(args.runs_dir))
    print(f"{len(kernels):,} CORRECT kernels under {args.runs_dir}/ "
          f"(the population the sweep would touch)\n")

    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_discover, kernels, chunksize=256))

    outcomes = collections.Counter(r[0] for r in results)
    shapes = collections.Counter(r[1] for r in results if r[0] == "tunable")
    configs = [r[2] for r in results if r[0] == "tunable"]

    n = len(kernels)
    print("=== knob discovery ===")
    for name, count in outcomes.most_common():
        print(f"  {name:<20} {count:>8,}  ({100 * count / n:>5.1f}%)")
    tunable = outcomes["tunable"]
    print(f"\n  DISCOVERY COVERAGE: {100 * tunable / n:.1f}%  ({tunable:,} kernels have >=1 knob)")

    print("\n=== shape of the tunable ones ===")
    for name, count in shapes.most_common():
        print(f"  {name:<20} {count:>8,}  ({100 * count / max(tunable, 1):>5.1f}%)")
    if configs:
        print(f"\n  configs per kernel: mean {sum(configs) / len(configs):.1f}, "
              f"total if all swept: {sum(configs):,} evals")

    sample = [k for k, r in zip(kernels, results) if r[0] == "tunable"]
    if args.patch_sample and len(sample) > args.patch_sample:
        random.Random(args.seed).shuffle(sample)
        sample = sample[: args.patch_sample]

    print(f"\n=== patch verification ({len(sample):,} kernels, whole grid each) ===")
    with ProcessPoolExecutor(args.workers) as pool:
        checks = list(pool.map(_patch_check, sample, chunksize=32))

    ok = sum(c[0] for c in checks)
    total = sum(c[1] for c in checks)
    clean = sum(1 for c in checks if c[2] is None and c[1] > 0)
    failures = [(p, c[2]) for p, c in zip(sample, checks) if c[2] is not None]

    print(f"  configs applied cleanly: {ok:,} / {total:,}  ({100 * ok / max(total, 1):.2f}%)")
    print(f"  kernels with a fully clean grid: {clean:,} / {len(sample):,} "
          f"({100 * clean / max(len(sample), 1):.1f}%)")
    if failures:
        # Full paths: the same basename exists in several run dirs, so a bare name is not
        # enough to find the offender again.
        print(f"\n  {len(failures)} kernels had at least one config fail. First 10:")
        for path, why in failures[:10]:
            print(f"    {path}\n      {why}")


if __name__ == "__main__":
    main()
