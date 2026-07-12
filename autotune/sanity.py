"""Does the sweep measure the same thing the evaluator measured?

    uv run python -m autotune.sanity --run-dir runs/<run>

This is the check that has to pass before any tuning gain this project reports can be
believed, and it is worth being explicit about why.

The sweep reports

    tuning gain = (runtime at the kernel's own constants) / (runtime at the best config)

If the numerator came from eval_results.json and the denominator from the sweep, then any
systematic difference between the two harnesses -- a different process, a different warmup,
a different L2 cache state -- would land directly in that ratio and look exactly like a
speedup. We would get a clean, publishable-looking result that was pure measurement artifact.

So the sweep re-measures the kernel's own constants itself, as config 0, through the same
code path. This script verifies that config 0 agrees with what the evaluator independently
recorded for the same kernel. Agreement means the two harnesses are interchangeable and the
ratio is honest. Disagreement means they are not, and nothing downstream is trustworthy.

We expect scatter of a few percent -- these are separate processes on a shared GPU. What we
are looking for is *bias*: a median ratio that is not 1.0 says one harness is systematically
faster than the other.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# A consistent offset larger than this means the harnesses disagree about what they measure.
BIAS_TOLERANCE = 0.05  # 5% on the median


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tolerance", type=float, default=BIAS_TOLERANCE)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    evaluated = json.loads((run_dir / "eval_results.json").read_text())
    summary = json.loads((run_dir / "sweep" / "sweep_summary.json").read_text())

    # eval_results.json is keyed by problem -> [samples]; index it by kernel stem instead.
    level = next(iter(summary)).split("_")[1] if summary else None
    eval_ms: dict[str, float] = {}
    for pid, samples in evaluated.items():
        for s in samples:
            if s.get("correctness") and s.get("runtime", -1) > 0:
                stem = f"level_{level}_problem_{pid}_sample_{s['sample_id']}_kernel"
                eval_ms[stem] = float(s["runtime"])

    rows = []
    for kernel, entry in summary.items():
        sweep_ms = entry.get("identity_ms")
        ref_ms = eval_ms.get(kernel)
        if sweep_ms and ref_ms:
            rows.append((kernel, ref_ms, sweep_ms, sweep_ms / ref_ms))

    if not rows:
        raise SystemExit(
            "No kernel has both an evaluator runtime and a swept identity runtime.\n"
            "Did the sweep run with --finalize? (Only the final phase re-times config 0.)"
        )

    ratios = [r[3] for r in rows]
    median = statistics.median(ratios)
    print(f"=== identity config: sweep vs evaluator, {len(rows)} kernels ===\n")
    print(f"  {'kernel':<44} {'eval ms':>9} {'sweep ms':>9} {'ratio':>7}")
    print("  " + "-" * 72)
    for kernel, ref, sweep, ratio in sorted(rows, key=lambda r: -abs(r[3] - 1))[:15]:
        flag = "  <-- off" if abs(ratio - 1) > 0.15 else ""
        print(f"  {kernel:<44} {ref:>9.4f} {sweep:>9.4f} {ratio:>7.3f}{flag}")

    spread = statistics.pstdev(ratios) if len(ratios) > 1 else 0.0
    within = sum(1 for r in ratios if abs(r - 1) <= 0.10)
    print(f"\n  median ratio      {median:.4f}   (1.000 = the harnesses agree)")
    print(f"  spread (sd)       {spread:.4f}")
    print(f"  within +/-10%     {within}/{len(ratios)} ({100 * within / len(ratios):.0f}%)")

    print("\n" + "=" * 72)
    if abs(median - 1.0) <= args.tolerance:
        print(f"  SANITY PASSED — median {median:.4f} is within {args.tolerance:.0%} of 1.0.")
        print("  The sweep and the evaluator measure the same thing, so a tuning gain is a")
        print("  real speedup rather than an artifact of two different harnesses.")
    else:
        direction = "FASTER" if median < 1.0 else "SLOWER"
        print(f"  SANITY FAILED — the sweep is systematically {direction} than the evaluator")
        print(f"  (median ratio {median:.4f}, tolerance {args.tolerance:.0%}).")
        print()
        print("  DO NOT trust any tuning gain until this is explained. A bias of this size")
        print("  feeds straight into the gain ratio and would masquerade as a speedup.")
        print("  Likely causes: different warmup or trial counts, a different timing method,")
        print("  L2 cache state, or the two runs landing on different GPUs.")
        raise SystemExit(1)
    print("=" * 72)


if __name__ == "__main__":
    main()
