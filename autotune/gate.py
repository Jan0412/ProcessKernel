"""The go/no-go readout after the round-1 sweep (arm A2).

    uv run python -m autotune.gate --run-dirs runs/<r1_level1> runs/<r1_level2>

One number decides whether the rest of the experiment is worth running: the median tuning
gain, i.e. how much faster the typical kernel gets purely from turning its launch knobs.

If that number is small, it does not merely weaken A2 -- it kills A4 as well, and the
reason is worth stating plainly. A4's premise is that the tuning result carries information
the model can act on. But a low tuning gain means the config-to-latency table is *flat*:
every row reports the same runtime. A flat table has nothing in it. We would be handing the
model a list of identical numbers and asking it to draw a conclusion. So a single
measurement can falsify the boring idea and the interesting idea at the same time, which is
exactly what makes it worth buying early -- it costs about a third of the total budget.

This script prints the distribution and a suggested verdict. It does not decide: a low
median with a fat tail (say 20% of kernels gaining 2x) means tuning matters intensely for
*some* kernels, and the right move is to narrow the experiment to those rather than abandon
it. Read the numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GATE_THRESHOLD = 1.10


def load(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in run_dirs:
        path = run_dir / "sweep" / "sweep_summary.json"
        if not path.exists():
            print(f"!! missing {path} -- run the sweep with --finalize first")
            continue
        for kernel, entry in json.loads(path.read_text()).items():
            rows.append({"run": run_dir.name, "kernel": kernel, **entry})
    return rows


def _hist(gains: list[float]) -> None:
    buckets = [(0, 1.01, "no gain      (<1.01x)"), (1.01, 1.05, "negligible   (1.01-1.05x)"),
               (1.05, 1.25, "moderate     (1.05-1.25x)"), (1.25, 2.0, "large        (1.25-2x)"),
               (2.0, float("inf"), "very large   (>2x)")]
    for lo, hi, label in buckets:
        n = sum(1 for g in gains if lo <= g < hi)
        bar = "#" * round(40 * n / max(len(gains), 1))
        print(f"  {label:<26} {n:>5,} ({100 * n / len(gains):>5.1f}%) {bar}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=GATE_THRESHOLD)
    args = ap.parse_args()

    rows = load([Path(d) for d in args.run_dirs])
    if not rows:
        raise SystemExit("no sweep summaries found")

    gains = [r["tuning_gain"] for r in rows if r.get("tuning_gain")]
    print(f"=== A2: tuning gain over {len(gains):,} swept kernels "
          f"({len(rows) - len(gains):,} had no gain measurable) ===\n")
    if not gains:
        raise SystemExit("no kernel produced both an identity and a best timing")

    med = statistics.median(gains)
    print(f"  median   {med:.3f}x")
    print(f"  p25/p75  {_pct(gains, 25):.3f}x / {_pct(gains, 75):.3f}x")
    print(f"  p90/max  {_pct(gains, 90):.3f}x / {max(gains):.3f}x")
    print(f"  mean     {statistics.mean(gains):.3f}x\n")
    _hist(gains)

    above = sum(1 for g in gains if g >= args.threshold)
    print(f"\n  kernels gaining >= {args.threshold}x: {above:,} / {len(gains):,} "
          f"({100 * above / len(gains):.1f}%)")

    edge = sum(1 for r in rows if r.get("at_grid_edge"))
    print(f"  winners at the edge of the grid: {edge:,} / {len(rows):,} "
          f"({100 * edge / len(rows):.1f}%)  <- the search wanted to keep going")

    # The fuzzer finding. Given how low correctness rates are, this may be worth more than
    # the speed story: a config that changes the answer means the kernel's correctness was
    # silently riding on the constant the model happened to write.
    broke = [r for r in rows if r.get("n_wrong_result", 0) > 0]
    print(f"\n=== correctness-vs-config ===")
    print(f"  kernels that produce a WRONG RESULT at some other block size: "
          f"{len(broke):,} / {len(rows):,} ({100 * len(broke) / len(rows):.1f}%)")
    print(f"  (these passed KernelBench only at the constants they happened to pick)")

    flat = sum(1 for g in gains if g < 1.05)
    print(f"\n=== response surface ===")
    print(f"  flat (<1.05x -- no information to feed back to A4): "
          f"{100 * flat / len(gains):.1f}% of kernels")

    print("\n" + "=" * 60)
    if med >= args.threshold:
        print(f"  GATE: GO   -- median gain {med:.3f}x >= {args.threshold}x")
        print("  Tuning moves the needle and the config tables carry signal. Proceed to A3/A4.")
    else:
        print(f"  GATE: NO-GO -- median gain {med:.3f}x < {args.threshold}x")
        print("  Turning the knobs buys little, so the config tables are close to flat and")
        print("  there is little for the model to read in them. Before abandoning, check the")
        print(f"  tail above: {100 * above / len(gains):.1f}% of kernels still gain >= {args.threshold}x.")
        print("  If that tail is substantial, narrow the experiment to it rather than stop.")
    print("=" * 60)


def _pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p / 100))]


if __name__ == "__main__":
    main()
