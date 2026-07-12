"""Final statistics for the four arms.

    uv run python -m autotune.analysis \\
        --a1-dir runs/<r1> --a3-dir runs/<r2timing> --a4-dir runs/<r2tuning> --level 1

Every arm is scored the same way -- best correct kernel per problem, measured at its best
correct launch config -- so what we compare is the quality of the *algorithm* the model
produced, with the constants held at their best everywhere. A1 is the exception by
definition: it is the as-generated runtime, which is what makes A2-vs-A1 the measure of what
tuning alone buys.

Two comparisons carry the result:

  A4 vs A2  does an extra LLM round beat simply running the tuner and going home?
  A4 vs A3  is a tuning result a better feedback signal than a plain timing? -- the claim.

Everything is paired per problem and bootstrapped. With a few hundred problems and GPU
timings that wobble by a few percent, an unpaired difference of means will happily report an
effect that is not there.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

from autotune.knobs import gaming_report

FASTP_THRESHOLDS = (1.0, 1.5, 2.0)


def load_baseline(path: Path, level: int) -> dict[int, float]:
    raw = json.loads(path.read_text())
    out = {}
    for fname, stats in raw.get(f"level{level}", {}).items():
        m = re.match(r"(\d+)_", str(fname))
        if m and isinstance(stats, dict) and stats.get("mean") is not None:
            out[int(m.group(1))] = float(stats["mean"])
    return out


def a1_scores(run_dir: Path) -> dict[int, float]:
    """Best correct AS-GENERATED runtime per problem."""
    results = json.loads((run_dir / "eval_results.json").read_text())
    out = {}
    for pid, samples in results.items():
        times = [float(s["runtime"]) for s in samples
                 if s.get("correctness") and s.get("runtime", -1) > 0]
        if times:
            out[int(pid)] = min(times)
    return out


def tuned_scores(run_dir: Path) -> dict[int, float]:
    """Best TUNED runtime per problem, over the swept samples."""
    summary = json.loads((run_dir / "sweep" / "sweep_summary.json").read_text())
    out: dict[int, float] = {}
    for entry in summary.values():
        pid, ms = entry.get("problem_id"), entry.get("best_ms")
        if pid is None or not ms:
            continue
        if pid not in out or ms < out[pid]:
            out[pid] = ms
    return out


def seed_scores(run_dir: Path) -> dict[int, float]:
    """The tuned runtime of the seed each round-2 problem started from (= that problem's A2)."""
    seeds = json.loads((run_dir / "seeds.json").read_text())["seeds"]
    return {int(pid): s["best_ms"] for pid, s in seeds.items() if s.get("best_ms")}


def bootstrap(pairs: list[tuple[float, float]], n: int = 10_000, seed: int = 0) -> dict:
    """Paired bootstrap on log-ratios: how much faster is arm X than arm Y, per problem?

    We work in log space because these are ratios; the geometric mean is the meaningful
    average of a speedup, and an arithmetic mean of ratios is dominated by whichever problem
    happened to produce a 20x outlier.
    """
    import math

    if not pairs:
        return {"n": 0}
    logs = [math.log(x / y) for x, y in pairs]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [logs[rng.randrange(len(logs))] for _ in range(len(logs))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {
        "n": len(pairs),
        "geomean_ratio": math.exp(sum(logs) / len(logs)),
        "ci95": (math.exp(means[int(0.025 * n)]), math.exp(means[int(0.975 * n)])),
        "median_ratio": math.exp(statistics.median(logs)),
        "wins": sum(1 for x, y in pairs if x < y),  # lower runtime = better
        "losses": sum(1 for x, y in pairs if x > y),
    }


def _report_pair(name: str, a: dict[int, float], b: dict[int, float], label_a: str, label_b: str) -> None:
    """a and b are {problem: runtime_ms}; lower is better. Reports speedup of a over b."""
    common = sorted(set(a) & set(b))
    pairs = [(a[p], b[p]) for p in common]
    r = bootstrap(pairs)
    if not r["n"]:
        print(f"  {name:<12} no overlapping problems")
        return
    lo, hi = r["ci95"]
    # geomean_ratio is a/b in runtime; invert so >1 means "a is faster".
    speed, slo, shi = 1 / r["geomean_ratio"], 1 / hi, 1 / lo
    sig = "  " if slo <= 1.0 <= shi else " *"
    print(f"  {name:<12} {speed:>6.3f}x  [{slo:.3f}, {shi:.3f}]{sig}  "
          f"n={r['n']:<4} {label_a} wins {r['wins']}, {label_b} wins {r['losses']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a1-dir", required=True, help="round-1 run dir (holds A1 and, via sweep/, A2)")
    ap.add_argument("--a3-dir", default=None, help="round-2 timing-feedback run dir")
    ap.add_argument("--a4-dir", default=None, help="round-2 tuning-feedback run dir")
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--baseline-file", default="timing/H100/baseline_time_torch.json")
    args = ap.parse_args()

    a1_dir = Path(args.a1_dir)
    baseline = load_baseline(Path(args.baseline_file), args.level)

    arms: dict[str, dict[int, float]] = {
        "A1 as-generated": a1_scores(a1_dir),
        "A2 tuned (no LLM)": tuned_scores(a1_dir),
    }
    if args.a3_dir:
        a3 = tuned_scores(Path(args.a3_dir))
        seeds3 = seed_scores(Path(args.a3_dir))
        arms["A3 timing fb (pure)"] = a3
        arms["A3 timing fb (keep-best)"] = _keep_best(a3, seeds3)
    if args.a4_dir:
        a4 = tuned_scores(Path(args.a4_dir))
        seeds4 = seed_scores(Path(args.a4_dir))
        arms["A4 tuning fb (pure)"] = a4
        arms["A4 tuning fb (keep-best)"] = _keep_best(a4, seeds4)

    print("=" * 78)
    print(f"  LEVEL {args.level}")
    print("=" * 78)

    print("\n=== fast_p: share of problems beating PyTorch eager by at least p ===\n")
    header = f"  {'arm':<26} {'n':>4} " + " ".join(f"{'>=' + str(t) + 'x':>8}" for t in FASTP_THRESHOLDS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, scores in arms.items():
        speedups = [baseline[p] / ms for p, ms in scores.items() if p in baseline and ms]
        if not speedups:
            continue
        cells = " ".join(
            f"{100 * sum(1 for s in speedups if s >= t) / len(speedups):>7.1f}%"
            for t in FASTP_THRESHOLDS
        )
        print(f"  {name:<26} {len(speedups):>4} {cells}")

    print("\n=== paired comparisons (speedup of the first arm over the second) ===")
    print("    * = 95% bootstrap CI excludes 1.0\n")
    if "A2 tuned (no LLM)" in arms and "A1 as-generated" in arms:
        _report_pair("A2 vs A1", arms["A2 tuned (no LLM)"], arms["A1 as-generated"], "A2", "A1")
    if "A3 timing fb (pure)" in arms:
        _report_pair("A3 vs A1", arms["A3 timing fb (pure)"], arms["A1 as-generated"], "A3", "A1")
    if "A4 tuning fb (pure)" in arms:
        print()
        print("  --- the two that decide the experiment ---")
        _report_pair("A4 vs A2", arms["A4 tuning fb (pure)"], arms["A2 tuned (no LLM)"], "A4", "A2")
        if "A3 timing fb (pure)" in arms:
            _report_pair("A4 vs A3", arms["A4 tuning fb (pure)"], arms["A3 timing fb (pure)"], "A4", "A3")
        print()
        print("  A4 vs A2 : does the extra LLM round beat just running the tuner?")
        print("  A4 vs A3 : is tuning information a better signal than raw timing?  <- the claim")

    _tuning_gain(a1_dir)
    if args.a4_dir:
        _gaming(Path(args.a4_dir), Path(args.a3_dir) if args.a3_dir else None)


def _keep_best(pure: dict[int, float], seeds: dict[int, float]) -> dict[int, float]:
    """What you would actually deploy: the round-2 kernel, unless the seed was better."""
    out = dict(seeds)
    for pid, ms in pure.items():
        if pid not in out or ms < out[pid]:
            out[pid] = ms
    return out


def _tuning_gain(a1_dir: Path) -> None:
    summary = json.loads((a1_dir / "sweep" / "sweep_summary.json").read_text())
    gains = [e["tuning_gain"] for e in summary.values() if e.get("tuning_gain")]
    if not gains:
        return
    flat = sum(1 for g in gains if g < 1.05)
    broke = sum(1 for e in summary.values() if e.get("n_wrong_result", 0) > 0)
    print("\n=== what tuning alone bought (arm A2) ===\n")
    print(f"  median gain            {statistics.median(gains):.3f}x   over {len(gains):,} kernels")
    print(f"  flat response surface  {100 * flat / len(gains):.1f}%  (<1.05x -- nothing for A4 to read)")
    print(f"  correctness broke at   {100 * broke / len(summary):.1f}%  of kernels when the "
          f"block size changed")
    print(f"                         (these passed KernelBench only at the constants they "
          f"happened to pick)")


def _gaming(a4_dir: Path, a3_dir: Path | None) -> None:
    """Did A4 game the benchmark by baking in the config it was shown?

    KernelBench always evaluates one input shape. A model told "1024 was fastest" can delete
    the tl.constexpr parameter and hardcode 1024: that scores well here and produces a kernel
    specialised to a single shape -- the opposite of what it was asked for. A3 is the control:
    it was never shown a config, so whatever rate it shows is the background rate.
    """
    print("\n=== gaming check: did the model hardcode what we showed it? ===\n")
    for label, run_dir in (("A4 (was shown configs)", a4_dir), ("A3 (control)", a3_dir)):
        if run_dir is None:
            continue
        seeds = json.loads((run_dir / "seeds.json").read_text())["seeds"]
        n, untunable, hardcoded = 0, 0, 0
        for pid, seed in seeds.items():
            best = seed.get("best_config") or {}
            for kernel in run_dir.glob(f"*_problem_{pid}_sample_*_kernel.py"):
                rep = gaming_report(kernel.read_text(), best)
                if rep.get("parse_error"):
                    continue
                n += 1
                untunable += bool(rep["is_untunable"])
                hardcoded += bool(rep["hardcoded_fed_value"])
        if not n:
            continue
        print(f"  {label:<24} {n:>5} kernels")
        print(f"    dropped the tl.constexpr knob entirely : {100 * untunable / n:>5.1f}%")
        print(f"    wrote exactly the config we showed it  : {100 * hardcoded / n:>5.1f}%"
              + ("   <- compare against the control" if "A4" in label else ""))


if __name__ == "__main__":
    main()
