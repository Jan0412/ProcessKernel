"""Read out an A5 run: did the lint loop actually produce better kernels?

    uv run python -m kernel_gen.readout --run-dir runs/<A5 run>

Joins three files: the refined run's ``eval_results.json``, ``rounds/round_0/``'s
``eval_results.json`` (the unrefined baseline), and ``lint_loop.jsonl`` (what the linter
said, round by round). Both evals must have been produced first -- this reads, it does
not measure.

**The headline is the paired transition table, not the linter's numbers.** The loop
optimizes the linter's score by construction, so "findings went down" is circular and
proves exactly nothing. What the linter's numbers *are* good for is the mechanism: which
checks resist repair, and whether F1.6 (a passthrough kernel that copies its input and
computes nothing) starts appearing once there is a feedback loop -- which is precisely
what that check was written expecting. Read that section as a diagnostic, and read it
with the caveat printed underneath it.

Because round 0 and the refined run share every sample slot, the comparison is *paired*:

    fixed    round 0 was wrong, the refinement made it correct   <- the case for the loop
    broken   round 0 was CORRECT and the refinement broke it     <- the case against it
    kept     correct both times (and then: did it get faster?)
    neither  wrong both times

An unpaired "63% vs 58% correct" cannot tell you that a loop which fixes 40 kernels and
breaks 35 others is a coin flip dressed up as an improvement. This can.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter

from triton_lint.runs import iter_samples, speedup


def load_slots(run_dir: str) -> dict[tuple[int, int], dict]:
    """``{(problem_id, sample_id): outcome}`` for one evaluated run dir."""
    out: dict[tuple[int, int], dict] = {}
    for sample in iter_samples(run_dir):
        out[(sample.problem_id, sample.sample_id)] = {
            "compiled": sample.compiled,
            "correct": sample.correct,
            "runtime": sample.runtime,
            "speedup": speedup(sample),
        }
    return out


def load_trajectories(run_dir: str) -> dict[tuple[int, int], dict]:
    path = os.path.join(run_dir, "lint_loop.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            out[(record["problem_id"], record["sample_id"])] = record
    return out


def transitions(baseline: dict, refined: dict) -> dict[str, list]:
    """Bucket every slot the two runs share by what refinement did to it."""
    buckets: dict[str, list] = {"fixed": [], "broken": [], "kept": [], "neither": []}
    for key in sorted(baseline.keys() & refined.keys()):
        was, now = baseline[key]["correct"], refined[key]["correct"]
        if now and not was:
            buckets["fixed"].append(key)
        elif was and not now:
            buckets["broken"].append(key)
        elif was and now:
            buckets["kept"].append(key)
        else:
            buckets["neither"].append(key)
    return buckets


def _rate(n: int, total: int) -> str:
    return f"{n:>5} ({n / total:>5.1%})" if total else f"{n:>5}"


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def report_correctness(baseline: dict, refined: dict) -> dict:
    shared = baseline.keys() & refined.keys()
    total = len(shared)
    buckets = transitions(baseline, refined)

    n_base = sum(1 for k in shared if baseline[k]["correct"])
    n_ref = sum(1 for k in shared if refined[k]["correct"])

    print("=" * 68)
    print("  Correctness -- refined vs its own round 0, slot for slot")
    print("-" * 68)
    print(f"  slots evaluated in both : {total}")
    print(f"  correct at round 0      : {_rate(n_base, total)}")
    print(f"  correct after refinement: {_rate(n_ref, total)}")
    print("-" * 68)
    print(f"  fixed   (wrong -> right): {_rate(len(buckets['fixed']), total)}")
    print(f"  broken  (right -> wrong): {_rate(len(buckets['broken']), total)}  <-- the regression")
    print(f"  kept    (right -> right): {_rate(len(buckets['kept']), total)}")
    print(f"  neither (wrong -> wrong): {_rate(len(buckets['neither']), total)}")
    print("-" * 68)
    net = len(buckets["fixed"]) - len(buckets["broken"])
    print(f"  net kernels gained      : {net:+d}")
    if buckets["broken"]:
        shown = ", ".join(f"p{p}/s{s}" for p, s in buckets["broken"][:8])
        print(f"  broken slots            : {shown}"
              f"{' …' if len(buckets['broken']) > 8 else ''}")

    return {
        "n_slots": total,
        "n_correct_round0": n_base,
        "n_correct_refined": n_ref,
        **{k: len(v) for k, v in buckets.items()},
        "net": net,
    }


def report_speed(baseline: dict, refined: dict, buckets: dict) -> dict:
    """Speedup, on the slots that were correct BOTH times.

    Restricting to `kept` is the only honest comparison: a slot that was wrong at round 0
    has no runtime to compare against, and including the newly-fixed ones would let the
    loop look faster simply by adding kernels that did not exist before.
    """
    pairs = [
        (baseline[k]["speedup"], refined[k]["speedup"])
        for k in buckets["kept"]
        if baseline[k]["speedup"] and refined[k]["speedup"]
    ]
    print()
    print("=" * 68)
    print("  Speedup vs eager -- on slots correct in BOTH runs")
    print("-" * 68)
    if not pairs:
        print("  no slot was correct in both runs with a usable timing")
        print("=" * 68)
        return {}

    base_s = [b for b, _ in pairs]
    ref_s = [r for _, r in pairs]
    faster = sum(1 for b, r in pairs if r > b)

    print(f"  paired slots            : {len(pairs)}")
    print(f"  mean speedup at round 0 : {statistics.mean(base_s):.2f}x")
    print(f"  mean speedup refined    : {statistics.mean(ref_s):.2f}x")
    print(f"  median round 0 / refined: {statistics.median(base_s):.2f}x / "
          f"{statistics.median(ref_s):.2f}x")
    print(f"  got faster              : {_rate(faster, len(pairs))}")
    print("=" * 68)
    return {
        "n_paired": len(pairs),
        "mean_speedup_round0": statistics.mean(base_s),
        "mean_speedup_refined": statistics.mean(ref_s),
        "n_faster": faster,
    }


def report_lint(trajectories: dict, rounds: int) -> dict:
    """The mechanism, NOT the result. See this module's docstring."""
    if not trajectories:
        print("\n(no lint_loop.jsonl -- skipping the mechanism section)")
        return {}

    print()
    print("=" * 68)
    print("  What the linter saw, round by round  [mechanism, not result]")
    print("-" * 68)

    per_round_ids: list[Counter] = [Counter() for _ in range(rounds)]
    per_round_n = [0] * rounds
    clean_at = Counter()

    for record in trajectories.values():
        for entry in record["rounds"]:
            r = entry["round"]
            if r >= rounds:
                continue
            per_round_n[r] += 1
            for check_id in entry.get("check_ids", []):
                per_round_ids[r][check_id] += 1
        if record.get("clean"):
            clean_at[record["final_round"]] += 1

    total = len(trajectories)
    print(f"  trajectories            : {total}")
    print(f"  lint-clean at the end   : {_rate(sum(clean_at.values()), total)}")
    for r in range(rounds):
        if per_round_n[r]:
            print(f"    round {r}: {per_round_n[r]:>5} ran, "
                  f"{clean_at[r]:>5} went clean here")

    check_ids = sorted({c for counter in per_round_ids for c in counter})
    if check_ids:
        print("-" * 68)
        header = "  check   " + "".join(f"  round {r}" for r in range(rounds))
        print(header)
        for check_id in check_ids:
            cells = "".join(
                f"  {per_round_ids[r][check_id] / per_round_n[r]:>6.1%}"
                if per_round_n[r] else "       -"
                for r in range(rounds)
            )
            print(f"  {check_id:<8}{cells}")
        print("-" * 68)
        print("  Rates are over the slots that RAN that round, and the slots that run a")
        print("  later round are exactly the ones the linter was unhappy with -- so a rate")
        print("  that climbs is expected and means nothing on its own. F1.6 is the one to")
        print("  watch: it fires on a kernel that copies its input and computes nothing,")
        print("  and a loop is exactly the thing that could induce one. But a rise is")
        print("  equally consistent with 'the model learned to evade a check we put in its")
        print("  prompt', and this arm alone cannot separate those two.")
    print("=" * 68)

    return {
        "n_trajectories": total,
        "n_clean": sum(clean_at.values()),
        "clean_by_round": dict(clean_at),
        "check_counts_by_round": [dict(c) for c in per_round_ids],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, help="the A5 run directory")
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="default: <run-dir>/rounds/round_0 -- the run's own unrefined round 0",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--json", default=None, help="also write the numbers here")
    args = parser.parse_args()

    baseline_dir = args.baseline_dir or os.path.join(args.run_dir, "rounds", "round_0")

    for path in (args.run_dir, baseline_dir):
        if not os.path.exists(os.path.join(path, "eval_results.json")):
            raise SystemExit(
                f"No eval_results.json in {path}.\n"
                f"Evaluate both halves first -- this script reads, it does not measure:\n"
                f"  uv run python -m autotune.eval_run --run-dir {args.run_dir}\n"
                f"  uv run python -m autotune.eval_run --run-dir {baseline_dir}"
            )

    refined = load_slots(args.run_dir)
    baseline = load_slots(baseline_dir)
    trajectories = load_trajectories(args.run_dir)

    summary = report_correctness(baseline, refined)
    summary.update(report_speed(baseline, refined, transitions(baseline, refined)))
    summary["lint"] = report_lint(trajectories, args.rounds)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
