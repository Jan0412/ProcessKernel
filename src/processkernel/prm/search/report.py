"""What the pool's verdicts say: what the ORM picked, what it could have picked, and the gap.

The oracle here is over what the search *finished*. It cannot see a candidate the PRM pruned
mid-generation -- that needs `keep_pruned` and a second eval pass, and until then "the PRM
never killed a winner" is not something this report is entitled to say.

With `--baseline`, each summary also carries KernelBench's speed metrics (geometric mean
speedup, fast_p) for three selectors: the ORM's pick, the PRM's top-1, and the oracle.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import Counter, defaultdict

SELECTORS = ("orm", "prm_top1", "oracle")
P_VALUES = (0.0, 0.5, 0.8, 1.0, 1.5, 2.0)
# KernelBench's convention (benchmark_eval_analysis.py), not the search's `speedup_stat: min`.
CONVENTION = "baseline.mean / kernel.runtime"
ORACLE_RULE = "fastest correct candidate in the finished pool"


def report(trees_dir: str, verdicts: dict, baseline: dict | None = None) -> dict:
    """`{"overall": {...}, "by_level": {level: {...}}}` over every tree in `trees_dir`."""
    per_level = defaultdict(list)
    missing = 0
    bad_runtime = Counter()
    stops = Counter()

    for path in sorted(glob.glob(os.path.join(trees_dir, "problem_*.json"))):
        with open(path) as fh:
            tree = json.load(fh)
        level, pid = tree["level"], tree["problem_id"]
        rows = []
        for e in tree["pool"]:
            stops[e["stop"]] += 1
            verdict = verdicts.get((pid, e["sample_id"]))
            if verdict is None:
                missing += 1
                continue
            ok = bool(verdict.get("compiled")) and bool(verdict.get("correctness"))
            rows.append((e, ok, verdict))
        if not rows:
            continue

        picked = tree.get("picked")
        picked_ok = any(ok for e, ok, _ in rows if picked and e["cid"] == picked["cid"])
        best_prm = max(range(len(rows)), key=lambda i: _last(rows[i][0]))
        summary = {
            "picked_correct": picked_ok,
            "oracle_correct": any(ok for _, ok, _ in rows),
            "prm_top1_hit": rows[best_prm][1],
            # Finished pool size, not the graded subset -- a missing verdict must not shrink
            # the reported pool (that's what n_missing_verdict is for).
            "pool_size": len(tree["pool"]),
        }
        if baseline is not None:
            summary["speed"] = _speedups(tree, rows, picked, best_prm, baseline, bad_runtime)
        per_level[level].append(summary)

    with_speed = baseline is not None
    by_level = {level: _summarise(rows, with_speed) for level, rows in per_level.items()}
    flat = [r for rows in per_level.values() for r in rows]
    overall = _summarise(flat, with_speed)
    overall["stop_reasons"] = dict(stops)
    overall["n_missing_verdict"] = missing
    if with_speed:
        overall["n_bad_runtime"] = bad_runtime["n"]
    return {"overall": overall, "by_level": by_level}


def _last(entry) -> float:
    scores = entry.get("prm_scores") or []
    return scores[-1] if scores else float("-inf")


def _speedups(tree, rows, picked, best_prm, baseline, bad_runtime) -> dict | None:
    """One problem's speedup per selector, `None` where that selector's pick was not correct.

    `None` for the whole problem means no baseline timing, so no ratio is defined -- those
    problems leave the speed block's denominator entirely.
    """
    base = (baseline.get(f"level{tree['level']}") or {}).get(tree["name"]) or {}
    ref = base.get("mean")
    if not ref or ref <= 0:
        return None

    # Once per candidate, not once per selector: the selectors overlap, and a second pass
    # would count the same unusable runtime again.
    ups = []
    for _, ok, verdict in rows:
        runtime = verdict.get("runtime") or 0.0
        if ok and runtime <= 0:
            bad_runtime["n"] += 1
        ups.append(ref / runtime if ok and runtime > 0 else None)

    correct = [u for u in ups if u is not None]
    orm = next((ups[i] for i, r in enumerate(rows)
                if picked and r[0]["cid"] == picked["cid"]), None)
    # The fastest correct kernel maximises fast_p at every threshold at once, so one oracle
    # column covers all of them.
    return {"orm": orm, "prm_top1": ups[best_prm], "oracle": max(correct, default=None)}


def _summarise(rows, with_speed: bool = False) -> dict:
    if not rows:
        return {"n_problems": 0}
    n = len(rows)
    picked = sum(r["picked_correct"] for r in rows) / n
    oracle = sum(r["oracle_correct"] for r in rows) / n
    out = {
        "n_problems": n,
        "picked_correct": picked,
        "oracle_correct": oracle,
        "selection_regret": oracle - picked,
        "prm_top1_hit": sum(r["prm_top1_hit"] for r in rows) / n,
        "pool_size": sum(r["pool_size"] for r in rows) / n,
    }
    if with_speed:
        out["speed"] = _speed(rows)
    return out


def _speed(rows) -> dict:
    """KernelBench's speed metrics per selector, over the problems that have a baseline.

    `n` here is below `n_problems` whenever a baseline timing is missing, so the block's own
    correctness_rate is the one to compare its fast_p against.
    """
    graded = [r["speed"] for r in rows if r.get("speed")]
    n = len(graded)
    out = {
        "convention": CONVENTION,
        "oracle_rule": ORACLE_RULE,
        "n": n,
        "n_missing_baseline": len(rows) - n,
    }
    for selector in SELECTORS:
        ups = [g[selector] for g in graded if g[selector] is not None]
        out[selector] = {
            "correct_count": len(ups),
            "correctness_rate": len(ups) / n if n else 0.0,
            "geo_mean_speedup": _geo_mean(ups),
            # score.fastp: strictly greater than p, over every problem -- not just the correct
            # ones. Log space because a few hundred ratios overflow the straight product.
            "fast_p": {str(p): (sum(u > p for u in ups) / n if n else 0.0) for p in P_VALUES},
        }
    return out


def _geo_mean(values) -> float:
    if not values:
        return 0.0
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def main(argv=None) -> None:
    import argparse

    from processkernel.prm.data import corpus

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", required=True)
    parser.add_argument("--eval-results", required=True, nargs="+",
                        help="eval_results.json files that together grade the pool run")
    parser.add_argument("--baseline",
                        help="KernelBench baseline_time_*.json; without it there is no speed block")
    args = parser.parse_args(argv)
    verdicts: dict = {}
    for path in args.eval_results:
        for key, entry in corpus.verdicts(path).items():
            if key in verdicts:
                raise ValueError(f"{key} is graded by two of the given files")
            verdicts[key] = entry
    baseline = None
    if args.baseline:
        with open(args.baseline) as fh:
            baseline = json.load(fh)
    print(json.dumps(report(args.trees, verdicts, baseline), indent=2))


if __name__ == "__main__":
    main()
