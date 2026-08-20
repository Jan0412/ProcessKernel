"""What the pool's verdicts say: what the ORM picked, what it could have picked, and the gap.

The oracle here is over what the search *finished*. It cannot see a candidate the PRM pruned
mid-generation -- that needs `keep_pruned` and a second eval pass, and until then "the PRM
never killed a winner" is not something this report is entitled to say.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict


def report(trees_dir: str, verdicts: dict) -> dict:
    """`{"overall": {...}, "by_level": {level: {...}}}` over every tree in `trees_dir`."""
    per_level = defaultdict(list)
    missing = 0
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
            rows.append((e, bool(verdict.get("compiled")) and bool(verdict.get("correctness"))))
        if not rows:
            continue

        picked = tree.get("picked")
        picked_ok = any(ok for e, ok in rows if picked and e["cid"] == picked["cid"])
        best_prm = max(rows, key=lambda pair: _last(pair[0]))
        per_level[level].append({
            "picked_correct": picked_ok,
            "oracle_correct": any(ok for _, ok in rows),
            "prm_top1_hit": best_prm[1],
            # Finished pool size, not the graded subset -- a missing verdict must not shrink
            # the reported pool (that's what n_missing_verdict is for).
            "pool_size": len(tree["pool"]),
        })

    by_level = {level: _summarise(rows) for level, rows in per_level.items()}
    flat = [r for rows in per_level.values() for r in rows]
    overall = _summarise(flat)
    overall["stop_reasons"] = dict(stops)
    overall["n_missing_verdict"] = missing
    return {"overall": overall, "by_level": by_level}


def _last(entry) -> float:
    scores = entry.get("prm_scores") or []
    return scores[-1] if scores else float("-inf")


def _summarise(rows) -> dict:
    if not rows:
        return {"n_problems": 0}
    n = len(rows)
    picked = sum(r["picked_correct"] for r in rows) / n
    oracle = sum(r["oracle_correct"] for r in rows) / n
    return {
        "n_problems": n,
        "picked_correct": picked,
        "oracle_correct": oracle,
        "selection_regret": oracle - picked,
        "prm_top1_hit": sum(r["prm_top1_hit"] for r in rows) / n,
        "pool_size": sum(r["pool_size"] for r in rows) / n,
    }


def main(argv=None) -> None:
    import argparse

    from reranker.src.prm import corpus

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", required=True)
    parser.add_argument("--eval-results", required=True, nargs="+",
                        help="eval_results.json files that together grade the pool run")
    args = parser.parse_args(argv)
    verdicts: dict = {}
    for path in args.eval_results:
        for key, entry in corpus.verdicts(path).items():
            if key in verdicts:
                raise ValueError(f"{key} is graded by two of the given files")
            verdicts[key] = entry
    print(json.dumps(report(args.trees, verdicts), indent=2))


if __name__ == "__main__":
    main()
