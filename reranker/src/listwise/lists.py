"""Build the listwise training/validation data from the labeled source dataset.

Reads the same labeled rows the pointwise/pairwise pipelines use
(``data.dataset_jsonl``), then:

  1. Builds a *fresh* problem-level train/val split (no test) — all candidates of
     a problem stay in one split, so the reranker is never validated on a problem
     it trained on. Optionally stratified by KernelBench level.
  2. For each problem, pools its candidates across all runs (keyed by
     ``(level, problem_id)``), keeps only **compiling** kernels, dedups
     near-identical sources, and assigns a graded relevance:
       negative (compiled-but-wrong) -> rel 0
       correct                       -> rel = 1 + p, where p in [0, 1] is the
                                        normalized speedup over the per-problem
                                        PyTorch baseline (KernelBench `fast_p`
                                        style). Correct kernels with no baseline
                                        speedup are dropped (never guessed).
     One fixed-size list per problem is emitted (``list_size`` budget, up to
     ``max_positives`` positives), so candidate-rich problems do not dominate.

Emits ``lists_train.jsonl`` / ``lists_val.jsonl`` (each line = one problem's
candidate list, referencing source rows by ``(run_name, sample_id)`` with their
``rel``) and ``lists_splits.json`` (reusable by RerankerDataset for pointwise
validation / model selection).

Usage:
    bash reranker/scripts/build_lists.sh [configs/listwise_config.yaml]
    python -m reranker.src.listwise.lists --config configs/listwise_config.yaml
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

from reranker.src.config import RerankerConfig, _resolve, load_config
from reranker.src.data.labels import code_hash, speed_p


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- fresh problem-level train/val split (mirrors pairwise/pairs.py) ----------
def _two_way_split(keys: list, train_ratio: float, rng: random.Random) -> dict:
    """Assign a shuffled list of problem keys to train/val (no test)."""
    keys = list(keys)
    rng.shuffle(keys)
    n_train = int(round(train_ratio * len(keys)))
    return {k: ("train" if i < n_train else "val") for i, k in enumerate(keys)}


def _build_split(rows: list[dict], cfg: RerankerConfig, rng: random.Random) -> dict:
    """Fresh problem-level train/val split over (level, problem_id) keys."""
    lw = cfg.listwise
    ratios = lw.split_ratios
    if len(ratios) != 2 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"listwise.split_ratios must be [train, val] summing to 1.0, got {ratios}")
    train_ratio = ratios[0]

    problem_level: dict[tuple, int] = {}
    for r in rows:
        problem_level[(r["level"], r["problem_id"])] = r["level"]

    assignment: dict[tuple, str] = {}
    if lw.stratify_by_level:
        by_level: dict[int, list] = defaultdict(list)
        for key, level in problem_level.items():
            by_level[level].append(key)
        for _level, keys in sorted(by_level.items()):
            assignment.update(_two_way_split(keys, train_ratio, rng))
    else:
        assignment.update(_two_way_split(list(problem_level), train_ratio, rng))
    return assignment


def _write_splits(assignment: dict, splits_path: str) -> None:
    serializable = {f"{lvl}:{pid}": split for (lvl, pid), split in assignment.items()}
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(serializable, f, indent=2)


# --- relevance + list construction -------------------------------------------
def _ref(row: dict, rel: float) -> dict:
    """Compact reference to a source row within its (level, problem_id) group."""
    return {"run_name": row["run_name"], "sample_id": row["sample_id"], "rel": round(float(rel), 6)}


def _subsample_positives_by_spread(
    positives: list[tuple[dict, float]], k: int, rng: random.Random
) -> list[tuple[dict, float]]:
    """Keep ``k`` positives that preserve the speed spread (not a uniform sample).

    Sort by relevance (speed) and pick evenly spaced ranks including both extremes,
    so the fastest and slowest correct kernels are always kept and the list retains
    a real fast-vs-slow gradient instead of possibly collapsing to one speed bucket.
    """
    if len(positives) <= k:
        return positives
    if k <= 1:
        return [max(positives, key=lambda pr: pr[1])]  # keep the single fastest
    ordered = sorted(positives, key=lambda pr: pr[1])
    n = len(ordered)
    idxs = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    if len(idxs) < k:  # rounding collapsed some ranks — top up from the rest
        remaining = [i for i in range(n) if i not in idxs]
        rng.shuffle(remaining)
        idxs = sorted(set(idxs) | set(remaining[: k - len(idxs)]))
    return [ordered[i] for i in idxs]


def _build_list_for_problem(rows: list[dict], lw, rng: random.Random) -> list[dict] | None:
    """Return one problem's candidate list (refs with rel), or None to skip it.

    Skips when there is no usable positive (correct kernel with a real baseline
    speedup), too few candidates, or no two distinct relevances to rank.
    """
    # Keep only compiling kernels (defensive even if the dataset still has
    # compile-fails) and dedup near-identical sources.
    seen: set[str] = set()
    compiling: list[dict] = []
    for r in rows:
        if not r.get("compiled"):
            continue
        if lw.dedup_by_code_hash:
            h = code_hash(r["kernel_src"])
            if h in seen:
                continue
            seen.add(h)
        compiling.append(r)

    # Positives = correct kernels graded by speedup; drop those without a baseline.
    speedup_field = "speedup_min" if lw.speedup_stat == "min" else "speedup"
    positives: list[tuple[dict, float]] = []
    for r in compiling:
        if r["label"] == 1:
            su = r.get(speedup_field)
            if su is None or su <= 0:
                continue
            positives.append((r, 1.0 + speed_p(su, lw.speedup_lo, lw.speedup_hi, lw.speed_quant)))
    negatives = [r for r in compiling if r["label"] == 0]

    if len(positives) < max(lw.min_positives, 1):
        return None
    if len(positives) + len(negatives) < lw.min_list_size:
        return None

    # Positive-rich budget: cap positives while preserving the speed spread, then
    # keep only a bounded number of negatives (enough to teach the correct-vs-wrong
    # gate, not so many they drown the fast-vs-slow pairs).
    if len(positives) > lw.max_positives:
        positives = _subsample_positives_by_spread(positives, lw.max_positives, rng)
    neg_budget = max(min(lw.max_negatives, lw.list_size - len(positives)), 0)
    if len(negatives) > neg_budget:
        negatives = rng.sample(negatives, neg_budget)

    selected = positives + [(r, 0.0) for r in negatives]
    if len({rel for _, rel in selected}) < 2:
        return None  # no valid ranking pair (all-equal relevance)

    rng.shuffle(selected)
    return [_ref(r, rel) for r, rel in selected]


def _emit_lists(
    by_problem: dict[tuple, list[dict]],
    split_problems: set[tuple],
    lw,
    rng: random.Random,
    out_path: str,
) -> dict[str, int]:
    """Write all lists for `split_problems`. Returns build stats."""
    n_lists = n_skipped = total_cand = total_pos = 0
    with open(out_path, "w") as f:
        for key in sorted(split_problems):
            level, problem_id = key
            refs = _build_list_for_problem(by_problem.get(key, []), lw, rng)
            if refs is None:
                n_skipped += 1
                continue
            f.write(json.dumps({
                "level": level,
                "problem_id": problem_id,
                "candidates": refs,
            }) + "\n")
            n_lists += 1
            total_cand += len(refs)
            total_pos += sum(1 for c in refs if c["rel"] > 0)
    return {
        "n_lists": n_lists,
        "n_skipped": n_skipped,
        "total_cand": total_cand,
        "total_pos": total_pos,
    }


def build_lists(cfg: RerankerConfig) -> tuple[str, str, str]:
    """Build lists_train.jsonl, lists_val.jsonl, lists_splits.json. Returns their paths."""
    lw = cfg.listwise
    if lw.speedup_stat not in ("mean", "min"):
        raise ValueError(f"listwise.speedup_stat must be 'mean' or 'min', got {lw.speedup_stat!r}")
    source_path = _resolve(cfg.data.dataset_jsonl)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(
            f"Source dataset not found: {source_path}. Build it with reranker.src.data.build_dataset first."
        )

    rows = _read_jsonl(source_path)
    by_problem: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_problem[(r["level"], r["problem_id"])].append(r)

    rng = random.Random(lw.split_seed)
    assignment = _build_split(rows, cfg, rng)
    splits_path = _resolve(lw.lists_splits_json)
    _write_splits(assignment, splits_path)

    train_problems = {k for k, s in assignment.items() if s == "train"}
    val_problems = {k for k, s in assignment.items() if s == "val"}

    list_rng = random.Random(lw.list_seed)
    train_path = _resolve(lw.lists_train_jsonl)
    val_path = _resolve(lw.lists_val_jsonl)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    print(f"\n[lists] source       : {source_path}  ({len(rows)} rows)")
    print(f"[lists] list_size    : {lw.list_size}  (max_positives {lw.max_positives})")
    print(f"[lists] speedup map  : {lw.speedup_lo}x -> 0 .. {lw.speedup_hi}x -> 1  (rel = 1 + p)")
    print(f"[lists] speedup stat : {lw.speedup_stat}  (quant {lw.speed_quant})")
    print(f"[lists] problems     : train {len(train_problems)} / val {len(val_problems)}")

    stats = {
        "train": _emit_lists(by_problem, train_problems, lw, list_rng, train_path),
        "val": _emit_lists(by_problem, val_problems, lw, list_rng, val_path),
    }

    print("\n" + "=" * 60)
    for split, st in stats.items():
        path = train_path if split == "train" else val_path
        mean_len = st["total_cand"] / max(st["n_lists"], 1)
        mean_pos = st["total_pos"] / max(st["n_lists"], 1)
        print(f"  {split:5s}: {st['n_lists']} lists "
              f"(mean {mean_len:.1f} candidates, {mean_pos:.1f} positives; "
              f"{st['n_skipped']} problems skipped — no positive / too few / all-equal)")
        print(f"         -> {path}")
    print(f"  splits -> {splits_path}")
    print("=" * 60)

    if stats["train"]["n_lists"] == 0:
        print("[ERROR] 0 training lists written. Check negative_mode / speedup coverage / dataset labels.")
        sys.exit(1)
    return train_path, val_path, splits_path


def main() -> None:
    cfg = load_config()
    build_lists(cfg)


if __name__ == "__main__":
    main()
