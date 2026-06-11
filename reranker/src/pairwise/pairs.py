"""Build the pairwise training/validation dataset from the labeled source dataset.

Reads the same labeled rows the pointwise pipeline uses (``data.dataset_jsonl``),
then:

  1. Builds a *fresh* problem-level train/val split (no test) — all candidates of
     a problem stay in one split, so the reranker is never validated on a problem
     it trained on. Optionally stratified by KernelBench level.
  2. For each problem, pools its candidates across all runs and forms
     (positive, negative) pairs (full cross product) per ``pair_mode``:
       compiled_wrong : negative = compiled but numerically incorrect (hard).
       all_negative   : negative = anything with label 0 (incl. compile failures).
     An optional ``max_negatives_per_positive`` caps the explosion.

Emits ``pairs_train.jsonl`` / ``pairs_val.jsonl`` (pairs reference source rows by
the unique key (run_name, level, problem_id, sample_id)) and ``pairs_splits.json``
(reusable by RerankerDataset for pointwise validation).

Usage:
    bash reranker/scripts/build_pairs.sh [configs/pairwise_config.yaml]
    python -m reranker.src.pairwise.pairs --config configs/pairwise_config.yaml
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict

from reranker.src.config import RerankerConfig, _resolve, load_config


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _two_way_split(
    keys: list, train_ratio: float, rng: random.Random
) -> dict:
    """Assign a shuffled list of problem keys to train/val (no test)."""
    keys = list(keys)
    rng.shuffle(keys)
    n_train = int(round(train_ratio * len(keys)))
    return {k: ("train" if i < n_train else "val") for i, k in enumerate(keys)}


def _build_split(rows: list[dict], cfg: RerankerConfig, rng: random.Random) -> dict:
    """Fresh problem-level train/val split over (level, problem_id) keys."""
    pw = cfg.pairwise
    ratios = pw.split_ratios
    if len(ratios) != 2 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"pairwise.split_ratios must be [train, val] summing to 1.0, got {ratios}")
    train_ratio = ratios[0]

    problem_level: dict[tuple, int] = {}
    for r in rows:
        problem_level[(r["level"], r["problem_id"])] = r["level"]

    assignment: dict[tuple, str] = {}
    if pw.stratify_by_level:
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


def _eligible_negatives(rows: list[dict], pair_mode: str) -> list[dict]:
    if pair_mode == "compiled_wrong":
        return [r for r in rows if r.get("compiled") and not r.get("correct")]
    if pair_mode == "all_negative":
        return [r for r in rows if r["label"] == 0]
    raise ValueError(f"pair_mode must be 'compiled_wrong' or 'all_negative', got {pair_mode}")


def _ref(row: dict) -> dict:
    """Compact reference to a source row within its (level, problem_id) group."""
    return {"run_name": row["run_name"], "sample_id": row["sample_id"]}


def _emit_pairs_for_split(
    by_problem: dict[tuple, list[dict]],
    split_problems: set[tuple],
    pair_mode: str,
    max_neg: int | None,
    rng: random.Random,
    out_f,
) -> tuple[int, int, int]:
    """Write all pairs for the problems in `split_problems`.

    Returns (n_pairs, n_positives_used, n_positives_without_pair).
    """
    n_pairs = 0
    n_pos_used = 0
    n_pos_no_pair = 0
    for key in split_problems:
        rows = by_problem.get(key, [])
        positives = [r for r in rows if r["label"] == 1]
        negatives = _eligible_negatives(rows, pair_mode)
        if not positives:
            continue
        if not negatives:
            n_pos_no_pair += len(positives)
            continue
        level, problem_id = key
        for pos in positives:
            negs = negatives
            if max_neg is not None and len(negs) > max_neg:
                negs = rng.sample(negs, max_neg)
            n_pos_used += 1
            for neg in negs:
                out_f.write(json.dumps({
                    "level": level,
                    "problem_id": problem_id,
                    "pair_mode": pair_mode,
                    "pos": _ref(pos),
                    "neg": _ref(neg),
                }) + "\n")
                n_pairs += 1
    return n_pairs, n_pos_used, n_pos_no_pair


def build_pairs(cfg: RerankerConfig) -> tuple[str, str, str]:
    """Build pairs_train.jsonl, pairs_val.jsonl, pairs_splits.json. Returns their paths."""
    pw = cfg.pairwise
    source_path = _resolve(cfg.data.dataset_jsonl)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(
            f"Source dataset not found: {source_path}. Build it with reranker.src.data.build_dataset first."
        )

    rows = _read_jsonl(source_path)
    by_problem: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_problem[(r["level"], r["problem_id"])].append(r)

    rng = random.Random(pw.split_seed)
    assignment = _build_split(rows, cfg, rng)
    splits_path = _resolve(pw.pairs_splits_json)
    _write_splits(assignment, splits_path)

    train_problems = {k for k, s in assignment.items() if s == "train"}
    val_problems = {k for k, s in assignment.items() if s == "val"}

    pair_rng = random.Random(pw.pair_seed)
    train_path = _resolve(pw.pairs_train_jsonl)
    val_path = _resolve(pw.pairs_val_jsonl)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    print(f"\n[pairs] source        : {source_path}  ({len(rows)} rows)")
    print(f"[pairs] pair_mode     : {pw.pair_mode}")
    print(f"[pairs] max_neg/pos   : {pw.max_negatives_per_positive}")
    print(f"[pairs] problems      : train {len(train_problems)} / val {len(val_problems)}")

    stats = {}
    with open(train_path, "w") as f:
        stats["train"] = _emit_pairs_for_split(
            by_problem, train_problems, pw.pair_mode, pw.max_negatives_per_positive, pair_rng, f
        )
    with open(val_path, "w") as f:
        stats["val"] = _emit_pairs_for_split(
            by_problem, val_problems, pw.pair_mode, pw.max_negatives_per_positive, pair_rng, f
        )

    print("\n" + "=" * 60)
    for split, (n_pairs, n_pos_used, n_pos_no_pair) in stats.items():
        path = train_path if split == "train" else val_path
        print(f"  {split:5s}: {n_pairs} pairs  "
              f"({n_pos_used} positives paired, {n_pos_no_pair} positives with no eligible negative)")
        print(f"         -> {path}")
    print(f"  splits -> {splits_path}")
    print("=" * 60)

    if stats["train"][0] == 0:
        print("[ERROR] 0 training pairs written. Check pair_mode / dataset labels.")
        sys.exit(1)
    return train_path, val_path, splits_path


def main() -> None:
    cfg = load_config()
    build_pairs(cfg)


if __name__ == "__main__":
    main()
