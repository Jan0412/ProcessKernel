"""Problem-level train/val/test split.

All samples of a given problem (across every run/model) go to the same split, so
the reranker is never evaluated on a problem it saw during training. Optionally
stratified by KernelBench level so each split covers all levels.

Usage (standalone):
    python -m reranker.data.splits --config configs/default.yaml
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from reranker.src.config import RerankerConfig, _resolve, load_config


def _split_ids(ids: list, ratios: list[float], rng: random.Random) -> dict:
    """Assign a shuffled list of ids to train/val/test by ratio."""
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    # test gets the remainder so the counts always sum to n
    assignment = {}
    for i, _id in enumerate(ids):
        if i < n_train:
            assignment[_id] = "train"
        elif i < n_train + n_val:
            assignment[_id] = "val"
        else:
            assignment[_id] = "test"
    return assignment


def build_splits(cfg: RerankerConfig) -> str:
    """Read the dataset JSONL, assign each problem to a split, write splits.json."""
    data_cfg = cfg.data
    dataset_path = _resolve(data_cfg.dataset_jsonl)
    splits_path = _resolve(data_cfg.splits_json)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run reranker.data.build_dataset first."
        )

    # Collect the unique problem keys and their level.
    # A problem key is (level, problem_id) — ids are only unique within a level.
    problem_level: dict[tuple, int] = {}
    with open(dataset_path) as f:
        for line in f:
            row = json.loads(line)
            key = (row["level"], row["problem_id"])
            problem_level[key] = row["level"]

    rng = random.Random(data_cfg.split_seed)
    ratios = data_cfg.split_ratios
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {ratios}")

    assignment: dict[tuple, str] = {}
    if data_cfg.stratify_by_level:
        by_level: dict[int, list] = defaultdict(list)
        for key, level in problem_level.items():
            by_level[level].append(key)
        for level, keys in sorted(by_level.items()):
            assignment.update(_split_ids(keys, ratios, rng))
    else:
        assignment.update(_split_ids(list(problem_level), ratios, rng))

    # Serialize with string keys "level:problem_id" (JSON can't key on tuples).
    serializable = {f"{lvl}:{pid}": split for (lvl, pid), split in assignment.items()}
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(serializable, f, indent=2)

    counts = {"train": 0, "val": 0, "test": 0}
    for split in assignment.values():
        counts[split] += 1
    print(f"Splits written: {splits_path}")
    print(f"  problems: {counts} (total {len(assignment)})")
    return splits_path


def load_splits(splits_path: str) -> dict[tuple, str]:
    """Load splits.json back into {(level, problem_id): split}."""
    with open(splits_path) as f:
        raw = json.load(f)
    out = {}
    for key, split in raw.items():
        lvl, pid = key.split(":")
        out[(int(lvl), int(pid))] = split
    return out


def main() -> None:
    cfg = load_config()
    build_splits(cfg)


if __name__ == "__main__":
    main()
