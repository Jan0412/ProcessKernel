"""Problem-level split over the built parts, joint across runs (PLAN §6).

    python -m reranker.src.prm.splits --config reranker/configs/prm_config.yaml
"""

from __future__ import annotations

import glob
import json
import os
import random
from collections import Counter
from collections.abc import Iterable

from reranker.src.config import RerankerConfig, _resolve, load_config
from reranker.src.data.splits import _split_ids
from reranker.src.prm.build import PARTS, write_atomic

SPLITS = "splits.json"


def problem_keys(parts_dir: str) -> list[tuple[int, int]]:
    """Every ``(level, problem_id)`` in the parts, sorted -- ``_split_ids`` shuffles from here."""
    keys = set()
    for part in sorted(glob.glob(os.path.join(parts_dir, "*.jsonl"))):
        with open(part) as f:
            for line in f:
                row = json.loads(line)
                keys.add((row["level"], row["problem_id"]))
    return sorted(keys)


def build_splits(cfg: RerankerConfig) -> str:
    """Assign each problem to train/val/test and write ``out_dir/splits.json``."""
    prm = cfg.prm
    prm.validate()

    out_dir = _resolve(prm.out_dir)
    parts_dir = os.path.join(out_dir, PARTS)
    # Over the union of ALL parts: every DeepSeek problem id also appears in gpt-oss, so a
    # per-run split would put the same problem in one run's train and the other's val.
    keys = problem_keys(parts_dir)
    if not keys:
        raise FileNotFoundError(f"no rows under {parts_dir}; run reranker.src.prm.build first")

    assignment = _split_ids(keys, prm.split_ratios, random.Random(prm.split_seed))
    path = os.path.join(out_dir, SPLITS)
    # Atomically, like every other artifact here: a truncated splits.json still parses,
    # just with problems missing from every split.
    serial = {f"{lvl}:{pid}": s for (lvl, pid), s in assignment.items()}
    write_atomic(path, json.dumps(serial, indent=2))

    counts = Counter(assignment.values())
    print(f"Splits written: {path}")
    print(f"  problems: {dict(sorted(counts.items()))} (total {len(assignment)})")
    return path


def main(argv: Iterable[str] | None = None) -> None:
    build_splits(load_config(None if argv is None else list(argv)))


if __name__ == "__main__":
    main()
