"""Build the pairwise training/validation dataset from the labeled source dataset.

Reads the same labeled rows the pointwise pipeline uses (``data.dataset_jsonl``),
then:

  1. Builds a *fresh* problem-level train/val split (no test) — all candidates of
     a problem stay in one split, so the reranker is never validated on a problem
     it trained on. Optionally stratified by KernelBench level.
  2. For each problem, pools its candidates across all runs, optionally dedups
     duplicate kernel sources (``dedup_by_code_hash``), and forms
     (positive, negative) pairs per ``pair_mode``:
       compiled_wrong : negative = compiled but numerically incorrect (hard).
       all_negative   : negative = anything with label 0 (incl. compile failures).
       speed          : graded fast_p pairs (see ``_emit_graded_pairs_for_split``).
     ``max_negatives_per_positive`` caps partners per anchor and
     ``max_pairs_per_problem`` caps each problem's total pairs, so candidate-rich
     problems do not dominate the epoch.

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
from reranker.src.data.labels import code_hash, speed_p


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


def _dedup_rows(rows: list[dict]) -> list[dict]:
    """Keep the first occurrence of each distinct kernel source (mirrors listwise)."""
    seen: set[str] = set()
    kept: list[dict] = []
    for r in rows:
        h = code_hash(r["kernel_src"])
        if h in seen:
            continue
        seen.add(h)
        kept.append(r)
    return kept


def _problem_rows(by_problem: dict[tuple, list[dict]], key: tuple, pw) -> list[dict]:
    rows = by_problem.get(key, [])
    return _dedup_rows(rows) if pw.dedup_by_code_hash else rows


def _write_problem_pairs(pairs: list[dict], cap: int | None, rng: random.Random, out_f) -> int:
    """Cap one problem's total pairs (uniform subsample), write them, return the count."""
    if cap is not None and len(pairs) > cap:
        pairs = rng.sample(pairs, cap)
    for p in pairs:
        out_f.write(json.dumps(p) + "\n")
    return len(pairs)


def _emit_pairs_for_split(
    by_problem: dict[tuple, list[dict]],
    split_problems: set[tuple],
    pw,
    rng: random.Random,
    out_f,
) -> tuple[int, int, int]:
    """Write all pairs for the problems in `split_problems`.

    Returns (n_pairs, n_positives_used, n_positives_without_pair).
    """
    n_pairs = 0
    n_pos_used = 0
    n_pos_no_pair = 0
    max_neg = pw.max_negatives_per_positive
    for key in split_problems:
        rows = _problem_rows(by_problem, key, pw)
        positives = [r for r in rows if r["label"] == 1]
        negatives = _eligible_negatives(rows, pw.pair_mode)
        if not positives:
            continue
        if not negatives:
            n_pos_no_pair += len(positives)
            continue
        level, problem_id = key
        problem_pairs: list[dict] = []
        for pos in positives:
            negs = negatives
            if max_neg is not None and len(negs) > max_neg:
                negs = rng.sample(negs, max_neg)
            n_pos_used += 1
            for neg in negs:
                problem_pairs.append({
                    "level": level,
                    "problem_id": problem_id,
                    "pair_mode": pw.pair_mode,
                    "pos": _ref(pos),
                    "neg": _ref(neg),
                    "weight": 1.0,
                    "group": "correctness",  # neg is always wrong in these modes
                })
        n_pairs += _write_problem_pairs(problem_pairs, pw.max_pairs_per_problem, rng, out_f)
    return n_pairs, n_pos_used, n_pos_no_pair


def _graded_candidates(
    rows: list[dict], lo: float, hi: float, quant: float
) -> list[tuple[dict, float]]:
    """Compiling kernels with graded relevance: wrong -> 0, correct -> 1 + speed_p.

    Mirrors the listwise grading exactly (including the ``quant`` deadband, so
    timing-noise speedup differences collapse to the same grade). Correct kernels
    lacking a valid baseline speedup are dropped (never guessed); non-compiling
    kernels are excluded.
    """
    out: list[tuple[dict, float]] = []
    for r in rows:
        if not r.get("compiled"):
            continue
        if r["label"] == 1:
            su = r.get("speedup")
            if su is None or su <= 0:
                continue
            out.append((r, 1.0 + speed_p(su, lo, hi, quant)))
        else:  # compiled-but-wrong
            out.append((r, 0.0))
    return out


def _emit_graded_pairs_for_split(
    by_problem: dict[tuple, list[dict]],
    split_problems: set[tuple],
    pw,
    rng: random.Random,
    out_f,
) -> tuple[int, int, int]:
    """Write speed-graded (higher-rel, lower-rel) pairs for `split_problems`.

    Each candidate acts as an anchor (positive) against every strictly-lower-rel
    partner whose relevance gap is at least `min_rel_gap` (with `speed_quant` on,
    near-tie speedups grade equally and never pair). The gap is stored as the
    pair `weight` (used only when pairwise.weighted_loss is on). This subsumes both
    correct>wrong and fast-correct>slow-correct pairs. `max_pairs_per_problem`
    finally caps each problem's total (uniform subsample).

    Returns (n_pairs, n_anchors_used, n_problems_skipped).
    """
    n_pairs = n_anchors = n_skipped = 0
    max_neg = pw.max_negatives_per_positive
    for key in split_problems:
        cands = _graded_candidates(
            _problem_rows(by_problem, key, pw), pw.speedup_lo, pw.speedup_hi, pw.speed_quant
        )
        if len(cands) < 2:
            n_skipped += 1
            continue
        level, problem_id = key
        problem_pairs: list[dict] = []
        for hi_row, hi_rel in cands:
            partners = [
                (lo_row, hi_rel - lo_rel, lo_rel)
                for lo_row, lo_rel in cands
                if hi_rel - lo_rel > 0 and hi_rel - lo_rel >= pw.min_rel_gap
            ]
            if not partners:
                continue
            if max_neg is not None and len(partners) > max_neg:
                partners = rng.sample(partners, max_neg)
            n_anchors += 1
            for lo_row, gap, lo_rel in partners:
                # A pair is a *correctness* pair when the low side is wrong
                # (rel 0) and a *speed* pair when both sides are correct
                # (rel > 0). The trainer normalizes and balances the two groups
                # separately (group-split + alpha), like the listwise loss, so
                # the correctness offset can't swamp the smaller speed gaps.
                group = "correctness" if lo_rel == 0.0 else "speed"
                problem_pairs.append({
                    "level": level,
                    "problem_id": problem_id,
                    "pair_mode": "speed",
                    "pos": _ref(hi_row),
                    "neg": _ref(lo_row),
                    "weight": round(float(gap), 6),
                    "group": group,
                })
        emitted_here = _write_problem_pairs(problem_pairs, pw.max_pairs_per_problem, rng, out_f)
        n_pairs += emitted_here
        if emitted_here == 0:
            n_skipped += 1
    return n_pairs, n_anchors, n_skipped


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
    print(f"[pairs] pair_mode     : {pw.pair_mode}  (dedup_by_code_hash {pw.dedup_by_code_hash})")
    print(f"[pairs] caps          : max_neg/pos {pw.max_negatives_per_positive}  "
          f"max_pairs/problem {pw.max_pairs_per_problem}")
    if pw.pair_mode == "speed":
        print(f"[pairs] speedup map   : {pw.speedup_lo}x -> 0 .. {pw.speedup_hi}x -> 1  "
              f"(rel = 1 + p, quant {pw.speed_quant}); min_rel_gap {pw.min_rel_gap}")
    print(f"[pairs] problems      : train {len(train_problems)} / val {len(val_problems)}")

    def _emit(problems, f) -> tuple[int, int, int]:
        if pw.pair_mode == "speed":
            return _emit_graded_pairs_for_split(by_problem, problems, pw, pair_rng, f)
        return _emit_pairs_for_split(by_problem, problems, pw, pair_rng, f)

    stats = {}
    with open(train_path, "w") as f:
        stats["train"] = _emit(train_problems, f)
    with open(val_path, "w") as f:
        stats["val"] = _emit(val_problems, f)

    print("\n" + "=" * 60)
    for split, (n_pairs, n_used, n_no_pair) in stats.items():
        path = train_path if split == "train" else val_path
        if pw.pair_mode == "speed":
            detail = f"({n_used} anchors paired, {n_no_pair} problems skipped — too few / all-equal)"
        else:
            detail = f"({n_used} positives paired, {n_no_pair} positives with no eligible negative)"
        print(f"  {split:5s}: {n_pairs} pairs  {detail}")
        print(f"         -> {path}")
    print(f"  splits -> {splits_path}")
    print("=" * 60)

    if stats["train"][0] == 0:
        print("[ERROR] 0 training pairs written. Check pair_mode / speedup coverage / dataset labels.")
        sys.exit(1)
    return train_path, val_path, splits_path


def main() -> None:
    cfg = load_config()
    build_pairs(cfg)


if __name__ == "__main__":
    main()
