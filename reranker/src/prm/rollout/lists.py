"""Measured prefixes -> the ranked lists a trainer reads: job D's second half (PLAN_v2 §6).

``rel = 2 * v_graded`` is the whole conversion, and it is exact rather than a rescaling: a
rollout's graded target is 0 for a failing kernel and ``(1 + speed_p(...))/2`` for a correct
one, so twice the mean lands on precisely the ladder listwise/lists.py:145 writes inline --
0.0 a total failure, [1.0, 2.0] correct and graded by speed. `lambdarank_loss`'s
``gain = 2^rel - 1`` therefore sees the range it sees today, and nothing under
``reranker/src/listwise/`` is touched to achieve it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from reranker.src.config import _resolve, load_config
from reranker.src.data.splits import load_splits
from reranker.src.prm.build import write_atomic
from reranker.src.prm.rollout import prefixes, stage, values
from reranker.src.prm.rollout.prefixes import TRAIN, VAL

LISTS = "lists_{split}.jsonl"
LISTS_MANIFEST = "lists_manifest.json"

TOO_SMALL, ALL_EQUAL = "too_small", "all_equal"
# A problem v1 held back for test: v2 writes train and val only, so it is counted out here
# rather than silently landing in one of them.
OTHER_SPLIT = "other_split"
LEDGER = (TOO_SMALL, ALL_EQUAL, OTHER_SPLIT)


@dataclass(frozen=True)
class Item:
    """One candidate in a ranked list.

    ``se`` is the error bar on V̂ and stays on V̂'s scale, *not* doubled with ``rel``: the
    confidence weight of §5 is ``|V̂i - V̂j| / sqrt(SEi^2 + SEj^2)``, a ratio of differences
    to error bars measured the same way. Doubling one side and not the other would inflate
    every weight by 2, saturating the clip and handing noisy pairs full weight.
    """

    prefix_id: str
    rel: float
    n_rollouts: int
    se: float


@dataclass(frozen=True)
class ListRow:
    """One row of ``lists_{train,val}.jsonl`` (PLAN_v2 §5)."""

    list_key: str
    run_tag: str
    level: int
    problem_id: int
    round: int
    cut_index: int
    rel_depth_mean: float
    split: str
    source: str
    items: list


def rel(value) -> float:
    """V̂ -> listwise relevance. One affine line, exactly invertible, pinned by a test."""
    return 2.0 * value.v_graded


def group(prefix_rows, value_rows) -> dict[str, list[tuple]]:
    """``list_key -> [(prefix, value), ...]``, ordered by prefix id.

    Driven by the *values*, not by the prefixes: a prefix the campaign could not measure is
    simply absent, which shortens its list rather than invalidating it. `lambdarank_loss`
    already handles ragged groups through `group_sizes`.
    """
    by_id = {p.prefix_id: p for p in prefix_rows}
    out: dict[str, list[tuple]] = defaultdict(list)
    for value in value_rows:
        if value.prefix_id not in by_id:
            raise KeyError(
                f"{value.prefix_id} has a value but no prefix: {values.VALUES} and "
                f"{prefixes.PREFIXES} come from different builds, and the list key, split "
                "and depth this row belongs in are all on the prefix"
            )
        prefix = by_id[value.prefix_id]
        out[prefix.list_key].append((prefix, value))
    return {key: sorted(pairs, key=lambda pv: pv[0].prefix_id) for key, pairs in out.items()}


def build_list(key: str, members: list[tuple], split: str, cfg, counts) -> ListRow | None:
    """One list, or ``None`` for one that cannot be ranked -- counted either way."""
    if len(members) < cfg.min_list_size:
        # Sized first so the ledger adds up: a one-item list is also trivially all-equal,
        # and counting it under both reasons would report more lists than were built.
        counts[TOO_SMALL] += 1
        return None
    members = sorted(members, key=lambda pv: pv[0].prefix_id)
    _check_invariants(key, [p for p, _ in members])

    items = [Item(p.prefix_id, rel(v), v.n_rollouts, v.se_graded) for p, v in members]
    if len({item.rel for item in items}) < 2:
        # No valid ranking pair. The same drop listwise/lists.py:163 already makes, restated
        # here rather than imported, because importing it would mean editing that file.
        counts[ALL_EQUAL] += 1
        return None

    first = members[0][0]
    return ListRow(
        list_key=key,
        run_tag=first.run_tag,
        level=first.level,
        problem_id=first.problem_id,
        round=first.round,
        cut_index=first.cut_index,
        rel_depth_mean=sum(p.rel_depth for p, _ in members) / len(members),
        split=split,
        source=first.source,
        items=items,
    )


def _check_invariants(key: str, members: list) -> None:
    """N2 and N4, raised rather than warned -- neither is visible downstream once written."""
    first = members[0]
    for p in members[1:]:
        if (p.run_tag, p.level, p.problem_id, p.round) != (
            first.run_tag, first.level, first.problem_id, first.round
        ):
            raise ValueError(
                f"N2: list {key} holds {first.prefix_id} from "
                f"({first.run_tag}, {first.level}, {first.problem_id}, round {first.round}) "
                f"and {p.prefix_id} from ({p.run_tag}, {p.level}, {p.problem_id}, round "
                f"{p.round}) -- a list spanning either ranks conditioning or policy rather "
                "than prefix quality"
            )
        if p.cut_index != first.cut_index:
            raise ValueError(
                f"N4: list {key} cuts {first.prefix_id} at chunk {first.cut_index} and "
                f"{p.prefix_id} at {p.cut_index} -- equal chunk counts are what make the "
                "items comparable states of one problem"
            )


# --- the pass: prefixes + values -> lists_train.jsonl / lists_val.jsonl -------------------


def build_lists(cfg) -> dict:
    """Group a scored campaign into its lists. CPU only, and it runs beside values.py."""
    rollout_cfg = cfg.prm_rollout
    rollout_cfg.validate()
    out_dir = _resolve(rollout_cfg.out_dir)

    prefix_rows = stage.read_prefixes(os.path.join(out_dir, prefixes.PREFIXES))
    value_rows = read_values(os.path.join(out_dir, values.VALUES))
    # v1's, read and never recomputed: recomputing would reshuffle problems between the two
    # label sets, putting a v2 val problem in v1's train set with nothing able to see it.
    splits = load_splits(_resolve(rollout_cfg.splits_json))

    counts: Counter = Counter()
    built: dict[str, list[ListRow]] = {TRAIN: [], VAL: []}
    for key, members in sorted(group(prefix_rows, value_rows).items()):
        first = members[0][0]
        if (first.level, first.problem_id) not in splits:
            raise KeyError(
                f"{first.level}:{first.problem_id} has no split in "
                f"{_resolve(rollout_cfg.splits_json)} -- v2 reads v1's splits and never "
                "recomputes them, so a problem missing from the file is an error rather "
                "than a fresh draw"
            )
        split = splits[first.level, first.problem_id]
        if split not in (TRAIN, VAL):
            counts[OTHER_SPLIT] += 1
            continue
        row = build_list(key, members, split, rollout_cfg, counts)
        if row is not None:
            built[split].append(row)

    for split, rows in built.items():
        write_atomic(
            os.path.join(out_dir, LISTS.format(split=split)),
            "".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in rows),
        )

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": dataclasses.asdict(rollout_cfg),
        "prefixes": len(prefix_rows),
        "values": len(value_rows),
        "lists": {split: len(rows) for split, rows in built.items()},
        "items": {split: sum(len(r.items) for r in rows) for split, rows in built.items()},
        # A corpus that is mostly pairs is a pairwise dataset wearing a listwise schema, and
        # should be recognised as such rather than discovered during training (§2).
        "two_item_lists": {
            split: sum(1 for r in rows if len(r.items) == 2) for split, rows in built.items()
        },
        "dropped": {reason: counts[reason] for reason in LEDGER},
    }
    write_atomic(os.path.join(out_dir, LISTS_MANIFEST), json.dumps(manifest, indent=2))
    return manifest


def read_values(path: str) -> list:
    with open(path) as f:
        return [values.Value(**json.loads(line)) for line in f]


def read_lists(path: str) -> list[ListRow]:
    """``lists_{split}.jsonl`` -> rows, items rebuilt as `Item` rather than dicts.

    Here beside the writer rather than in a reader: job E and the stats report both read this
    file back, and neither should own the shape the other parses it into.
    """
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return [ListRow(**{**r, "items": [Item(**i) for i in r["items"]]}) for r in rows]


def main(argv=None) -> None:
    manifest = build_lists(load_config(None if argv is None else list(argv)))
    fired = {r: n for r, n in manifest["dropped"].items() if n}
    for split in (TRAIN, VAL):
        print(
            f"{split}: {manifest['lists'][split]} lists, {manifest['items'][split]} items "
            f"({manifest['two_item_lists'][split]} of them pairs)"
        )
    print(f"  dropped {fired or 'nothing'}")


if __name__ == "__main__":
    main()
