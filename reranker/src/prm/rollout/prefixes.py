"""v1's parts -> the cut prefixes a v2 campaign measures: job A (PLAN_v2 §6).

    python -m reranker.src.prm.rollout.prefixes --config reranker/configs/prm_rollout.yaml

Reads v1's parts and its splits; writes ``prefixes.jsonl``. Nothing here generates text or
touches a GPU -- job A only decides *where to cut* and *which cuts belong in one list*.
"""

from __future__ import annotations

import bisect
import dataclasses
import glob
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from reranker.src.config import (
    CUT,
    RANDOM,
    PRMRolloutConfig,
    RerankerConfig,
    _resolve,
    load_config,
)
from reranker.src.data.splits import load_splits
from reranker.src.prm.build import MANIFEST, _git, _sha1, write_atomic

TRAIN, VAL = "train", "val"
PREFIXES = "prefixes.jsonl"
# What job A reads off a v1 row. `prompt` and `raw` are deliberately absent: they are the
# bulk of a row, job A needs neither, and job B resolves them from the same parts by
# prefix_id -- so a part costs its cuts in memory rather than its corpus.
ROW_FIELDS = (
    "run_name", "shard", "round", "level", "problem_id", "sample_id", "stem",
    "cuts", "cut_kinds", "cut_index",
)


@dataclass(frozen=True)
class Prefix:
    """One row of ``prefixes.jsonl`` (PLAN_v2 §5).

    ``prompt``, ``system_prompt`` and ``raw`` are deliberately absent: they are frozen in
    v1's parts and ``prefix_id`` is the lookup, so job B resolves them once per unit rather
    than every campaign artifact carrying a copy of the corpus.
    """

    prefix_id: str
    source: str
    run_name: str
    run_tag: str
    shard: str
    round: int
    level: int
    problem_id: int
    sample_id: int
    stem: str

    cut_char: int      # the prefix is raw[:cut_char]
    cut_index: int     # chunk depth -- the list key component (N4)
    cut_kind: str      # prose | code -- selects job B's continuation mode
    # ANALYSIS METADATA ONLY, never a model input: at inference the generation is unfinished
    # and the total does not exist. The names are verbose so a reader cannot mistake them
    # for features (same rule as v1 §6).
    n_cuts_total: int
    rel_depth: float

    list_key: str
    split: str
    selection: str
    selection_score: float | None

    K: int
    min_rollouts: int
    beam_text: str | None = None

    def __post_init__(self) -> None:
        # N3, in the type rather than in the caller or in config validation: unbiased val
        # is the invariant the whole comparison rests on, so no code path -- present or
        # future, config-driven or not -- may be able to construct a scored val prefix.
        if self.split == VAL and (self.selection != RANDOM or self.selection_score is not None):
            raise ValueError(
                f"N3: a val prefix is selected at random and carries no score, but "
                f"{self.prefix_id} has selection={self.selection!r} and "
                f"selection_score={self.selection_score!r}"
            )


def n_cuts_total(row: dict) -> int:
    """The completion's own cut count: the last surviving index, plus one.

    Not ``len(row["cuts"])``. prm.min_frac drops *leading* cuts, so what a part holds is a
    suffix of the completion's cuts and counting it would shrink the ruler the deepest cut
    was measured against -- inflating rel_depth for exactly the rows that were trimmed.
    """
    return row["cut_index"][-1] + 1


def in_window(prefix: Prefix, cfg: PRMRolloutConfig) -> bool:
    """Is this cut inside the learnable band? Below it every prefix of a problem shares the
    base rate; above it the outcome is already decided and the cut decides nothing."""
    return cfg.min_rel_depth <= prefix.rel_depth <= cfg.max_rel_depth


def depth_indices(group: list[dict], cfg: PRMRolloutConfig, rng: random.Random) -> list[int]:
    """The cut depths this group is cut at -- one shared ``k`` per list, so N4 holds.

    Stratified over *rel_depth*, not over ``k``: completions differ in length by 3-4x, so
    uniform-over-k would over-sample the shallow region of the long ones. One draw per
    stratum rather than ``depths_per_group`` free draws, so the aggregate histogram comes
    out flat instead of clumping -- the report slices by depth bucket and an empty bucket
    is a metric it cannot compute.

    The ruler is the group's median length. Siblings differ, so a shared ``k`` lands at
    slightly different rel_depths for each of them; that is the raggedness §6 accepts, and
    ``stratify`` is what drops the ones it carries out of the window.
    """
    n_ref = statistics.median_low([n_cuts_total(r) for r in group])
    # Clamped to depths whose rel_depth is inside the window *by construction*: rounding a
    # drawn t can land just outside it, and every such cut would be filtered right back out
    # -- a list paid for in enumeration and dropped before it is ever emitted.
    lo = max(1, math.ceil(cfg.min_rel_depth * n_ref))
    hi = min(n_ref - 1, math.floor(cfg.max_rel_depth * n_ref))
    if lo > hi:
        return []  # too short to hold a cut inside the window at all
    span = (cfg.max_rel_depth - cfg.min_rel_depth) / cfg.depths_per_group
    # A set: two strata can round onto one k in a short completion, and emitting that k
    # twice would be two lists under one list_key.
    return sorted(
        {
            min(hi, max(lo, round((cfg.min_rel_depth + (j + rng.random()) * span) * n_ref)))
            for j in range(cfg.depths_per_group)
        }
    )


def stratify(
    candidates: list[Prefix], cfg: PRMRolloutConfig, rng: random.Random
) -> list[Prefix]:
    """One list's candidates, bounded at both ends (§6): filter to the window, then cap.

    In that order, never the other. Capping first would let an out-of-window candidate take
    one of the ``max_list_size`` slots and then be filtered out of it, so a budget knob
    would be deciding which depths survive -- which is a correctness question.

    The subsample is random and seeded in *both* splits: N3 forbids a score in val, scored
    selection is deferred anyway, and V̂ is unknowable before it has been paid for, so no
    better rule is available. What is not drawn stays on v1's parts, so widening the cap
    later is a top-up rather than a fresh campaign.
    """
    kept = [p for p in candidates if in_window(p, cfg)]
    if len(kept) < cfg.min_list_size:
        return []  # one candidate forms no pair, and would buy nothing for its K evals
    if len(kept) > cfg.max_list_size:
        kept = sorted(rng.sample(kept, cfg.max_list_size), key=lambda p: p.sample_id)
    return kept


def enumerate_cut_prefixes(
    rows: Iterable[dict],
    cfg: PRMRolloutConfig,
    splits: dict[tuple[int, int], str],
    counts: Counter | None = None,
) -> Iterator[Prefix]:
    """Every prefix ``rows`` yields, grouped into lists by (run, level, problem, round).

    ``rows`` must contain whole groups. One v1 part is exactly that -- run and round are in
    its file name and a shard holds every sample of its problems -- which is what lets a
    campaign stream the corpus a part at a time instead of holding it.
    """
    counts = Counter() if counts is None else counts
    if cfg.source != CUT:
        raise NotImplementedError(
            f"source={cfg.source!r}: beam prefixes are stage 2 and are not built (PLAN_v2 §11)"
        )
    for key, group in _groups(rows, cfg, counts).items():
        yield from _group_prefixes(key, group, cfg, splits, counts)


def group_key(row: dict) -> tuple:
    """The list a row can belong to: one run, one level, one problem, one round (N2)."""
    return (row["run_name"], row["level"], row["problem_id"], row["round"])


def _groups(
    rows: Iterable[dict], cfg: PRMRolloutConfig, counts: Counter
) -> dict[tuple, list[dict]]:
    """Rows the campaign is about, bucketed by list group and ordered by sample_id."""
    out: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        counts["rows_read"] += 1
        # run_tags is the run *filter* as well as the tag: one parts dir holds every run v1
        # built, and a campaign takes one of them.
        if row["run_name"] not in cfg.run_tags:
            counts["row_run_not_selected"] += 1
            continue
        if row["round"] not in cfg.rounds:
            counts["row_round_not_selected"] += 1
            continue
        out[group_key(row)].append(row)
    for group in out.values():
        group.sort(key=lambda r: r["sample_id"])
    return out


def _group_prefixes(
    key: tuple,
    group: list[dict],
    cfg: PRMRolloutConfig,
    splits: dict[tuple[int, int], str],
    counts: Counter,
) -> Iterator[Prefix]:
    run_name, level, problem_id, rnd = key
    tag = cfg.run_tags[run_name]
    counts["groups"] += 1
    if (level, problem_id) not in splits:
        raise KeyError(
            f"{level}:{problem_id} has no split in v1's splits.json. v2 reads the split and "
            "never recomputes one (N3), so there is nothing to fall back to -- rerun "
            "reranker.src.prm.splits over the parts this campaign reads"
        )
    split = splits[(level, problem_id)]
    if split not in (TRAIN, VAL):
        # v1 holds back a test split. A v2 campaign that trained on it would be scoring
        # itself, and the held-out set would stop meaning anything for either version.
        counts[f"group_split_{split}"] += 1
        return
    # N3 short-circuits HERE, before any scorer is reached: val is random whatever the
    # config says, so a deferred mode can never raise on the split it may not touch anyway.
    selection = RANDOM if split == VAL else cfg.train_selection
    if selection != RANDOM:
        raise NotImplementedError(
            f"train_selection={selection!r} is deferred, not built (PLAN_v2 §6, decided "
            "2026-08-12): a random campaign has to run first, or there is no arm to "
            "compare a scored one against"
        )

    rng = _rng(cfg.select_seed, tag, level, problem_id, rnd)
    for k in depth_indices(group, cfg, rng):
        list_key = f"{tag}:{level}:{problem_id}:{rnd}:{k}"
        counts["lists_considered"] += 1
        candidates = [
            p
            for p in (_cut_prefix(row, k, tag, list_key, split, selection, cfg) for row in group)
            if p is not None
        ]
        counts["cand_out_of_window"] += sum(1 for p in candidates if not in_window(p, cfg))
        # Seeded off the list, not off the group's stream: a campaign resumed, resharded or
        # run at a different num_workers must draw the same samples.
        kept = stratify(candidates, cfg, _rng(cfg.select_seed, list_key))
        if not kept:
            counts["list_too_narrow"] += 1
            continue
        counts["lists"] += 1
        counts["lists_at_cap"] += len(kept) == cfg.max_list_size
        counts[f"prefixes_{split}"] += len(kept)
        yield from kept


def _cut_prefix(
    row: dict,
    k: int,
    tag: str,
    list_key: str,
    split: str,
    selection: str,
    cfg: PRMRolloutConfig,
) -> Prefix | None:
    """This sample's prefix at depth ``k``, or ``None`` when it has no cut that deep."""
    idx = row["cut_index"]
    pos = bisect.bisect_left(idx, k)
    if pos == len(idx) or idx[pos] != k:
        return None  # ragged by design: a short completion simply is not in this list
    total = n_cuts_total(row)
    return Prefix(
        # The tag carries the level by convention (`ds-l2`), which is why the id does not.
        # build_prefixes checks the ids are unique, so a run spanning two levels fails loudly.
        prefix_id=f"{tag}__{row['shard'].replace('_', '')}__r{row['round']}"
        f"__p{row['problem_id']}__s{row['sample_id']}__k{k:03d}",
        source=CUT,
        run_name=row["run_name"],
        run_tag=tag,
        shard=row["shard"],
        round=row["round"],
        level=row["level"],
        problem_id=row["problem_id"],
        sample_id=row["sample_id"],
        stem=row["stem"],
        cut_char=row["cuts"][pos],
        cut_index=k,
        cut_kind=row["cut_kinds"][pos],
        n_cuts_total=total,
        rel_depth=k / total,
        list_key=list_key,
        split=split,
        selection=selection,
        selection_score=None,  # random selection scores nothing, in either split
        K=cfg.K,
        min_rollouts=cfg.min_rollouts,
    )


def _rng(seed: int, *parts) -> random.Random:
    """Seeded off the identity of what is being drawn, never off iteration order.

    One stream advanced per group would make every draw depend on how many groups came
    before it, so a resumed, resharded or reordered campaign would cut somewhere else.
    """
    return random.Random(f"{seed}:" + ":".join(str(p) for p in parts))


# --- job A: v1's parts on disk -> prefixes.jsonl ---------------------------------------


def build_prefixes(cfg: RerankerConfig) -> str:
    """Enumerate every prefix this campaign will measure; returns the ``prefixes.jsonl`` path.

    One part at a time, because a part holds whole groups (checked, not assumed). Read-only
    over v1: nothing here writes into the corpus it reads.
    """
    rollout = cfg.prm_rollout
    rollout.validate()

    pattern = _resolve(rollout.parts_glob)
    parts = sorted(glob.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"no v1 parts match {pattern}; run reranker.src.prm.build first")
    splits = load_splits(_resolve(rollout.splits_json))

    counts: Counter = Counter()
    out: list[Prefix] = []
    home: dict[tuple, str] = {}            # group -> the one part that holds it
    levels: dict[str, set[int]] = defaultdict(set)
    for part in parts:
        rows = list(_read_part(part))
        _check_layout(rows, part, home, levels, rollout)
        out.extend(enumerate_cut_prefixes(rows, rollout, splits, counts))

    # A tag naming a run no row carries selects nothing, and the campaign would otherwise
    # report an empty dataset that looks entirely valid.
    silent = sorted(set(rollout.run_tags) - set(levels))
    if silent:
        raise ValueError(
            f"prm_rollout.run_tags names {silent}, which no row under {pattern} carries -- "
            "a misspelled run name selects nothing and builds an empty campaign"
        )

    out_dir = _resolve(rollout.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, PREFIXES)
    manifest = _manifest(rollout, parts, out, counts)
    write_atomic(path, "".join(json.dumps(dataclasses.asdict(p)) + "\n" for p in out))
    write_atomic(os.path.join(out_dir, MANIFEST), json.dumps(manifest, indent=2))

    print(f"Prefixes: {manifest['prefixes']} in {manifest['lists']} lists -> {path}")
    print(f"  splits {manifest['splits']}   widths {manifest['list_width_histogram']}")
    print(f"  rel_depth {manifest['rel_depth_histogram']}")
    print(f"  counts {dict(sorted(counts.items()))}")
    return path


def _read_part(path: str) -> Iterator[dict]:
    """One part's rows, stripped to ``ROW_FIELDS`` as they are parsed.

    Strict, and named. A tolerant ``.get`` would turn a renamed field into ``None`` and
    surface three steps later as an empty campaign; a bare ``KeyError`` names the field but
    not which of the parts carries it, and a campaign reads 144 of them.
    """
    with open(path) as f:
        for n, line in enumerate(f, 1):
            row = json.loads(line)
            try:
                lean = {k: row[k] for k in ROW_FIELDS}
            except KeyError as missing:
                raise ValueError(
                    f"{os.path.basename(path)} line {n} carries no {missing.args[0]!r}. "
                    f"Job A reads {list(ROW_FIELDS)} off every v1 row, so these parts were "
                    "not written by this reranker.src.prm.build -- point parts_glob at a "
                    "build that was, or rebuild this one"
                ) from missing
            yield lean


def _check_layout(
    rows: list[dict],
    part: str,
    home: dict[tuple, str],
    levels: dict[str, set[int]],
    cfg: PRMRolloutConfig,
) -> None:
    """The two layout facts the campaign relies on, checked against the corpus itself."""
    for row in rows:
        run = row["run_name"]
        # Only the runs this campaign reads: how another run is laid out is not its problem.
        if run not in cfg.run_tags:
            continue
        levels[run].add(row["level"])
        key = group_key(row)
        if home.setdefault(key, part) != part:
            raise ValueError(
                f"group {key} is split across two parts ({os.path.basename(home[key])} and "
                f"{os.path.basename(part)}): each half would be cut against its own median "
                "and emit its own lists under one list_key, which is N2"
            )
    spanning = {r: sorted(v) for r, v in levels.items() if len(v) > 1}
    if spanning:
        raise ValueError(
            f"{spanning} carry several levels under one tag. prefix_id omits the level "
            "because the tag carries it by convention (`ds-l2`), so two levels under one "
            "tag collide on it -- give each (run, level) its own run_tag"
        )


def _manifest(rollout: PRMRolloutConfig, parts: list[str], out: list[Prefix], counts: Counter):
    """The campaign as it stands on disk, and everything needed to trace it back."""
    # Beside the parts dir, so a campaign records which v1 build it read and two campaigns
    # over different corpora cannot be mistaken for one.
    v1 = os.path.join(os.path.dirname(os.path.dirname(_resolve(rollout.parts_glob))), MANIFEST)
    baseline = _resolve(rollout.baseline_timing_json)
    widths = Counter(Counter(p.list_key for p in out).values())
    dirty = _git("status", "--porcelain")
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "config": dataclasses.asdict(rollout),
        "parts": len(parts),
        "v1_manifest": v1,
        "v1_manifest_sha1": _sha1(v1),
        "baseline_timing_json": baseline,
        "baseline_sha1": _sha1(baseline),
        "prefixes": len(out),
        "lists": counts["lists"],
        "splits": {s: counts[f"prefixes_{s}"] for s in (TRAIN, VAL)},
        "list_width_histogram": {str(w): n for w, n in sorted(widths.items())},
        "rel_depth_histogram": depth_histogram(out, rollout),
        "counts": dict(sorted(counts.items())),
    }


def depth_histogram(out: Iterable[Prefix], cfg: PRMRolloutConfig) -> dict[str, int]:
    """Prefixes per depth band -- the slice §6 requires every headline number to be reported in.

    An empty band here is a report that cannot be computed, not a curiosity: shallow
    prefixes have tiny true gaps and maximal estimator noise, deep ones the reverse, and a
    single averaged metric hides both.
    """
    lo, hi, n = cfg.min_rel_depth, cfg.max_rel_depth, cfg.depth_buckets
    edges = [lo + (hi - lo) * i / n for i in range(n + 1)]
    hist: Counter = Counter()
    for p in out:
        hist[min(n - 1, max(0, int((p.rel_depth - lo) / (hi - lo) * n)))] += 1
    return {f"{edges[i]:.2f}-{edges[i + 1]:.2f}": hist[i] for i in range(n)}


def main(argv: Iterable[str] | None = None) -> None:
    build_prefixes(load_config(None if argv is None else list(argv)))


if __name__ == "__main__":
    main()
