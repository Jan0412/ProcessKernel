"""The campaign report: is there signal in these lists? (PLAN_v2 §6, §12)

    python -m processkernel.prm.rollout.stats --config configs/prm_rollout.yaml

Read-only, CPU only, and torch-free by design -- it imports nothing that needs a CUDA venv,
because a diagnostics pass that only runs on a GPU node is one nobody runs. Exits non-zero
when a §12 acceptance check fails.
"""

from __future__ import annotations

import glob
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

from processkernel.config import RANDOM, _resolve, load_config
from processkernel.prm.data import build, corpus
from processkernel.prm.data.build import write_atomic
from processkernel.prm.rollout import calibrate, lists, prefixes, stage, values

STATS = "rollout_stats.json"


def spread(row: lists.ListRow) -> float:
    """Within-list V̂ spread: the campaign's headline number, on V̂'s scale not ``rel``'s.

    ``rel`` is ``2 * v_graded`` (lists.py), so the gap is halved back. §12 states the gate in
    V̂ units and the noise floor it is compared against is computed from `scores`, which are
    also V̂ units -- reporting this one in `rel` would double it against both.
    """
    rels = [item.rel for item in row.items]
    return (max(rels) - min(rels)) / 2.0


def null_spreads(
    rows, scores_by_id: dict[str, list[float]], rng: random.Random, replicates: int
) -> list[float]:
    """The spread §12 would see if every item of a list had the *same* true value.

    The gate "median within-list spread > 0.15" passes on pure noise: ``max - min`` is the
    range of `n` noisy draws and its floor climbs with list width even when nothing differs.
    So the floor is measured instead of assumed -- resample each item's own ``n_rollouts``
    draws from its **list's** pooled per-rollout scores, recompute V̂, take the range. No
    extra evals: the scores are already on `values.jsonl`.

    Pooling per list, not campaign-wide, is what makes this the right null: it keeps the
    list's base rate, its width and each item's rollout count, and varies only the thing
    under test. Where a list's items genuinely differ the pool is wider than any one item's
    truth, so the floor comes out slightly too high -- conservative, in the safe direction
    for a gate that decides whether to fund stage 2.

    Called with one row at a time by `beats_own_null`, which is the gate, and with all of
    them by `spread_report`, which prints the pooled distribution for description only. Do
    not gate on the pooled one: its q95 is set by the widest lists and the observed median by
    the narrowest, and comparing them came out negative on corpora with and without signal
    alike.
    """
    out: list[float] = []
    for row in rows:
        pool = [s for item in row.items for s in _scores(scores_by_id, item)]
        counts = [len(_scores(scores_by_id, item)) for item in row.items]
        for _ in range(replicates):
            drawn = [sum(rng.choices(pool, k=n)) / n for n in counts]
            out.append(max(drawn) - min(drawn))
    return out


def _scores(scores_by_id: dict[str, list[float]], item) -> list[float]:
    if item.prefix_id not in scores_by_id:
        raise KeyError(
            f"{item.prefix_id} is on a list but carries no per-rollout scores: "
            f"{lists.LISTS} and {values.VALUES} come from different builds, and the null "
            "this report compares the observed spread against cannot be bootstrapped "
            "without them"
        )
    return scores_by_id[item.prefix_id]


V_BINS = 10


def v_hat(item) -> float:
    """An item's measured value, back off ``rel``'s scale."""
    return item.rel / 2.0


def value_histogram(rows) -> dict[str, int]:
    """V̂ over ``[0, 1]`` in tenths -- the shape that says whether the campaign saturated.

    Every bin is emitted, zeros included: a corpus piled at 0.0 and one piled at 1.0 are
    both failures of level choice (§12's 15-70% solve rate), and only the empty middle
    tells them apart from a healthy spread.
    """
    hist: Counter = Counter()
    for row in rows:
        for item in row.items:
            # Closed at the top: V̂ = 1.0 is every rollout correct at full speed, a real
            # value, and floor() alone would file it in an eleventh bin.
            hist[min(int(v_hat(item) * V_BINS), V_BINS - 1)] += 1
    return {f"{i / V_BINS:.1f}-{(i + 1) / V_BINS:.1f}": hist[i] for i in range(V_BINS)}


def size_histogram(rows) -> dict[str, int]:
    """Lists per width. Ragged is expected (§2); a corpus of one width is not."""
    hist = Counter(len(row.items) for row in rows)
    return {str(w): hist[w] for w in sorted(hist)}


def two_item(rows) -> int:
    """Lists holding exactly a pair -- §2 wants these counted, not averaged into the median."""
    return sum(1 for row in rows if len(row.items) == 2)


@dataclass(frozen=True)
class Band:
    """One depth band and the lists that fell in it."""

    lo: float
    hi: float
    rows: list


def depth_bands(rows, cfg) -> list[Band]:
    """The campaign's lists cut into ``depth_buckets`` bands over the depth window (§6).

    Shallow prefixes have tiny true gaps and maximal estimator noise, deep ones the reverse,
    so a single averaged number hides both. The bands are `prefixes.bucket_edges`', the same
    ones job A histogrammed and job E reports in -- three definitions would drift.
    """
    edges = prefixes.bucket_edges(cfg)
    grouped: dict[int, list] = {i: [] for i in range(cfg.depth_buckets)}
    for row in rows:
        grouped[prefixes.bucket_of(row.rel_depth_mean, edges)].append(row)
    return [Band(edges[i], edges[i + 1], grouped[i]) for i in range(cfg.depth_buckets)]


def outside_window(rows, cfg) -> int:
    """Lists whose depth the window no longer covers -- `bucket_of` clamps them, this counts.

    Non-zero means the depth knobs moved between job A and this report, so the bands are
    describing a campaign that was enumerated against different ones.
    """
    return sum(
        1 for row in rows if not cfg.min_rel_depth <= row.rel_depth_mean <= cfg.max_rel_depth
    )


def slice_counts(prefix_rows, measured: set[str], list_rows, attr: str) -> dict[str, dict]:
    """The A->D funnel cut by one prefix attribute -- §6 asks for ``source``.

    Keyed off the prefixes rather than the survivors, so a slice that enumerated prefixes
    and measured none of them still appears with ``measured: 0``. A funnel built from what
    came out the far end cannot show where a campaign lost its rows.
    """
    out: dict[str, dict] = {}
    for prefix in prefix_rows:
        rec = out.setdefault(
            str(getattr(prefix, attr)), {"prefixes": 0, "measured": 0, "lists": 0, "items": 0}
        )
        rec["prefixes"] += 1
        rec["measured"] += prefix.prefix_id in measured
    for row in list_rows:
        # `setdefault` again: a list can only exist where a prefix did, but a mismatched pair
        # of files would otherwise raise a KeyError deep in a report rather than show itself.
        rec = out.setdefault(
            str(getattr(row, attr)), {"prefixes": 0, "measured": 0, "lists": 0, "items": 0}
        )
        rec["lists"] += 1
        rec["items"] += len(row.items)
    return dict(sorted(out.items()))


ACCOUNTING = "eval_accounting.json"


def dedup_rate(stage_manifest: dict) -> float:
    """Share of rollouts whose eval another rollout had already paid for.

    §12: a high rate is a saving, not a bug -- but only if it is written down, because it is
    also the difference between the evals the campaign planned and the ones it bought.
    """
    return pct(stage_manifest["deduped"], stage_manifest["rollouts"])


def evals_per_gpu_hour(accounting: dict | None, evals: int) -> float | None:
    """§9's constant, observed rather than assumed. ``None`` when job C's time was not recorded.

    Job C is a Slurm array and the harness writes no timing of its own, so the elapsed time
    comes from the scheduler and lands in ``eval_accounting.json`` beside the campaign::

        sacct -j <ARRAYJOBID> --format=JobID,Elapsed,AllocTRES --noheader --parsable2

    one ``{"elapsed_s": ..., "gpus": ...}`` per array task. ``gpus`` is
    ``num_gpu_devices``, not the node's count: at the default 1 the harness runs one work
    item at a time whatever the node holds, and charging 8 GPU-hours for that would understate
    the observed rate eightfold.
    """
    if accounting is None:
        return None
    gpu_seconds = sum(s["elapsed_s"] * s["gpus"] for s in accounting["shards"])
    if gpu_seconds <= 0:
        raise ValueError(
            f"{ACCOUNTING} records no GPU time: the rate §9's budget is rescaled from would "
            "be infinite, which is how an unrecorded campaign gets reported as a fast one"
        )
    return evals * 3600.0 / gpu_seconds


def pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


def mean_lengths(rollout_rows) -> dict[str, float]:
    """``prefix_id -> mean continuation tokens``, over the rollouts that finished.

    Truncated rollouts are excluded, and that is the point: each one sits at
    ``max_new_tokens`` by construction, so counting them would stretch exactly the prefixes
    that had truncations and manufacture the length effect this is here to detect. It is the
    same set values.py aggregated V̂ from, so the two sides of the correlation match.
    """
    lengths: dict[str, list[int]] = defaultdict(list)
    for r in rollout_rows:
        if r.truncation == corpus.OK:
            lengths[r.prefix_id].append(r.n_gen_tokens)
    return {pid: statistics.fmean(ns) for pid, ns in lengths.items()}


def mean_encoded_lengths(rollout_rows, orm_scores: dict[str, dict]) -> dict[str, float]:
    """``prefix_id -> mean ORM-encoded sequence length``, over rollouts that finished and were
    scored by job B2 (empty on a measured campaign, which never runs job B2).

    Despite its name, ``n_code_tokens`` is NOT code length: it is the whole sequence
    `orm_score.SequenceEncoder` encoded -- instruction + reference + separator + code + EOS
    (task 3/6). Reported here as ``encoded_seq_len``, never as a "code length" control, so a
    reader cannot mistake it for one. Same truncation exclusion as `mean_lengths`, for the
    same reason: a truncated rollout's fragment is not comparable length signal.
    """
    lengths: dict[str, list[int]] = defaultdict(list)
    for r in rollout_rows:
        if r.truncation == corpus.OK and r.rollout_id in orm_scores:
            lengths[r.prefix_id].append(orm_scores[r.rollout_id]["n_code_tokens"])
    return {pid: statistics.fmean(ns) for pid, ns in lengths.items()}


def correlation(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r, or ``None`` where it does not exist (§6's stage-2 guard).

    A strong positive r means the campaign may be teaching "longer continuation = better"
    rather than "better prefix = better" -- a beam-source artifact that a headline NDCG
    would happily report as signal.
    """
    if len(xs) < 2:
        return None
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        # A constant series has no correlation, and 0.0 would read as "measured, and length
        # does not matter" rather than "there was nothing here to measure".
        return None


# --- the pass: a campaign on disk -> the report and its §12 checks -------------------------

NULL_REPLICATES = 200
# The observed median is held against a high quantile of the null, not against its median:
# "the typical list spreads more than 95% of lists that differ by nothing but noise" is a
# claim about signal; "more than the average noisy list" is a coin flip (§12).
NULL_Q = 0.95


def quantile(xs: list[float], q: float) -> float:
    """The q-th quantile by nearest rank -- no interpolation, no numpy in a CPU report."""
    ordered = sorted(xs)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def beats_own_null(rows, scores_by_id, rng, replicates) -> tuple[int, int]:
    """How many lists spread further than **their own** noise floor, and out of how many.

    Each list against a null bootstrapped from that list alone -- not against a null pooled
    over the campaign. Pooling is what broke the first version of this gate: a pooled q95 is
    set by the widest lists (the range of `n` draws grows with `n`) while the observed median
    is set by the narrowest, so the two describe different populations and the comparison
    came out negative on corpora with signal and without it alike.

    The share this returns has a null value of exactly ``1 - NULL_Q`` by construction, which
    is what lets the gate be stated without inventing a spread threshold.
    """
    beats = sum(
        1
        for row in rows
        if spread(row) > quantile(null_spreads([row], scores_by_id, rng, replicates), NULL_Q)
    )
    return beats, len(rows)


def spread_report(rows, scores_by_id, cfg) -> dict:
    """Observed within-list spread against the floor the same lists produce on noise alone."""
    if not rows:
        return {"lists": 0, "median": None, "null_median": None, "null_q95": None,
                "margin": None, "beats_null": 0, "beats_null_pct": None,
                "replicates": NULL_REPLICATES, "seed": cfg.select_seed}
    observed = [spread(row) for row in rows]
    # The campaign's own seed, so the report is reproducible from the config that built it.
    null = null_spreads(rows, scores_by_id, random.Random(cfg.select_seed), NULL_REPLICATES)
    beats, n = beats_own_null(
        rows, scores_by_id, random.Random(cfg.select_seed), NULL_REPLICATES
    )
    median, floor = statistics.median(observed), quantile(null, NULL_Q)
    return {
        "lists": len(rows),
        "median": median,
        "p25": quantile(observed, 0.25),
        "p75": quantile(observed, 0.75),
        # Descriptive only. The pooled null is worth printing -- it is what §12 reasoned
        # about -- but it is not what the gate reads; see `beats_own_null`.
        "null_median": statistics.median(null),
        "null_q95": floor,
        "margin": median - floor,
        "beats_null": beats,
        "beats_null_pct": pct(beats, n),
        "replicates": NULL_REPLICATES,
        "seed": cfg.select_seed,
    }


def read_json(path: str, default=None):
    if not os.path.isfile(path):
        return default
    with open(path) as f:
        return json.load(f)


def check_label_source(cfg_label_source: str, value_rows) -> str:
    """The config's ``label_source`` must agree with what ``values.jsonl`` actually carries --
    a stale file left over from a different build (or a hand-edited config) would otherwise
    silently mislabel every number in this report. Empty ``value_rows`` has nothing to
    disagree with and is not itself an error here (§12's own checks already gate on it).
    """
    seen = {v.label_source for v in value_rows}
    if seen and seen != {cfg_label_source}:
        raise ValueError(
            f"{values.VALUES} carries label_source {sorted(seen)} but the config says "
            f"{cfg_label_source!r} -- the two disagree about what these V-hats mean"
        )
    return cfg_label_source


def calibration_summary(out_dir: str, prefix_rows, label_source: str) -> dict | None:
    """Job D's curve, offsets and how its fit went, surfaced into the campaign report.

    ``None`` unless THIS campaign's labels came from the curve. Keying off
    ``calibrate_manifest.json`` existing was wrong: the level-1 validation sweep runs job D
    beside a fully measured campaign, so the file is there and the report described a curve
    that labelled nothing in it.

    Read only (N7): unlike ``values.load_curve`` (which REFUSES to apply a broken curve before
    labelling anything), a report has to stay legible even when the curve it is describing IS
    broken, so the same defects are FLAGGED here (`curve_flags`) instead of raising.
    """
    if label_source != "imputed":
        return None
    manifest = read_json(os.path.join(out_dir, calibrate.CALIB_MANIFEST))
    if manifest is None:
        return None
    curve = read_json(os.path.join(out_dir, calibrate.CURVE)) or {}
    offsets_blob = read_json(os.path.join(out_dir, calibrate.OFFSETS)) or {}
    offsets, meta = offsets_blob.get("offsets", {}), offsets_blob.get("meta", {})
    anchors = manifest.get("anchors", {})

    xs, resid = curve.get("knots_x") or [], curve.get("resid_var_by_band") or []
    flags = []
    # `>=`, matching values.load_curve's refusal: a tie is not garbage to np.interp, but it
    # makes one band unreachable by band(), so a report that passed it would disagree with
    # the gate that would have refused the same file.
    if any(a >= b for a, b in zip(xs, xs[1:])):
        flags.append("knots_x is not strictly ascending -- np.interp would return garbage")
    if resid and all(v == 0.0 for v in resid):
        flags.append("resid_var_by_band is all zero -- se_imputed would understate its error")
    fit = manifest.get("fit") or {}
    # What the apply pass actually paid, off values_manifest.json: `Curve.n_clipped` is a live
    # counter aggregate_imputed drove, and the share of rollouts that landed outside the
    # curve's fitted score range is the distribution-shift signal job D's own fit cannot see.
    valued = read_json(os.path.join(out_dir, values.VALUES_MANIFEST)) or {}

    campaign_problems = {f"{p.level}:{p.problem_id}" for p in prefix_rows}
    no_anchor = campaign_problems - set(offsets)
    target_dist = curve.get("target_dist", {})

    return {
        "label_source": manifest.get("label_source"),
        "curve_flags": flags,
        # A non-converged fit is not a footnote: the offsets and the curve were still moving
        # when job D stopped, and every V-hat in the campaign came off that curve.
        "fit": {
            "converged": fit.get("converged"),
            "n_iters": fit.get("n_iters"),
            "max_offset_delta": fit.get("max_offset_delta"),
            "max_knot_delta": fit.get("max_knot_delta"),
        },
        "clipped": {
            "n": valued.get("clipped"),
            "lookups": valued.get("curve_lookups"),
            "rate_pct": valued.get("clip_rate_pct"),
        },
        # band -> mean score -> mean target -> n, in one table (the fitted curve itself).
        "curve_bands": [
            {"band": i, "score": x, "target": y, "n": n}
            for i, (x, y, n) in enumerate(
                zip(xs, curve.get("knots_y") or [], curve.get("n_by_band") or [None] * len(xs))
            )
        ],
        "offsets": {
            "tau2": meta.get("tau2"), "tau2_raw": meta.get("tau2_raw"),
            "tau2_estimable": meta.get("tau2_estimable"),
            "shrink_mean": meta.get("shrink_mean"),
            "clamp_rate": pct(meta.get("n_clamped", 0), meta.get("n_problems", 0)),
            "n_problems": meta.get("n_problems"),
            # This campaign's own problems, not the anchor set's -- offset_for silently
            # returns 0.0 for one with no anchors, which this makes visible instead.
            "no_anchor_rate": (
                pct(len(no_anchor), len(campaign_problems)) if campaign_problems else None
            ),
        },
        # The finding-2 diagnostic: everything above split by whether a candidate could even
        # appear in the ORM's own training lists.
        "orm_seen_vs_unseen": {
            "n_seen": anchors.get("orm_seen"), "n_unseen": anchors.get("orm_unseen"),
            "target_dist_fit_unseen_only": target_dist.get("fit"),
            "target_dist_all": target_dist.get("all"),
        },
    }


def report(cfg) -> dict:
    """Measure a finished campaign and check it against §12; returns the report it writes."""
    rollout_cfg = cfg.prm_rollout
    rollout_cfg.validate()
    out_dir = _resolve(rollout_cfg.out_dir)

    def at(name: str) -> str:
        return os.path.join(out_dir, name)

    manifest = read_json(at(build.MANIFEST))
    if not manifest:
        raise FileNotFoundError(
            f"no {build.MANIFEST} in {out_dir}; run processkernel.prm.rollout.prefixes first"
        )
    staged = read_json(at(stage.STAGE_MANIFEST), default={})
    valued = read_json(at(values.VALUES_MANIFEST), default={})
    listed = read_json(at(lists.LISTS_MANIFEST), default={})
    accounting = read_json(at(ACCOUNTING))

    prefix_rows = stage.read_prefixes(at(prefixes.PREFIXES))
    value_rows = lists.read_values(at(values.VALUES))
    label_source = check_label_source(rollout_cfg.label_source, value_rows)
    by_split = {
        split: lists.read_lists(at(lists.LISTS.format(split=split)))
        for split in (prefixes.TRAIN, prefixes.VAL)
    }
    all_rows = [row for rows in by_split.values() for row in rows]
    scores_by_id = {v.prefix_id: v.scores for v in value_rows}
    measured = set(scores_by_id)

    rollout_rows = [
        r
        for part in sorted(glob.glob(at(os.path.join(stage.ROLLOUTS, "*.jsonl.gz"))))
        for r in stage.read_rollouts(part)
    ]
    lengths = mean_lengths(rollout_rows)
    paired = [(pid, lengths[pid]) for pid in sorted(scores_by_id) if pid in lengths]
    # Empty on a measured campaign (job B2 never runs there); mean_encoded_lengths then
    # returns {} and the second correlation below reports n=0, r=None rather than crashing.
    orm_scores = values.read_rollout_scores(out_dir)
    enc_lengths = mean_encoded_lengths(rollout_rows, orm_scores)
    enc_paired = [(pid, enc_lengths[pid]) for pid in sorted(scores_by_id) if pid in enc_lengths]

    out = {
        "out_dir": out_dir,
        "label_source": label_source,
        "created": manifest.get("created"),
        "git_sha": manifest.get("git_sha"),
        "git_dirty": manifest.get("git_dirty"),
        "config": {
            k: getattr(rollout_cfg, k)
            for k in ("source", "K", "min_rollouts", "min_rel_depth",
                      "max_rel_depth", "depth_buckets", "min_list_size", "max_list_size",
                      "train_selection", "label_mode", "speedup_stat", "dedup_by_code_sha1")
        },
        "pipeline": {
            "prefixes": manifest.get("prefixes", len(prefix_rows)),
            "rollouts": staged.get("rollouts", 0),
            "staged": staged.get("staged", 0),
            "evals": valued.get("evals", 0),
            "values": valued.get("values", len(value_rows)),
            "lists": listed.get("lists", {}),
            "items": listed.get("items", {}),
        },
        "drops": {
            "values": {r: {"n": n, "pct": pct(n, manifest.get("prefixes", 0))}
                       for r, n in (valued.get("dropped") or {}).items()},
            "lists": {r: {"n": n, "pct": pct(n, n + len(all_rows))}
                      for r, n in (listed.get("dropped") or {}).items()},
        },
        "by_source": slice_counts(prefix_rows, measured, all_rows, "source"),
        "dedup": {
            "rollouts": staged.get("rollouts", 0),
            "shared": staged.get("deduped", 0),
            "hit_rate_pct": dedup_rate(staged) if staged else 0.0,
        },
        "evals_per_gpu_hour": evals_per_gpu_hour(accounting, valued.get("evals", 0)),
        "value_histogram": value_histogram(all_rows),
        "spread": spread_report(all_rows, scores_by_id, rollout_cfg),
        "list_sizes": {
            "histogram": size_histogram(all_rows),
            "two_item": two_item(all_rows),
            "two_item_pct": pct(two_item(all_rows), len(all_rows)),
        },
        "lists_outside_window": outside_window(all_rows, rollout_cfg),
        "length_correlation": {
            "n": len(paired),
            "r": correlation(
                [statistics.fmean(scores_by_id[pid]) for pid, _ in paired],
                [n for _, n in paired],
            ),
            # On both length measures (task 6 extra req. e): continuation length (above,
            # `n_gen_tokens`, meaningful on either label source) and the ORM's own encoded
            # sequence length (below -- imputed campaigns only, and NOT "code length", see
            # mean_encoded_lengths).
            "encoded_seq_len_n": len(enc_paired),
            "encoded_seq_len_r": correlation(
                [statistics.fmean(scores_by_id[pid]) for pid, _ in enc_paired],
                [n for _, n in enc_paired],
            ),
        },
        "calibration": calibration_summary(out_dir, prefix_rows, label_source),
        "depth": [
            {
                "lo": band.lo,
                "hi": band.hi,
                "n_lists": len(band.rows),
                "n_items": sum(len(r.items) for r in band.rows),
                **{k: v for k, v in spread_report(band.rows, scores_by_id, rollout_cfg).items()
                   if k in ("median", "null_q95", "margin", "beats_null", "beats_null_pct")},
            }
            for band in depth_bands(all_rows, rollout_cfg)
        ],
    }
    out["checks"] = run_checks(out, by_split, prefix_rows, staged, valued, listed)
    write_atomic(at(STATS), json.dumps(out, indent=2, default=str))
    return out

# --- §12's acceptance table, as checks that exit non-zero ----------------------------------

MAX_TRUNCATED_PCT = 2.0
MAX_TOO_FEW_PCT = 5.0
ALL_EQUAL_RANGE = (2.0, 10.0)
MIN_LISTS, MIN_VAL_LISTS = 300, 60
# The share statistic's own null rate, by construction: under "every item of a list shares
# one true value", exactly 1 - NULL_Q of lists clear their own q95. So this is a calibrated
# baseline rather than an invented spread threshold. Measured on synthetic null corpora it
# comes out at 0.7-1.3% (K=5 and K=8, three seeds), so the gate carries a 4-7x margin.
SIGNAL_SHARE_MIN = 100.0 * (1 - NULL_Q)


def run_checks(out: dict, by_split, prefix_rows, staged, valued, listed) -> list[dict]:
    """§12's table, measured. Each entry is ``{check, ok, detail}``; `main` exits 1 on any fail."""
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    dropped = valued.get("dropped") or {}
    n_rollouts = staged.get("rollouts", 0)
    truncated = pct(dropped.get(values.TRUNCATED, 0), n_rollouts)
    check(
        f"rollouts truncated under {MAX_TRUNCATED_PCT}%",
        truncated < MAX_TRUNCATED_PCT,
        f"{dropped.get(values.TRUNCATED, 0)} of {n_rollouts} ({truncated:.2f}%)",
    )

    n_prefixes = out["pipeline"]["prefixes"]
    too_few = pct(dropped.get(values.TOO_FEW_ROLLOUTS, 0), n_prefixes)
    check(
        f"prefixes dropped too_few_rollouts under {MAX_TOO_FEW_PCT}%",
        too_few < MAX_TOO_FEW_PCT,
        f"{dropped.get(values.TOO_FEW_ROLLOUTS, 0)} of {n_prefixes} ({too_few:.2f}%)",
    )

    n_lists = sum(out["pipeline"]["lists"].values())
    all_equal = (listed.get("dropped") or {}).get(lists.ALL_EQUAL, 0)
    rate = pct(all_equal, all_equal + n_lists)
    lo, hi = ALL_EQUAL_RANGE
    check(
        f"all-equal lists dropped within {lo}-{hi}%",
        lo <= rate <= hi,
        f"{all_equal} of {all_equal + n_lists} ({rate:.2f}%) -- under {lo}% the labels may "
        f"not be moving at all, over {hi}% the level is saturated",
    )

    val_lists = out["pipeline"]["lists"].get(prefixes.VAL, 0)
    check(
        f"usable lists >= {MIN_LISTS} with >= {MIN_VAL_LISTS} in val",
        n_lists >= MIN_LISTS and val_lists >= MIN_VAL_LISTS,
        f"{n_lists} lists, {val_lists} in val",
    )

    # N3, and the split is the whole of it. `Prefix.__post_init__` already refuses to
    # construct a val prefix that was scored, so "is it random" cannot fail here -- the file
    # would not have read back. What that guard cannot see is a *train* prefix landing on a
    # val list: lists.py takes the split from splits.json rather than from the prefix, so a
    # disagreement between the two is invisible to both files and shows up only across them.
    by_id = {p.prefix_id: p for p in prefix_rows}
    leaked = [
        item.prefix_id
        for row in by_split[prefixes.VAL]
        for item in row.items
        if item.prefix_id not in by_id or by_id[item.prefix_id].split != prefixes.VAL
    ]
    check(
        "every prefix on a val list is a val-split prefix (N3)",
        not leaked,
        f"{len(leaked)} leaked" + (f", e.g. {leaked[0]}" if leaked else "")
        + (" -- the v1-PRM ranking number would be inflated" if leaked else "")
        + (f"; {RANDOM} selection is enforced in Prefix.__post_init__" if not leaked else ""),
    )

    spanning = [
        row.list_key
        for rows in by_split.values()
        for row in rows
        if not _one_key(row, by_id)
    ]
    check(
        "every list is one run, one cut depth (N2, N4)",
        not spanning,
        f"{len(spanning)} span" + (f", e.g. {spanning[0]}" if spanning else ""),
    )

    empty = [f"{b['lo']}-{b['hi']}" for b in out["depth"] if not b["n_lists"]]
    check(
        "every depth bucket is populated",
        not empty,
        f"{len(empty)} empty: {empty}" if empty else f"{len(out['depth'])} bands, all populated",
    )
    check(
        "every list is inside the depth window",
        not out["lists_outside_window"],
        f"{out['lists_outside_window']} outside [{out['config']['min_rel_depth']}, "
        f"{out['config']['max_rel_depth']}] -- the depth knobs moved since job A",
    )

    sp = out["spread"]
    check(
        f"share of lists whose V-hat spread beats their own null over {SIGNAL_SHARE_MIN}%",
        sp["beats_null_pct"] is not None and sp["beats_null_pct"] > SIGNAL_SHARE_MIN,
        f"{sp['beats_null']} of {sp['lists']} ({_f(sp['beats_null_pct'])}%); observed median "
        f"{_f(sp['median'])}, pooled null median {_f(sp['null_median'])} q{NULL_Q} "
        f"{_f(sp['null_q95'])} -- §12's fixed 0.15 passes on pure noise, and a pooled-null "
        "margin compares widths that were never comparable, so each list is held against a "
        "null bootstrapped from itself",
    )

    check(
        "observed evals/GPU-hour recorded",
        out["evals_per_gpu_hour"] is not None,
        f"{_f(out['evals_per_gpu_hour'])}" if out["evals_per_gpu_hour"] is not None
        else f"no {ACCOUNTING} in the campaign -- §9's budget cannot be rescaled from a run "
             "whose GPU time nobody wrote down",
    )
    return checks


def _one_key(row, by_id: dict) -> bool:
    """N2 and N4 re-checked off the prefixes, which is where these fields actually live."""
    keys = {
        (p.run_tag, p.level, p.problem_id, p.cut_index)
        for p in (by_id.get(i.prefix_id) for i in row.items)
        if p is not None
    }
    return len(keys) <= 1


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _n(x) -> str:
    return "n/a" if x is None else f"{x:,}"


def render(out: dict) -> str:
    """The report as text; the same numbers ``rollout_stats.json`` carries."""
    p, sp, sz = out["pipeline"], out["spread"], out["list_sizes"]
    lines = [
        "=" * 78,
        f"PRM rollout campaign  {out['out_dir']}",
        f"  built {out['created']}  git {str(out['git_sha'])[:12]}"
        f"{' DIRTY' if out['git_dirty'] else ''}",
        f"  {out['config']}",
        "=" * 78,
        f"prefixes {p['prefixes']:,} -> rollouts {p['rollouts']:,} -> staged {p['staged']:,}"
        f" -> evals {p['evals']:,} -> values {p['values']:,}",
        f"lists {p['lists']}   items {p['items']}   "
        f"pairs {sz['two_item']} ({sz['two_item_pct']:.1f}%)   widths {sz['histogram']}",
        f"dedup {out['dedup']['shared']:,} of {out['dedup']['rollouts']:,} rollouts "
        f"({out['dedup']['hit_rate_pct']:.1f}%)   "
        f"evals/GPU-h {_f(out['evals_per_gpu_hour'])}",
        "",
        "drops",
    ]
    for stage_name, reasons in out["drops"].items():
        fired = {r: d for r, d in reasons.items() if d["n"]}
        lines.append(
            f"  {stage_name:<8} " + ("   ".join(
                f"{r} {d['n']:,} ({d['pct']:.2f}%)" for r, d in fired.items()
            ) or "(none)")
        )

    lines += [
        "",
        f"within-list V-hat spread over {sp['lists']:,} lists",
        f"  observed  median {_f(sp['median'])}   p25 {_f(sp.get('p25'))}"
        f"   p75 {_f(sp.get('p75'))}",
        f"  null      median {_f(sp['null_median'])}   q{NULL_Q} {_f(sp['null_q95'])}"
        f"   ({sp['replicates']} replicates, seed {sp['seed']})   [descriptive only]",
        f"  SIGNAL    {sp['beats_null']:,} of {sp['lists']:,} lists beat their OWN null"
        f" ({_f(sp['beats_null_pct'])}%, null rate {SIGNAL_SHARE_MIN}%)"
        "   <- what decides whether stage 2 is funded",
        "",
        "by depth band",
    ]
    lines += [
        f"  {b['lo']:.2f}-{b['hi']:.2f}   lists {b['n_lists']:>6,}   items {b['n_items']:>6,}"
        f"   spread {_f(b['median'])}   beat own null {_f(b['beats_null_pct'])}%"
        for b in out["depth"]
    ]

    lc = out["length_correlation"]
    lines += [
        "",
        f"V-hat histogram   {out['value_histogram']}",
        f"V-hat vs continuation length   r {_f(lc['r'])} over {lc['n']:,} prefixes"
        "   (a strong positive r is 'longer is better', not signal)",
        f"V-hat vs ORM encoded seq. length   r {_f(lc['encoded_seq_len_r'])} "
        f"over {lc['encoded_seq_len_n']:,} prefixes"
        "   (n_code_tokens is the WHOLE encoded sequence, not code alone)",
        "",
        "by source   " + str(out["by_source"]),
    ]

    cal = out.get("calibration")
    if cal is not None:
        off, fit, clip = cal["offsets"], cal["fit"], cal["clipped"]
        # A banner, not a field: `converged: False` is the state of the real level-1 fit, and
        # a reader who has to notice one key among forty will not. `None` (an old manifest
        # that recorded no fit) shouts too -- unknown convergence is not convergence.
        if not fit["converged"]:
            lines += [
                "",
                "!" * 78,
                f"!!  JOB D'S FIT DID NOT CONVERGE after {fit['n_iters']} iterations: "
                f"max_offset_delta {_f(fit['max_offset_delta'])} (tol {calibrate.TOL}), "
                f"max_knot_delta {_f(fit['max_knot_delta'])}",
                "!!  every V-hat above was labelled from that curve -- raise curve_iters and "
                "re-fit before trusting them",
                "!" * 78,
            ]
        lines += [
            "",
            f"calibration (label_source={cal['label_source']!r})"
            f"{'   FLAGS: ' + '; '.join(cal['curve_flags']) if cal['curve_flags'] else ''}",
            f"  fit   converged {fit['converged']}   iters {fit['n_iters']}   "
            f"max_offset_delta {_f(fit['max_offset_delta'])}   "
            f"max_knot_delta {_f(fit['max_knot_delta'])}",
            f"  curve bands   " + str(cal["curve_bands"]),
            f"  clipped   {_n(clip['n'])} of {_n(clip['lookups'])} curve lookups "
            f"({_f(clip['rate_pct'])}%)   <- rollouts outside the curve's fitted score range",
            f"  offsets   tau2 {_f(off['tau2'])}   shrink_mean {_f(off['shrink_mean'])}   "
            f"clamp_rate {_f(off['clamp_rate'])}%   no_anchor_rate {_f(off['no_anchor_rate'])}%"
            f"   n_problems {off['n_problems']}",
            f"  ORM-seen vs unseen   seen {cal['orm_seen_vs_unseen']['n_seen']}   "
            f"unseen {cal['orm_seen_vs_unseen']['n_unseen']}   "
            f"target_dist(fit) {cal['orm_seen_vs_unseen']['target_dist_fit_unseen_only']}   "
            f"target_dist(all) {cal['orm_seen_vs_unseen']['target_dist_all']}",
        ]

    lines += [
        "",
        "checks",
        "-" * 78,
    ]
    for ck in out["checks"]:
        lines.append(f"  [{'PASS' if ck['ok'] else 'FAIL'}] {ck['check']:<52} {ck['detail']}")
    failed = [ck for ck in out["checks"] if not ck["ok"]]
    lines += ["-" * 78, f"{len(out['checks']) - len(failed)}/{len(out['checks'])} checks passed"]
    return "\n".join(lines)


def main(argv=None) -> None:
    out = report(load_config(None if argv is None else list(argv)))
    print(render(out))
    print(f"Written: {os.path.join(out['out_dir'], STATS)}")
    if any(not ck["ok"] for ck in out["checks"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
