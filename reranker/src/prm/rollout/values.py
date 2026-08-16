"""What the evals said -> V̂ per prefix: job D's measurement half (PLAN_v2 §6), plus the
imputed path that turns ORM scores into V̂ via job D's calibrated curve (PLAN_v3 §8).

N1 is the whole point: a row is measured OR imputed, never averaged. On the measured path
every number comes from the prefix's own rollouts -- the v1 verdict for the completion the
prefix was cut out of is one draw from a *different* distribution, so pooling it in would
bias V̂ toward the observed path, and nothing in this module reads v1's parts. On the imputed
path every number comes from `curve.json`/`offsets.json` (read, never re-fit -- N7) applied
to that SAME prefix's own rollouts' ORM scores; an imputed row's measured-only counters
(`n_compiled`, `n_correct`, `v_binary`, `se_binary`, `se_graded`) are `None`, never `0` --
mixing the two label sources under one schema is the failure N1 exists to prevent.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from checker.submission import SubmissionAnalyzer

from reranker.src.config import _resolve, load_config
from reranker.src.data.labels import load_baseline_times
from reranker.src.prm import corpus, targets
from reranker.src.prm.build import (
    NO_BASELINE,
    NO_FINISH_REASON,
    NO_RUNTIME,
    TRUNCATED,
    write_atomic,
)
from reranker.src.prm.rollout import calibrate, orm_score, prefixes, stage
from reranker.src.prm.targets import MEAN, MIN

VALUES = "values.jsonl"
VALUES_MANIFEST = "values_manifest.json"

NO_EVAL_ENTRY = corpus.NO_EVAL_ENTRY
# Applied in this order, and the order is what makes the ledger readable: a rollout that is
# both truncated and unevaluated is counted once, as truncated (§6). v1's names, from v1's
# build.py, so one campaign's drops can be read against the other's. Shared by both paths.
REASONS = (TRUNCATED, NO_FINISH_REASON, NO_EVAL_ENTRY, NO_BASELINE, NO_RUNTIME)
# Not in REASONS: these drop the *prefix*, so no row survives to carry them. They land on
# the campaign ledger instead, which is why aggregate_measured/aggregate_imputed take one.
TOO_FEW_ROLLOUTS = "too_few_rollouts"
NO_ROLLOUTS = "no_rollouts"
# Imputed-path only, and deliberately NOT in REASONS: an S1 failure does not drop the
# rollout (it still counts toward n_rollouts, scored 0.0 -- a deduction, not a drop), so it
# belongs on the campaign ledger beside TOO_FEW_ROLLOUTS, not on the per-row n_dropped dict.
S1_FAIL = "s1_fail"
LEDGER = REASONS + (TOO_FEW_ROLLOUTS, NO_ROLLOUTS, S1_FAIL)

_SUBMISSION_ANALYZER = SubmissionAnalyzer()


@dataclass(frozen=True)
class Eval:
    """One rollout's verdict, and whether the dedup had it share another rollout's eval."""

    verdict: dict | None
    shared: bool = False


@dataclass(frozen=True)
class Value:
    """One row of ``values.jsonl`` (PLAN_v2 §5, PLAN_v3 §8).

    ``n_rollouts`` is the merge key with v1: a v1 row is this same measurement at
    ``n_rollouts = 1``. ``scores`` is kept per rollout so `label_mode`, `speedup_stat` and
    the speed knobs stay re-derivable from a built dataset without re-running an eval.

    ``label_source`` is ``"measured"`` or ``"imputed"`` (N1) -- never both, and a reader must
    not average across the two. The measured-only counters (``n_compiled``, ``n_correct``,
    ``v_binary``, ``se_binary``, ``se_graded``) are ``None`` on an imputed row -- ``0`` would
    claim a compile/correctness verdict this path never observed, and would make the row
    silently averageable with a measured one. ``orm_offset`` (the per-problem ``c`` applied
    inside the curve lookup) and ``se_imputed`` are the imputed-only counterparts, ``None``
    on a measured row.
    """

    prefix_id: str
    K_requested: int
    n_rollouts: int
    n_dropped: dict
    n_dedup_shared: int
    n_compiled: int | None
    n_correct: int | None
    v_binary: float | None
    v_graded: float
    se_binary: float | None
    se_graded: float | None
    scores: list
    label_source: str = "measured"
    orm_offset: float | None = None
    se_imputed: float | None = None


def read_verdicts(runs_dir: str, run_name: str) -> dict[tuple[int, int], dict]:
    """Every eval shard's ``eval_results.json`` as one ``(problem_id, sample_id) -> entry`` table.

    One file per shard is not a preference: `add_to_eval_results_file` appends by
    load-rewrite of the whole file, so two jobs sharing one would drop each other's results.
    Job D is where they come back together.
    """
    # Anchored on the shard number, not left as the glob's prefix match: `prm_rollout_v1_s*`
    # also matches `prm_rollout_v1_smoke_s00` -- and this repo already names campaigns X and
    # X_smoke -- so the smoke run's verdicts would be read into the real campaign's table,
    # where both stage dense sample ids from 0 over the same level. The anchor also drops an
    # archived `..._s02.tar` beside them, which would otherwise read as an unevaluated shard.
    shard_dir = re.compile(rf"^{re.escape(run_name)}_s\d+$")
    run_dirs = sorted(
        d
        for d in glob.glob(os.path.join(runs_dir, f"{run_name}_s*"))
        if shard_dir.match(os.path.basename(d))
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"no eval run dirs matching {run_name}_s<NN> under {runs_dir} -- either the "
            "campaign was staged under a different eval_run_name, or nothing has been "
            "staged yet"
        )
    table: dict[tuple[int, int], dict] = {}
    for run_dir in run_dirs:
        path = os.path.join(run_dir, stage.EVAL_RESULTS)
        if not os.path.isfile(path):
            # An empty table would be indistinguishable from a shard whose every kernel
            # failed, and would drop that shard's prefixes as `no_eval_entry` instead.
            raise FileNotFoundError(
                f"{path} does not exist: shard {os.path.basename(run_dir)} has not been "
                "evaluated yet. Run job C for it before values.py, or its prefixes are "
                "silently dropped rather than measured"
            )
        for key, entry in corpus.verdicts(path).items():
            if key in table:
                raise ValueError(
                    f"(problem {key[0]}, sample {key[1]}) is graded by two eval shards, the "
                    f"second in {os.path.basename(run_dir)} -- the shard plan partitions the "
                    "problems, so two run dirs were pointed at overlapping subsets and "
                    "neither verdict can be taken as the sample's"
                )
            table[key] = entry
    return table


def join(rows, rollout_map: dict[str, dict], verdicts: dict[tuple[int, int], dict]):
    """``rollout_id -> Eval``: what the harness said about each rollout, via stage.py's map.

    The map is the only thing that knows a rollout's synthetic sample id, and deduped
    rollouts resolve through the kernel that was actually staged -- so a saved eval stays a
    measurement instead of becoming a lost one.
    """
    out: dict[str, Eval] = {}
    for r in rows:
        if r.rollout_id not in rollout_map:
            raise KeyError(
                f"{r.rollout_id} is in no eval run dir: {stage.ROLLOUT_MAP} does not place "
                "it, which means it was generated after the staging pass. Re-stage before "
                "scoring, or the prefix reports a K the campaign never bought"
            )
        placed = rollout_map[r.rollout_id]
        out[r.rollout_id] = Eval(
            verdicts.get((placed["problem_id"], placed["sample_id"])),
            bool(placed.get("shared", False)),
        )
    return out


def aggregate_measured(prefix, rows, joined, baselines, cfg, counts) -> Value | None:
    """Aggregate one prefix's rollouts into its measured value, or ``None`` if too few survive.

    ``counts`` is the campaign ledger and is updated for every drop, including the ones that
    happened inside a prefix this returns ``None`` for: those evals were paid for, and a
    ledger that forgot them would make the campaign look cheaper than it was.

    Pure movement from v2's ``value_for`` (PLAN_v3 §8) -- this function never touches the ORM,
    the curve or the offsets; see ``aggregate_imputed`` for the sibling that does.
    """
    baseline = baselines.get(prefix.level, {}).get(prefix.problem_id)
    dropped: Counter = Counter()
    scores, n_compiled, n_correct, shared = [], 0, 0, 0

    def drop(reason: str) -> None:
        dropped[reason] += 1
        counts[reason] += 1

    # Sorted, not in arrival order: `scores` is a stored column and job B returns its
    # rollouts bucketed by token budget, so the unsorted order is not stable across reruns.
    for r in sorted(rows, key=lambda r: r.rollout_id):
        if r.truncation == corpus.TRUNCATED:
            drop(TRUNCATED)
            continue
        # Not folded into the branch above: v1 tells the two apart because a rollout whose
        # finish reason never arrived is a harness gap, not a generation that ran out of room.
        if r.truncation != corpus.OK:
            drop(NO_FINISH_REASON)
            continue
        ev = joined[r.rollout_id]
        if ev.verdict is None:
            drop(NO_EVAL_ENTRY)
            continue
        compiled = bool(ev.verdict.get("compiled", False))
        target = targets.target_for(
            compiled=compiled,
            correct=bool(ev.verdict.get("correctness", False)),
            ours=_runtimes(ev.verdict),
            baseline=baseline,
            mode=cfg.label_mode,
            stat=cfg.speedup_stat,
            lo=cfg.speedup_lo,
            hi=cfg.speedup_hi,
            quant=cfg.speed_quant,
        )
        if target is None:
            # Only a correct kernel reaches here. v1's PRM-4 distinction, restated: a kernel
            # the harness never timed is not an absent baseline, and one entry for both
            # would hide an eval that ran and reported nothing.
            timed = targets.usable((baseline or {}).get(cfg.speedup_stat))
            drop(NO_RUNTIME if timed else NO_BASELINE)
            continue
        scores.append(target.value)
        n_compiled += compiled
        n_correct += target.label
        shared += ev.shared

    n = len(scores)
    if n < cfg.min_rollouts:
        # A V̂ from too few draws is a v1 label wearing a v2 schema, and mixing those is
        # exactly what this plan forbids -- so the prefix leaves the corpus, not just the list.
        counts[TOO_FEW_ROLLOUTS] += 1
        return None
    v_binary = n_correct / n
    v_graded = sum(scores) / n
    return Value(
        prefix_id=prefix.prefix_id,
        K_requested=prefix.K,
        n_rollouts=n,
        # Every reason, zeros included: a reader must not have to decide whether a missing
        # key is a zero or a reason this build never knew about.
        n_dropped={reason: dropped[reason] for reason in REASONS},
        n_dedup_shared=shared,
        n_compiled=n_compiled,
        n_correct=n_correct,
        v_binary=v_binary,
        v_graded=v_graded,
        se_binary=math.sqrt(v_binary * (1 - v_binary) / n),
        # One rollout has no measurable spread, and `statistics.stdev` raises on it rather
        # than returning 0. min_rollouts is what keeps such a prefix out of the corpus at
        # all; this branch only decides what the row says when it is deliberately let in.
        se_graded=statistics.stdev(scores) / math.sqrt(n) if n > 1 else 0.0,
        scores=scores,
    )


def write_values(values, path) -> None:
    """The writer, extracted (PLAN_v3 §8) so both aggregation passes share one serialization."""
    write_atomic(path, "".join(json.dumps(dataclasses.asdict(v)) + "\n" for v in values))


# --- the imputed path: rollouts + the ORM's scores + job D's curve -> values.jsonl --------


def submission_ok(code: str) -> bool:
    """Can CPython even ``compile()`` and load this source? Wraps
    ``checker.submission.SubmissionAnalyzer`` -- every one of its checks (S1.0-S1.3) is a
    hard failure, so any finding at all means the evaluator would score this kernel 0
    with certainty.

    Deliberately the ONLY gate `aggregate_imputed` consults before the curve. The linter
    (``checker.lint``, F1/F2) is never imported by this module and must never be: it flags
    36% of kernels and holds 28% of all correct ones, so gating on it would delete a quarter
    of the positives it is meant to score. `SubmissionAnalyzer`'s own registry holds only
    S1.* checks (never F1./F2.), so this is safe even if a caller passes it code the linter
    would also flag -- see `test_a_linter_hard_fail_is_still_imputed_normally`.
    """
    return not _SUBMISSION_ANALYZER.analyze(code, path="<generated>").findings


def aggregate_imputed(prefix, rows, scores, curve, offsets, cfg, counts) -> Value | None:
    """One prefix's rollouts, scored through the ORM curve instead of an eval verdict.

    ``scores`` is ``rollout_id -> raw ORM logit`` (pre-offset). The offset is applied INSIDE
    the lookup, ``curve.predict(score + c)``, never ``predict(score) + c`` (the second leaves
    [0, 1] entirely -- Task 4's own invariant, re-used here rather than re-derived).

    N1's null-not-zero: `n_compiled`/`n_correct`/`v_binary`/`se_binary`/`se_graded` are
    `None` here, never `0` -- this path never observes a compile or a correctness verdict.

    The S1 submission gate is a deduction, not a prediction: a kernel CPython cannot
    `compile()` scores 0.0 with certainty and never reaches the curve at all -- measured,
    1.1% of kernels, 0% of them correct. It still counts toward `n_rollouts` (it is not
    dropped, unlike TRUNCATED/NO_FINISH_REASON, which drop the rollout because it is not a
    draw from the policy at all).

    ``se_imputed = sqrt(var(t̂)/K + mean(resid_var_by_band))`` -- the residual term is
    deliberately NOT divided by K: the curve's errors on one prefix's rollouts share its text
    and style, so they do not average away like independent sampling noise would.
    """
    c = calibrate.offset_for(offsets, prefix.level, prefix.problem_id)
    ts, bands, dropped = [], [], Counter()
    # Sorted for the same reason as the measured path: `scores` is a stored column and must
    # not depend on job B's arrival order.
    for r in sorted(rows, key=lambda r: r.rollout_id):
        if r.truncation == corpus.TRUNCATED:
            # v2's rule, both paths: a stopped generation is not a draw from the policy.
            dropped[TRUNCATED] += 1
            counts[TRUNCATED] += 1
            continue
        if r.truncation != corpus.OK:
            dropped[NO_FINISH_REASON] += 1
            counts[NO_FINISH_REASON] += 1
            continue
        if not submission_ok(r.code):
            counts[S1_FAIL] += 1
            ts.append(0.0)
            bands.append(0)
            continue
        x = scores[r.rollout_id] + c
        ts.append(curve.predict(x))
        bands.append(curve.band(x))

    n = len(ts)
    if n < cfg.min_rollouts:
        counts[TOO_FEW_ROLLOUTS] += 1
        return None
    resid = sum(curve.resid_var_by_band[b] for b in bands) / n
    return Value(
        prefix_id=prefix.prefix_id,
        K_requested=prefix.K,
        n_rollouts=n,
        n_dropped={reason: dropped[reason] for reason in REASONS},
        n_dedup_shared=0,
        n_compiled=None, n_correct=None, v_binary=None,   # null, never 0 -- N1
        v_graded=sum(ts) / n,
        se_binary=None, se_graded=None,
        scores=ts,
        label_source="imputed",
        orm_offset=c,
        se_imputed=math.sqrt(statistics.pvariance(ts) / n + resid),
    )


def read_rollout_scores(out_dir: str) -> dict[str, dict]:
    """``rollout_id -> its orm_scores.jsonl/ row`` (``kind == "rollout"``), over every part
    under `orm_score.ORM_SCORES` except the anchors part -- those are v1's evaluated
    kernels, a different id space, and not this campaign's rollouts.
    """
    out: dict[str, dict] = {}
    anchors_part = f"{orm_score.ANCHORS_UNIT}.jsonl"
    for part in sorted(glob.glob(os.path.join(out_dir, orm_score.ORM_SCORES, "*.jsonl"))):
        if os.path.basename(part) == anchors_part:
            continue
        with open(part) as f:
            for line in f:
                row = json.loads(line)
                if row["kind"] == "rollout":
                    out[row["id"]] = row
    return out


def load_curve(out_dir: str) -> calibrate.Curve:
    """``curve.json``, validated beyond what `calibrate.curve_from_dict` checks (lengths and
    non-emptiness only):

    - **Ascending knots.** A non-ascending ``knots_x`` makes ``np.interp`` return garbage
      with no error, and this is the curve every imputed label trusts.
    - **A residual variance that was not silently filled with 0.0.** `calibrate.py` fills
      `resid_var_by_band` with 0.0 when no band anywhere had >= 2 points to estimate a
      residual from. 0.0 there claims a precision `se_imputed` never measured, not an
      absence of noise -- refused here rather than consumed blindly.

    Read, never re-fit (N7): this only calls `calibrate.read_curve`.
    """
    curve = calibrate.read_curve(os.path.join(out_dir, calibrate.CURVE))
    xs = list(curve.knots_x)
    if any(a > b for a, b in zip(xs, xs[1:])):
        raise ValueError(
            f"{calibrate.CURVE} knots_x is not ascending: {xs} -- np.interp silently returns "
            "garbage on an unsorted x, and this curve is what every imputed label trusts"
        )
    if curve.resid_var_by_band and all(v == 0.0 for v in curve.resid_var_by_band):
        raise ValueError(
            f"{calibrate.CURVE} resid_var_by_band is all zero -- every band had under 2 "
            "points to estimate a residual from, and 0.0 there claims a precision se_imputed "
            "never measured rather than an absence of data"
        )
    return curve


def _read_calib_manifest(out_dir: str) -> dict:
    """Job D's manifest, and the gate this module is the sole enforcer of: `calibrate.py`
    itself no longer refuses a `label_source=measured` fit (it has to run there too, for the
    level-1 validation sweep -- PLAN_v3), so this is the only machine-checkable barrier left
    against labelling a real campaign from a curve fit for validation only (N1).
    """
    path = os.path.join(out_dir, calibrate.CALIB_MANIFEST)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} is missing -- run job D (calibrate.py) before imputing values"
        )
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("label_source") != "imputed":
        raise ValueError(
            f"{calibrate.CALIB_MANIFEST} was fit with label_source="
            f"{manifest.get('label_source')!r}, not 'imputed' -- a curve fit against measured "
            "labels is a validation artifact (the level-1 sweep); it may never label a "
            "campaign (N1)"
        )
    return manifest


def _unit_of(part_path: str) -> str:
    name = os.path.basename(part_path)
    return name[: -len(".jsonl.gz")] if name.endswith(".jsonl.gz") else name


def _assert_scores_complete(by_unit: dict[str, list], scores: dict) -> None:
    """Every landed rollout unit must be fully scored before any value is written.

    `orm_score.score_unit` silently skips (writes nothing for) a unit whose job-B `.meta`
    sidecar went missing, and that silence is indistinguishable on disk from a unit that
    legitimately scored nothing -- both leave no part under `orm_score.ORM_SCORES`. On a
    1.1M-row campaign a silent completeness gap here is worse than a crash, so this checks
    every rollout of every landed unit has a score, and names exactly the units that do not.
    """
    missing = sorted(
        unit for unit, rows in by_unit.items() if any(r.rollout_id not in scores for r in rows)
    )
    if missing:
        raise ValueError(
            f"{len(missing)} rollout unit(s) are missing an ORM score for at least one of "
            f"their rollouts: {missing} -- run job B2 (orm_score.py) for them before writing "
            "any imputed value; a partial campaign must fail loudly, not report short"
        )


def _assert_checkpoint_matches(scores: dict, curve_ckpt_sha: str | None) -> None:
    """The rollout scores must have come from the same ORM checkpoint the curve was fit
    under -- two checkpoints' logits do not share a scale, and `curve.json` records which
    one it was fit for (Task 4's `orm_checkpoint_sha`).
    """
    shas = {row["orm_checkpoint_sha"] for row in scores.values()}
    if curve_ckpt_sha is not None and shas - {curve_ckpt_sha}:
        raise ValueError(
            f"rollout scores carry orm_checkpoint_sha {sorted(shas)} but the curve was fit "
            f"under {curve_ckpt_sha!r} -- two models' logits do not share a scale, so this "
            "curve cannot be applied to them"
        )


def build_values_imputed(cfg) -> dict:
    """Score a whole campaign off job D's curve instead of an eval verdict. CPU only, and it
    reads no v1 label and no eval shard -- job B's rollouts and job B2's ORM scores are the
    whole input, plus job D's `curve.json`/`offsets.json` (read, never re-fit -- N7).
    """
    rollout_cfg = cfg.prm_rollout
    rollout_cfg.validate()
    out_dir = _resolve(rollout_cfg.out_dir)

    calib_manifest = _read_calib_manifest(out_dir)   # gate (b): refuses a non-imputed fit
    curve = load_curve(out_dir)                      # validated: ascending, resid_var (c, d)
    offsets, _offsets_meta = calibrate.read_offsets(os.path.join(out_dir, calibrate.OFFSETS))

    prefix_rows = stage.read_prefixes(os.path.join(out_dir, prefixes.PREFIXES))
    parts = sorted(glob.glob(os.path.join(out_dir, stage.ROLLOUTS, "*.jsonl.gz")))
    by_unit: dict[str, list] = {}
    rows = []
    for part in parts:
        part_rows = stage.read_rollouts(part)
        by_unit[_unit_of(part)] = part_rows
        rows.extend(part_rows)

    raw_scores = read_rollout_scores(out_dir)
    _assert_scores_complete(by_unit, raw_scores)      # gate (a): every landed unit is scored
    _assert_checkpoint_matches(
        raw_scores, calib_manifest.get("anchors", {}).get("orm_checkpoint_sha")
    )
    scores = {rid: float(row["orm_score"]) for rid, row in raw_scores.items()}

    by_prefix = defaultdict(list)
    for r in rows:
        by_prefix[r.prefix_id].append(r)

    counts: Counter = Counter()
    imputed = []
    for prefix in prefix_rows:
        mine = by_prefix.get(prefix.prefix_id)
        if not mine:
            counts[NO_ROLLOUTS] += 1
            continue
        value = aggregate_imputed(prefix, mine, scores, curve, offsets, rollout_cfg, counts)
        if value is not None:
            imputed.append(value)

    write_values(imputed, os.path.join(out_dir, VALUES))
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": dataclasses.asdict(rollout_cfg),
        "label_source": "imputed",
        "parts": len(parts),
        "prefixes": len(prefix_rows),
        "rollouts": len(rows),
        "scored": len(scores),
        "values": len(imputed),
        "shared": 0,
        "dropped": {reason: counts[reason] for reason in LEDGER},
        # N7: what this build actually applied. A relabel under a different curve is visible
        # here without re-hashing anything.
        "curve_sha1": calib_manifest.get("curve_sha1"),
        "offsets_sha1": calib_manifest.get("offsets_sha1"),
    }
    write_atomic(os.path.join(out_dir, VALUES_MANIFEST), json.dumps(manifest, indent=2))
    return manifest


# --- the pass: prefixes + rollouts + every eval shard -> values.jsonl --------------------


def build_values_measured(cfg) -> dict:
    """Score a whole campaign off its own evals. CPU only, minutes long, and it reads no v1
    label anywhere.
    """
    rollout_cfg = cfg.prm_rollout
    rollout_cfg.validate()
    out_dir = _resolve(rollout_cfg.out_dir)

    prefix_rows = stage.read_prefixes(os.path.join(out_dir, prefixes.PREFIXES))
    with open(os.path.join(out_dir, stage.ROLLOUT_MAP)) as f:
        rollout_map = json.load(f)
    parts = sorted(glob.glob(os.path.join(out_dir, stage.ROLLOUTS, "*.jsonl.gz")))
    rows = [r for part in parts for r in stage.read_rollouts(part)]
    verdicts = read_verdicts(_resolve(rollout_cfg.eval_runs_dir), rollout_cfg.eval_run_name)
    joined = join(rows, rollout_map, verdicts)
    baselines = load_baseline_times(_resolve(rollout_cfg.baseline_timing_json))

    by_prefix = defaultdict(list)
    for r in rows:
        by_prefix[r.prefix_id].append(r)

    counts: Counter = Counter()
    measured = []
    for prefix in prefix_rows:
        mine = by_prefix.get(prefix.prefix_id)
        if not mine:
            # Job B is resumable per unit, so a campaign scored before every unit has run
            # is a normal state -- counted, and legible against `prefixes` in the manifest.
            counts[NO_ROLLOUTS] += 1
            continue
        value = aggregate_measured(prefix, mine, joined, baselines, rollout_cfg, counts)
        if value is not None:
            measured.append(value)

    write_values(measured, os.path.join(out_dir, VALUES))
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": dataclasses.asdict(rollout_cfg),
        "label_source": "measured",
        "parts": len(parts),
        # Read against each other: prefixes - values is what the ledger below has to explain.
        "prefixes": len(prefix_rows),
        "rollouts": len(rows),
        "evals": len(verdicts),
        "values": len(measured),
        "shared": sum(v.n_dedup_shared for v in measured),
        "dropped": {reason: counts[reason] for reason in LEDGER},
    }
    write_atomic(os.path.join(out_dir, VALUES_MANIFEST), json.dumps(manifest, indent=2))
    return manifest


def build_values(cfg) -> dict:
    """Score a whole campaign, measured or imputed per ``prm_rollout.label_source`` (N1: a
    campaign is always wholly one or the other, never a mix).
    """
    if cfg.prm_rollout.label_source == "imputed":
        return build_values_imputed(cfg)
    return build_values_measured(cfg)


def main(argv=None) -> None:
    manifest = build_values(load_config(None if argv is None else list(argv)))
    fired = {r: n for r, n in manifest["dropped"].items() if n}
    verb = "Imputed" if manifest.get("label_source") == "imputed" else "Measured"
    against = f" against {manifest['evals']} evals" if "evals" in manifest else ""
    print(
        f"{verb} {manifest['values']} of {manifest['prefixes']} prefixes from "
        f"{manifest['rollouts']} rollouts{against}"
    )
    print(f"  dropped {fired or 'nothing'}")


def _runtimes(verdict: dict) -> dict[str, float | None]:
    """The harness's two timings under the names `targets.speedups` grades them by.

    ``runtime`` is the mean and is what `build_dataset` grades, so v1 and v2 agree; the
    mapping is corpus._row's, restated rather than imported because that one reads a v1
    attempt record and this reads an eval entry.
    """
    stats = verdict.get("runtime_stats") or {}
    return {MEAN: _num(verdict.get("runtime")), MIN: _num(stats.get("min"))}


def _num(value) -> float | None:
    return None if value is None else float(value)


if __name__ == "__main__":
    main()
