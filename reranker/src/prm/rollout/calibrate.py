"""Job D: ORM scores on evaluated anchors -> a score->value curve plus one offset per problem.

Written once per campaign and sha'd into the manifest (N7). Task 6 reads `curve.json` and
`offsets.json` and never re-fits: `curve.predict(orm_score + offset_for(...))`.

Pure CPU arithmetic over orm_score.py's output, deliberately: re-fitting the curve must not
need a GPU allocation or the ORM checkpoint.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterator

import numpy as np
from sklearn.isotonic import IsotonicRegression

from reranker.src.config import _resolve, load_config
from reranker.src.prm import build, targets
from reranker.src.prm.rollout import orm_score

CURVE, OFFSETS = "curve.json", "offsets.json"
CALIB_MANIFEST = "calibrate_manifest.json"

TOL = 0.01          # offset units -- the alternating fit's convergence test
MIN_BAND = 2        # below this a band cannot estimate a residual variance of its own
NO_INFO_VAR = 1e6   # the variance of an offset the curve carries no information about
MIN_INFORMATIVE_FRAC = 0.1   # below this share of problems informing tau2, main warns


@dataclasses.dataclass(frozen=True)
class Anchor:
    """One evaluated kernel: what the ORM said, and what the eval measured."""

    key: str            # orm_score's anchor id: run__shard__rN__stem
    level: int
    problem_id: int
    score: float        # the raw ORM logit, uncorrected
    target: float       # the re-graded v1 target, [0, 1]
    orm_seen: bool      # in the ORM's own training lists -- excluded from the CURVE fit only
    # Defaulted so every existing positional Anchor(...) construction still works. Added for
    # validate()'s _length_control (Task 5): the ORM's pair accuracy has to be checked against
    # code length, and length lives on the score row, not anywhere else this module carries.
    n_code_tokens: int = 0

    @property
    def pkey(self) -> str:
        return f"{self.level}:{self.problem_id}"


def anchor_target(row: dict, conf) -> float | None:
    """Re-grade a v1 row at calibrate time. No rebuild: the row stores the speedup ratio.

    `None` is a drop the caller counts (a correct kernel with no usable speedup), never 0.0 --
    0.0 is the grade of a wrong kernel and graded_target maps only correct ones.
    """
    if not row["correct"]:
        return 0.0
    s = row["speedup_min"] if conf.speedup_stat == targets.MIN else row["speedup"]
    if not targets.usable(s):
        return None
    return targets.graded_target(s, lo=conf.speedup_lo, hi=conf.speedup_hi,
                                 quant=conf.calib_speed_quant)


def anchor_key(row: dict) -> str:
    """The id orm_score.anchor_item writes for this row -- the join key, pinned by a test."""
    return f"{row['run_name']}__{row['shard']}__r{row['round']}__{row['stem']}"


def orm_seen_key(row: dict) -> tuple[str, int, int]:
    """A v1 row -> the key the ORM's own lists name it by.

    build_dataset names a source row by its run *directory*, which for these corpora is one
    unit: "<run>__<shard>__round<N>". A v1 row keeps the three apart, so matching on its bare
    run_name would find nothing and mark every memorized anchor fresh.
    """
    return (f"{row['run_name']}__{row['shard']}__round{row['round']}",
            int(row["problem_id"]), int(row["sample_id"]))


def _expand_braces(pattern: str) -> list[str]:
    """`a{x,y}b` -> [axb, ayb]. glob does no brace expansion and the configs ship one."""
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    out: list[str] = []
    for alt in m.group(1).split(","):
        out += _expand_braces(pattern[:m.start()] + alt + pattern[m.end():])
    return out


def load_orm_seen(glob_pattern: str) -> set[tuple[str, int, int]]:
    """The ORM's training lists -> the (unit, problem_id, sample_id) keys it memorized.

    Raises rather than returning an empty set on a pattern that matches nothing: silence here
    fits the curve on the kernels the ORM was trained on, which reads every rollout hot.
    """
    paths = sorted({p for pat in _expand_braces(glob_pattern) for p in glob.glob(_resolve(pat))})
    if not paths:
        raise FileNotFoundError(
            f"orm_lists_glob matched no file: {glob_pattern!r} -- an empty contamination set "
            "would fit the curve on the kernels the ORM memorized"
        )
    seen: set[tuple[str, int, int]] = set()
    for path in paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                for cand in row["candidates"]:
                    seen.add((cand["run_name"], int(row["problem_id"]), int(cand["sample_id"])))
    return seen


# --- the curve ----------------------------------------------------------------------------


class Curve:
    """Monotone ORM score -> value lookup, linear between knots.

    `predict` takes an **offset-corrected** score: the knots are in corrected units, so it is
    `predict(score + c)` and never `predict(score) + c`, which would leave [0, 1].
    """

    def __init__(self, knots_x, knots_y, resid_var_by_band, n_fit, n_by_band=None):
        self.knots_x = np.asarray(knots_x, dtype=float)
        self.knots_y = np.asarray(knots_y, dtype=float)
        self.resid_var_by_band = [float(v) for v in resid_var_by_band]
        self.n_by_band = [int(n) for n in (n_by_band or [])]
        self.n_fit, self.n_clipped = int(n_fit), 0
        if self.knots_x.size == 0:
            raise ValueError("a curve needs at least one knot")
        if not (self.knots_x.size == self.knots_y.size == len(self.resid_var_by_band)):
            raise ValueError(
                f"curve arrays disagree: {self.knots_x.size} x, {self.knots_y.size} y, "
                f"{len(self.resid_var_by_band)} band variances -- band() indexes the "
                "variances off the knots"
            )

    def predict(self, x: float) -> float:
        if x <= self.knots_x[0] or x >= self.knots_x[-1]:
            self.n_clipped += 1
        return float(np.interp(x, self.knots_x, self.knots_y))

    def band(self, x: float) -> int:
        """Which band's residual variance describes `x`. Clamped, so every band is reachable."""
        i = int(np.searchsorted(self.knots_x, x, side="right")) - 1
        return int(np.clip(i, 0, self.knots_x.size - 1))

    def to_dict(self) -> dict:
        return {"knots_x": [float(v) for v in self.knots_x],
                "knots_y": [float(v) for v in self.knots_y],
                "resid_var_by_band": self.resid_var_by_band,
                "n_by_band": self.n_by_band,
                "n_fit": self.n_fit}


def curve_from_dict(d: dict) -> Curve:
    return Curve(d["knots_x"], d["knots_y"], d["resid_var_by_band"], d["n_fit"],
                 d.get("n_by_band"))


def read_curve(path: str) -> Curve:
    with open(path) as f:
        return curve_from_dict(json.load(f))


def fit_isotonic(anchors, offsets: dict, conf) -> Curve:
    """Equal-count bands over OFFSET-CORRECTED scores, band means, then PAVA via sklearn.

    `offsets` is pkey -> c, a plain float map (not fit_offsets' rich dict). Fit on the
    ORM-unseen anchors only: a memorized score is sharper than a fresh one, so a curve fit
    through them would read every rollout hot. Monotonicity is enforced rather than hoped
    for -- it is the ORM's only claim, and a sparse band that inverts by chance would
    otherwise assert something the ORM never said.
    """
    fit = [a for a in anchors if not a.orm_seen]
    if not fit:
        raise ValueError(
            f"no ORM-unseen anchors to fit the curve on ({len(anchors)} anchors, every one of "
            "them in the ORM's training lists) -- fitting on memorized scores is what "
            "orm_lists_glob exists to prevent"
        )
    xs = np.array([a.score + offsets.get(a.pkey, 0.0) for a in fit], dtype=float)
    ys = np.array([a.target for a in fit], dtype=float)
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    bins = max(1, int(conf.curve_bins))
    edges = np.quantile(xs, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges[1:-1], xs, side="right"), 0, bins - 1)
    cx, cy, groups = [], [], []
    for b in range(bins):
        m = idx == b
        if not m.any():        # ties collapse quantile edges; an empty band is not a knot
            continue
        cx.append(float(xs[m].mean()))
        cy.append(float(ys[m].mean()))
        groups.append(ys[m])
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(cx, cy)
    ky = [float(v) for v in iso.predict(cx)]
    # Residual about the FITTED knot, not the band mean: where PAVA pooled an inverting band
    # the two differ, and the band's own spread would understate the error exactly there.
    rv = [float(np.mean((g - y) ** 2)) if g.size >= MIN_BAND else None for g, y in zip(groups, ky)]
    pooled = [v for v in rv if v is not None]
    fill = float(np.mean(pooled)) if pooled else 0.0
    return Curve(cx, ky, [fill if v is None else v for v in rv], len(fit),
                 [int(g.size) for g in groups])


# --- the offsets ---------------------------------------------------------------------------


def _solve(anchors, curve: Curve, clamp: float) -> tuple[float, bool]:
    """The c that matches this problem's predicted sum to its measured one, and whether it
    hit the clamp. ĝ is monotone, so f is monotone in c and bisection cannot miss a root."""
    want = sum(a.target for a in anchors)

    def f(c: float) -> float:
        return sum(curve.predict(a.score + c) for a in anchors) - want

    lo, hi = -clamp, clamp
    # No root inside the bracket: the measured sum is outside what the curve can reach.
    if f(lo) >= 0:
        return lo, True
    if f(hi) <= 0:
        return hi, True
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, False


def eb_shrink(c_hat: dict, var: dict, kappa="auto", n_anchors: dict | None = None,
              informative=None) -> dict:
    """Per-problem shrink factors toward c = 0. `auto` is empirical Bayes, tau2/(tau2+var_i).

    tau2 is estimated over `informative` only -- by default the problems whose variance is
    finite. One no-information problem carries var = NO_INFO_VAR = 1e6, and averaging that
    into the moment estimator drives tau2 to its floor and every shrink to 0, which silently
    turns the whole campaign back into the pooled fit v3 exists to avoid. Every problem is
    still *shrunk*, informative or not; only the estimate of the spread is restricted.
    """
    # Before `core`, which reads `var`: a caller who forgot n_anchors must hear about that
    # rather than about whichever key of `var` happens to be missing.
    if kappa != "auto" and n_anchors is None:
        raise ValueError("a fixed offset_kappa needs n_anchors: the form is n/(n+kappa)")
    core = {p for p in c_hat if (p in informative if informative is not None
                                 else var[p] < NO_INFO_VAR)}
    if kappa != "auto":
        k = float(kappa)
        return {"tau2": None, "tau2_raw": None, "tau2_estimable": False,
                "n_informative": len(core),
                "shrink": {p: n_anchors[p] / (n_anchors[p] + k) for p in c_hat}}
    if len(core) < 2:
        # Fewer than two usable ĉ is no ensemble to borrow strength from. The one problem (if
        # any) that IS identified keeps its solved offset -- Var over a single value is 0 by
        # construction, not by evidence, and shrinking on it would discard the only offset
        # measured. Every other problem here is censored at the clamp or unidentified, so its
        # ĉ is not a measurement: it goes to 0 rather than being applied at full magnitude,
        # which would push ±offset_clamp into the labels of exactly the problems just
        # classified as carrying no information. `tau2_estimable` makes the state visible.
        return {"tau2": None, "tau2_raw": None, "tau2_estimable": False,
                "n_informative": len(core),
                "shrink": {p: (1.0 if p in core else 0.0) for p in c_hat}}
    vals = np.array([c_hat[p] for p in sorted(core)], dtype=float)
    tau2_raw = float(vals.var() - np.mean([var[p] for p in sorted(core)]))
    # Floored: the spread between problems can come out below the noise inside them, which
    # means no real spread, not a negative one. The pre-floor value is kept, because a
    # floored 0.0 and a measured 0.0 mean very different things.
    tau2 = max(0.0, tau2_raw)
    return {"tau2": tau2, "tau2_raw": tau2_raw, "tau2_estimable": True,
            "n_informative": len(core),
            "shrink": {p: (tau2 / (tau2 + var[p]) if tau2 + var[p] > 0 else 0.0) for p in c_hat}}


def _no_offsets_meta(conf) -> dict:
    """The meta of a fit that produced no offsets. Same keys as a real one, so every reader
    (main's summary, the manifest, Task 5's ablation) sees one shape."""
    return {"tau2": None, "tau2_raw": None, "tau2_estimable": False,
            "kappa_mode": str(conf.offset_kappa), "c_std": 0.0, "c_std_core": 0.0,
            "n_clamped": 0, "n_no_anchors": 0, "n_no_info": 0, "n_informative": 0,
            "frac_informative": 0.0, "shrink_mean": 0.0, "n_problems": 0}


def fit_offsets(anchors, curve: Curve, conf) -> tuple[dict, dict]:
    """One offset per problem, moment-matched against `curve` then shrunk.

    Uses **all** of a problem's anchors, memorized ones included: the contamination bias
    points the other way here and partly cancels.
    """
    by_p: dict[str, list] = {}
    for a in anchors:
        by_p.setdefault(a.pkey, []).append(a)
    if not by_p:
        return {}, _no_offsets_meta(conf)
    raw, var, n, clamped = {}, {}, {}, {}
    for p, group in by_p.items():
        c, hit = _solve(group, curve, conf.offset_clamp)
        raw[p], clamped[p], n[p] = c, hit, len(group)
        # A logistic-scale proxy for Var(ĉ), not the exact delta-method variance: anchors the
        # curve reads as certain (q at 0 or 1) say nothing about where the problem sits.
        info = sum((q := curve.predict(a.score + c)) * (1 - q) for a in group)
        var[p] = 1.0 / info if info > 0 else NO_INFO_VAR
    # What tau2 is measured on. A clamped ĉ sits at the bracket end rather than at its own
    # value, so it would widen the spread on nothing; a no-information ĉ is not a measurement
    # at all. Both are still shrunk with everyone else, just not used to estimate the spread.
    core = {p for p in raw if var[p] < NO_INFO_VAR and not clamped[p]}
    eb = eb_shrink(raw, var, conf.offset_kappa, n, informative=core)
    out = {p: {"c": raw[p] * eb["shrink"][p], "raw": raw[p], "var": var[p],
               "shrink": eb["shrink"][p], "n_anchors": n[p], "clamped": clamped[p]}
           for p in raw}
    unclamped = [o["c"] for o in out.values() if not o["clamped"]]
    n_no_info = sum(1 for p in raw if var[p] >= NO_INFO_VAR)
    meta = {"tau2": eb["tau2"], "tau2_raw": eb["tau2_raw"],
            "tau2_estimable": eb["tau2_estimable"], "kappa_mode": str(conf.offset_kappa),
            "c_std": float(np.std([o["c"] for o in out.values()])),
            "c_std_core": float(np.std(unclamped)) if unclamped else 0.0,
            "n_clamped": sum(clamped.values()),
            # Dead by construction: a problem is only known here through its anchors, so one
            # with none never reaches offsets.json. offset_for gives it 0.0.
            "n_no_anchors": 0,
            "n_no_info": n_no_info, "n_informative": eb["n_informative"],
            "frac_informative": eb["n_informative"] / len(out),
            "shrink_mean": float(np.mean([o["shrink"] for o in out.values()])),
            "n_problems": len(out)}
    return out, meta


def offset_for(offsets: dict, level: int, problem_id: int) -> float:
    """A problem with no anchors has no offset, which is 0.0 rather than a KeyError."""
    return offsets.get(f"{level}:{problem_id}", {}).get("c", 0.0)


def read_offsets(path: str) -> tuple[dict, dict]:
    with open(path) as f:
        blob = json.load(f)
    return blob["offsets"], blob["meta"]


def fit_joint(anchors, conf) -> tuple[Curve, dict, dict]:
    """One pooled fit averages across the offset spread and flattens the curve (PLAN_v3 §2).

    The returned offsets were solved against the returned curve, so their composition is what
    is moment-matched -- the curve is deliberately not refit after the last offset solve.

    `c_i` is difficulty plus the ORM's arbitrary per-problem drift, confounded, and nothing
    downstream needs to tell them apart. Do not report it as "problem difficulty".
    """
    c: dict[str, float] = {}
    curve, meta, new = None, {}, {}
    knots, delta_k = None, None
    for it in range(max(1, conf.curve_iters)):
        curve = fit_isotonic(anchors, c, conf)
        # Diagnostic only, and on the knot *values*: convergence is tested on the offsets,
        # which are what move the knots. None where the two fits are not comparable.
        delta_k = None if knots is None or len(knots) != len(curve.knots_y) else float(
            np.max(np.abs(curve.knots_y - knots)))
        knots = curve.knots_y
        if not conf.use_anchors:
            # The offsets-off ablation. Full meta, not a short one: main() and the manifest
            # read the same keys on both paths.
            return curve, {}, {**_no_offsets_meta(conf), "n_iters": it + 1, "converged": True,
                               "max_offset_delta": 0.0, "max_knot_delta": delta_k}
        new, meta = fit_offsets(anchors, curve, conf)
        delta = max((abs(new[p]["c"] - c.get(p, 0.0)) for p in new), default=0.0)
        c = {p: new[p]["c"] for p in new}
        meta |= {"n_iters": it + 1, "converged": delta < TOL, "max_offset_delta": float(delta),
                 "max_knot_delta": delta_k}
        if delta < TOL:
            break
    return curve, new, meta


# --- the anchors on disk ---------------------------------------------------------------------


def iter_anchor_rows(cfg) -> Iterator[dict]:
    """The v1 rows this campaign anchors on -- the same run_tags x anchor_rounds selection
    orm_score.iter_anchor_items makes, pinned to it by a test.

    Not gated on `use_anchors`, unlike orm_score's: there the flag decides whether anchors are
    ever *scored*, here it decides only whether per-problem offsets are fit. The offsets-off
    ablation re-fits an already-scored campaign, and gating here would leave it with no rows
    to join and no curve at all.
    """
    conf = cfg.prm_rollout
    for part in sorted(glob.glob(_resolve(conf.parts_glob))):
        with open(part) as f:
            for line in f:
                row = json.loads(line)
                if row["run_name"] in conf.run_tags and row["round"] in conf.anchor_rounds:
                    yield row


def read_anchor_scores(out_dir: str) -> list[dict]:
    """The `_anchors` part of ORM_SCORES. One file, so no glob: the rollout parts beside it
    are the 1.1M rows this job must not read."""
    path = orm_score.unit_score_path(out_dir, orm_score.ANCHORS_UNIT)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} is missing -- job B2 has not scored the anchors yet, and a curve fit on "
            "nothing is worse than no curve"
        )
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r["kind"] == "anchor"]
    shas = {r["orm_checkpoint_sha"] for r in rows}
    if len(shas) > 1:
        raise ValueError(
            f"anchors scored under {len(shas)} ORM checkpoints {sorted(shas)} -- two models' "
            "logits do not share a scale, so one curve cannot describe both"
        )
    return rows


def anchor_index(cfg) -> tuple[dict[str, tuple], int]:
    """key -> (level, problem_id, target|None, orm_seen), and how many keys the ORM memorized.

    Deliberately compact rather than the rows themselves: the v1 parts are 6 GB of `raw`, and
    this job is meant to re-fit on a CPU.
    """
    conf = cfg.prm_rollout
    seen = load_orm_seen(conf.orm_lists_glob) if conf.orm_lists_glob else set()
    idx = {anchor_key(row): (int(row["level"]), int(row["problem_id"]),
                             anchor_target(row, conf), orm_seen_key(row) in seen)
           for row in iter_anchor_rows(cfg)}
    return idx, len(seen)


def load_anchors(cfg) -> tuple[list[Anchor], dict]:
    """Anchor scores joined to their re-graded targets and their contamination flag."""
    conf = cfg.prm_rollout
    idx, n_seen_keys = anchor_index(cfg)
    scored = read_anchor_scores(_resolve(conf.out_dir))
    out: list[Anchor] = []
    ledger = {"scored": len(scored), "no_row": 0, "no_target": 0, "orm_seen": 0, "orm_unseen": 0,
              "orm_seen_keys": n_seen_keys,
              "orm_checkpoint_sha": scored[0]["orm_checkpoint_sha"] if scored else None}
    for s in scored:
        row = idx.get(s["id"])
        if row is None:
            ledger["no_row"] += 1
            continue
        level, problem_id, target, is_seen = row
        if (level, problem_id) != (s["level"], s["problem_id"]):
            raise ValueError(
                f"anchor {s['id']} is scored as {s['level']}:{s['problem_id']} but its v1 row "
                f"is {level}:{problem_id} -- the join key is not what it names"
            )
        if target is None:
            ledger["no_target"] += 1
            continue
        ledger["orm_seen" if is_seen else "orm_unseen"] += 1
        out.append(Anchor(s["id"], level, problem_id, float(s["orm_score"]), target, is_seen,
                          int(s["n_code_tokens"])))
    ledger["kept"] = len(out)
    ledger["problems"] = len({a.pkey for a in out})
    return out, ledger


# --- the job -----------------------------------------------------------------------------------


def target_dist(anchors) -> dict:
    """What this anchor set's targets look like. Reported for the fit set and for all anchors.

    The unseen set is not a random slice: a candidate is missing from the ORM's lists exactly
    when it failed to compile, was a code-hash duplicate, was correct with no usable speedup,
    or sat in a problem with no usable positive -- all of which enrich it in 0.0 targets. The
    curve is fit on that slice, so the skew has to be visible before 1.1M labels are written.
    """
    if not anchors:
        return {"n": 0, "frac_zero": 0.0, "mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    ys = np.array([a.target for a in anchors], dtype=float)
    q = np.quantile(ys, [0.25, 0.5, 0.75])
    return {"n": int(ys.size), "frac_zero": float(np.mean(ys == 0.0)), "mean": float(ys.mean()),
            "p25": float(q[0]), "p50": float(q[1]), "p75": float(q[2])}


def calibrate(cfg) -> dict:
    """Fit once, write `curve.json` and `offsets.json`, sha both into the manifest (N7).

    Runs under either `label_source`. Under `measured` it is the level-1 validation sweep --
    the curve is fit and reported against labels the campaign already measured, and nothing
    downstream is allowed to label from it; that gate belongs to the apply pass.
    """
    conf = cfg.prm_rollout
    conf.validate()
    if not conf.orm_lists_glob:
        raise ValueError(
            "calibrate needs orm_lists_glob: with no list of what the ORM trained on, the "
            "curve is fit through memorized scores and every imputed label runs hot"
        )
    out_dir = _resolve(conf.out_dir)
    anchors, ledger = load_anchors(cfg)
    curve, offsets, meta = fit_joint(anchors, conf)

    dist = {"fit": target_dist([a for a in anchors if not a.orm_seen]),
            "all": target_dist(anchors)}
    curve_path = os.path.join(out_dir, CURVE)
    offsets_path = os.path.join(out_dir, OFFSETS)
    build.write_atomic(curve_path, json.dumps(
        {**curve.to_dict(), "curve_bins": conf.curve_bins,
         "orm_checkpoint_sha": ledger["orm_checkpoint_sha"], "target_dist": dist}, indent=2))
    build.write_atomic(offsets_path, json.dumps({"meta": meta, "offsets": offsets}, indent=2))

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": dataclasses.asdict(conf),
        "label_source": conf.label_source,
        "anchors": ledger,
        "fit": meta,
        "curve": {"n_knots": len(curve.knots_x), "n_fit": curve.n_fit,
                  "y_lo": float(curve.knots_y[0]), "y_hi": float(curve.knots_y[-1]),
                  "x_lo": float(curve.knots_x[0]), "x_hi": float(curve.knots_x[-1]),
                  "n_by_band": curve.n_by_band, "target_dist": dist},
        # N7: what Task 6 must be applying. A relabel under a different curve is visible here.
        "curve_sha1": build._sha1(curve_path),
        "offsets_sha1": build._sha1(offsets_path),
    }
    build.write_atomic(os.path.join(out_dir, CALIB_MANIFEST), json.dumps(manifest, indent=2))
    return manifest


def summary(manifest: dict) -> tuple[list[str], list[str]]:
    """The log lines and the warning lines. Split out so both are testable without a config."""
    a, fit, cv = manifest["anchors"], manifest["fit"], manifest["curve"]
    fit_d, all_d = cv["target_dist"]["fit"], cv["target_dist"]["all"]
    lines = [
        f"anchors: {a['kept']} kept of {a['scored']} scored ({a['orm_seen']} memorized, "
        f"{a['orm_unseen']} fit the curve), {a['problems']} problems; dropped "
        f"{a['no_row']} no-row, {a['no_target']} no-target",
        f"curve: {cv['n_knots']} knots, y {cv['y_lo']:.3f}..{cv['y_hi']:.3f}",
        f"targets: fit set {fit_d['frac_zero']:.1%} zero (mean {fit_d['mean']:.3f}) vs all "
        f"anchors {all_d['frac_zero']:.1%} zero (mean {all_d['mean']:.3f})",
        f"offsets: tau2={fit['tau2']} (raw {fit['tau2_raw']}), "
        f"shrink_mean={fit['shrink_mean']:.3f}, c_std_core={fit['c_std_core']:.3f}, "
        f"{fit['n_clamped']} clamped, {fit['n_no_info']} no-info, "
        f"{fit['n_informative']} informative",
    ]
    warn = []
    if not fit["converged"]:
        warn.append(
            f"WARNING: the alternating fit did not converge in {fit['n_iters']} iterations "
            f"(max_offset_delta={fit['max_offset_delta']:.4f} > {TOL}) -- raise curve_iters "
            "before trusting these labels")
    if fit["n_problems"] > 1 and fit["kappa_mode"] == "auto" and not fit["tau2_estimable"]:
        # Says what eb_shrink's degrade branch actually does: every unidentified offset goes
        # to 0 and at most one survives. With none surviving the run is the pooled fit, which
        # is the one thing this warning exists to say.
        tail = ("and the one identified offset stands as solved" if fit["n_informative"]
                else "and none survives -- every offset is 0.0, so this IS the pooled fit")
        warn.append(
            f"WARNING: tau2 could not be estimated -- only {fit['n_informative']} of "
            f"{fit['n_problems']} problems are both informative and unclamped, so "
            f"{fit['n_problems'] - fit['n_informative']} offsets were zeroed as "
            f"unidentified {tail}")
    if fit["n_problems"] and fit["frac_informative"] < MIN_INFORMATIVE_FRAC:
        warn.append(
            f"WARNING: only {fit['n_informative']} of {fit['n_problems']} problems informed "
            f"tau2 ({fit['frac_informative']:.1%} < {MIN_INFORMATIVE_FRAC:.0%}) -- the "
            "between-problem spread rests on a small, non-random subset")
    if fit["tau2_raw"] is not None and fit["tau2_raw"] < 0:
        warn.append(
            f"WARNING: tau2 was floored (raw {fit['tau2_raw']:.4f} < 0), so every offset is "
            "shrunk to 0 and this is the pooled fit -- the anchors carry more noise within a "
            "problem than spread between problems")
    if manifest["label_source"] != "imputed":
        warn.append(
            f"NOTE: label_source={manifest['label_source']!r} -- this is a validation fit "
            "against measured labels; nothing may be labelled from this curve")
    return lines, warn


# --- validate: level-1 full-corpus calibration check ---------------------------------------
#
# Level 1 is 100% measured. Production, on level 6, fits the curve and every problem's offset
# on that problem's own already-evaluated anchors, then applies it to that SAME problem's
# newly generated rollouts -- the curve has seen the problems it labels, never the kernels.
# The headline here is the identical composition run on level 1: fit_joint on every level-1
# anchor (no problem split anywhere), impute every level-1 prefix from its own rollouts,
# compare to its own measured V̂. `pair_agreement` from that run is the gate (>= 0.65) on
# whether level 6 gets 1.1M imputed labels. The core-size sweep below is a separate
# DIAGNOSTIC (how many problems' anchors the CURVE needs), never the gate.

PAIR_GAP = 0.15           # validate()'s headline threshold: pairs closer than this are noise
LENGTH_DECILES = 10       # _length_control's length-conditioning granularity


@dataclasses.dataclass(frozen=True)
class Measured:
    """One level-1 prefix's ground truth, keyed by prefix_id in the `measured` map `validate`
    takes. `round` and `depth_bucket` are not in the brief's docstring ("prefix -> (list_key,
    V̂, level, pid)") but are required by its own prose ("per depth bucket and per round"),
    and by nothing else this module carries -- they have to live here.
    """

    list_key: str
    v: float               # measured V̂ (job C's own evaluation, never touches the ORM)
    level: int
    problem_id: int
    round: int
    depth_bucket: int

    @property
    def pkey(self) -> str:
        return f"{self.level}:{self.problem_id}"


def _impute_one(prefix_id: str, m: Measured, rollout_scores: dict, curve: Curve,
                offsets: dict) -> float | None:
    """A prefix's imputed V̂: mean of curve.predict(score + offset) over its own rollouts' raw
    ORM scores -- the same lookup Task 6's aggregate_imputed uses, minus the drop bookkeeping
    this check does not need. `None`, not 0.0, when the prefix has no scored rollouts: a
    missing score is a data gap, not a measurement of zero.
    """
    scores = rollout_scores.get(prefix_id)
    if not scores:
        return None
    c = offset_for(offsets, m.level, m.problem_id)
    return float(np.mean([curve.predict(s + c) for s in scores]))


def _cal_error(measured: dict, imputed: dict) -> float | None:
    diffs = [abs(imputed[p] - measured[p].v) for p in measured if p in imputed]
    return float(np.mean(diffs)) if diffs else None


def _slice(measured: dict, imputed: dict, key) -> dict:
    """cal_error by `key(m)` -- a good average hiding a bad bucket is exactly what per-round
    and per-depth-bucket reporting exists to catch. `n` rides along so a 2-point bucket is
    not read the way a 200-point one is.
    """
    by_bucket = defaultdict(list)
    for p, m in measured.items():
        if p in imputed:
            by_bucket[key(m)].append(abs(imputed[p] - m.v))
    return {str(k): {"cal_error": float(np.mean(v)), "n": len(v)}
            for k, v in sorted(by_bucket.items(), key=lambda kv: str(kv[0]))}


def _pair_agreement(measured: dict, imputed: dict, gap: float = PAIR_GAP) -> dict:
    """The headline number: of prefix pairs INSIDE ONE LIST whose measured V̂ differ by more
    than `gap`, the fraction the imputed labels order the same way. Within a list, never
    across problems -- lambdarank only ever compares two items sharing a list, so that is the
    only scale a shell label's ordering is ever asked to be right on. Strict `>` on the gap: a
    pair sitting exactly on the boundary is not evidence either way. An imputed tie (equal
    values) counts as disagreement -- it failed to preserve an order that was there to
    preserve, whatever the reason.
    """
    by_list = defaultdict(list)
    for p, m in measured.items():
        if p in imputed:
            by_list[m.list_key].append((m.v, imputed[p]))
    agree = n = 0
    for items in by_list.values():
        for i, (vi, ti) in enumerate(items):
            for vj, tj in items[i + 1:]:
                if abs(vi - vj) <= gap:
                    continue
                n += 1
                agree += (vi > vj) == (ti > tj)
    return {"pair_agreement": (agree / n) if n else None, "n_pairs": n}


def _pkey_pairs(anchors, gap: float = PAIR_GAP):
    """Within-problem anchor pairs whose measured target differs by more than `gap` -- the
    population `_length_control`'s pair-accuracy numbers are computed over. Within-problem for
    the same reason as `_pair_agreement`: a raw ORM score is not on a shared scale across
    problems (that is what the offset corrects for), so an across-problem pair is not
    comparable at all.
    """
    by_p = defaultdict(list)
    for a in anchors:
        by_p[a.pkey].append(a)
    for group in by_p.values():
        for i, ai in enumerate(group):
            for aj in group[i + 1:]:
                if abs(ai.target - aj.target) > gap:
                    yield ai, aj


def _pair_acc(pairs) -> float | None:
    pairs = list(pairs)
    if not pairs:
        return None
    return sum((a.score > b.score) == (a.target > b.target) for a, b in pairs) / len(pairs)


def _length_control(anchors, gap: float = PAIR_GAP, n_deciles: int = LENGTH_DECILES) -> dict:
    """The ORM's raw-score pair accuracy, overall and length-controlled.

    Length-controlled = restricted to pairs whose two anchors share a length decile: within a
    decile length cannot tell the pair apart, so whatever accuracy survives there is not
    length. If `decile_mean` sits well below `overall`, the overall number is mostly length
    doing the work -- the ORM is a length proxy, imputed labels built on it would inherit
    that, and the shell must not ship. This is the one diagnostic in the sweep that is not
    about the curve or the offsets at all: a perfectly calibrated curve fit on a length proxy
    is still a length proxy.
    """
    all_pairs = list(_pkey_pairs(anchors, gap))
    overall = _pair_acc(all_pairs)
    lengths = np.array([a.n_code_tokens for a in anchors], dtype=float)
    if lengths.size == 0 or lengths.max() == lengths.min():
        return {"overall": overall, "decile_mean": None, "decile_min": None, "n_deciles_used": 0}
    edges = np.quantile(lengths, np.linspace(0, 1, n_deciles + 1))
    decile_of = {a.key: int(np.clip(np.searchsorted(edges[1:-1], a.n_code_tokens, side="right"),
                                    0, n_deciles - 1))
                 for a in anchors}
    by_decile = defaultdict(list)
    for a, b in all_pairs:
        if decile_of[a.key] == decile_of[b.key]:
            by_decile[decile_of[a.key]].append((a, b))
    accs = [acc for acc in (_pair_acc(p) for p in by_decile.values()) if acc is not None]
    return {"overall": overall, "decile_mean": float(np.mean(accs)) if accs else None,
            "decile_min": float(np.min(accs)) if accs else None, "n_deciles_used": len(accs)}


def _fit_error_row(eval_problems: set, msg: str) -> dict:
    """The shape a fold/ablation reports when its fit raised. Same keys a real fit returns
    (`main`'s summary and Task 8 read one shape), every metric `None`, the reason named --
    visible in the output, never a silently shortened sweep or ablation table.
    """
    return {"eval_problems": sorted(eval_problems), "fit_error": msg,
            "cal_error": None, "cal_error_by_round": {}, "cal_error_by_depth": {},
            "pair_agreement": None, "n_pairs": 0, "n_clipped": None, "tau2": None,
            "clamp_rate": None, "no_anchor_rate": None, "n_missing_scores": None,
            "c_std": None, "converged": False, "max_knot_delta": None}


def _measure_fit(curve: Curve, offsets: dict, meta: dict, rollout_scores: dict,
                 measured: dict) -> dict:
    """Impute every prefix in `measured` from `rollout_scores` against an already-fitted
    `curve`/`offsets`, and report every diagnostic against the ground truth. Pure comparison,
    agnostic to how the fit was produced -- shared by the full-corpus headline/ablations and
    the core-size sweep's curve-subset diagnostic, so both report off the identical machinery.

    `curve.n_clipped` is a live counter that also incremented on every `_solve` bisection call
    inside `fit_offsets` -- one problem's offset solve alone can be dozens of predict() calls,
    many of them out of range while the search still brackets. Snapshotting it here, before
    this function's own impute loop runs, is what keeps the reported figure the imputation
    signal Task 7 gates on (rollouts landing outside the curve's fitted range) rather than
    solver noise from fitting.
    """
    clipped_before = curve.n_clipped
    imputed: dict = {}
    n_missing = 0
    for p, m in measured.items():
        v = _impute_one(p, m, rollout_scores, curve, offsets)
        if v is None:
            n_missing += 1
        else:
            imputed[p] = v
    eval_problems = {m.pkey for m in measured.values()}
    no_anchor = eval_problems - set(offsets)
    return {
        "eval_problems": sorted(eval_problems), "fit_error": None,
        "cal_error": _cal_error(measured, imputed),
        "cal_error_by_round": _slice(measured, imputed, key=lambda m: m.round),
        "cal_error_by_depth": _slice(measured, imputed, key=lambda m: m.depth_bucket),
        **_pair_agreement(measured, imputed),
        "n_clipped": curve.n_clipped - clipped_before, "tau2": meta.get("tau2"),
        "clamp_rate": meta.get("n_clamped", 0) / max(len(offsets), 1),
        "no_anchor_rate": (len(no_anchor) / len(eval_problems)) if eval_problems else None,
        "n_missing_scores": n_missing,
        "c_std": meta.get("c_std_core"), "converged": meta.get("converged"),
        "max_knot_delta": meta.get("max_knot_delta"),
    }


def _fold(anchors, rollout_scores, measured, cfg) -> dict:
    """The full-corpus headline, or an ablation of it: fit_joint on every one of `anchors` --
    no problem split -- then impute and compare every prefix in `measured`. This is exactly
    production's own composition (curve and every problem's offset from the SAME anchor set),
    which is why it is the number the gate reads, not the core-size sweep below.
    """
    try:
        curve, offsets, meta = fit_joint(anchors, cfg)
    except ValueError as e:
        return _fit_error_row({m.pkey for m in measured.values()}, str(e))
    return _measure_fit(curve, offsets, meta, rollout_scores, measured)


def _fit_curve_then_offsets(curve_anchors, offset_anchors, cfg) -> tuple[Curve, dict, dict]:
    """`fit_joint`'s own alternation (curve <-> offsets), decoupled: the curve is refit on
    `curve_anchors` every iteration, offsets on `offset_anchors` -- so a problem can inform
    the curve, the offsets, both or neither, independently. `fit_joint` takes one anchor set
    and cannot express that, and it is not to be changed for this; this composes the same two
    calls it makes (`fit_isotonic`, `fit_offsets`, both untouched) over two different sets.

    Mirrors `fit_joint`'s `use_anchors=False` early return too: today every caller (the
    core-size sweep, its stability re-draws) always passes a `cfg` with `use_anchors=True`, so
    this branch is unreached, but the two functions are meant to do the same thing and a
    silent divergence here would be a trap for whoever reuses this with a different `cfg`.
    """
    c: dict = {}
    curve, meta, new = None, {}, {}
    knots, delta_k = None, None
    for it in range(max(1, cfg.curve_iters)):
        curve = fit_isotonic(curve_anchors, c, cfg)
        delta_k = None if knots is None or len(knots) != len(curve.knots_y) else float(
            np.max(np.abs(curve.knots_y - knots)))
        knots = curve.knots_y
        if not cfg.use_anchors:
            return curve, {}, {**_no_offsets_meta(cfg), "n_iters": it + 1, "converged": True,
                               "max_offset_delta": 0.0, "max_knot_delta": delta_k}
        new, meta = fit_offsets(offset_anchors, curve, cfg)
        delta = max((abs(new[p]["c"] - c.get(p, 0.0)) for p in new), default=0.0)
        c = {p: new[p]["c"] for p in new}
        meta |= {"n_iters": it + 1, "converged": delta < TOL, "max_offset_delta": float(delta),
                 "max_knot_delta": delta_k}
        if delta < TOL:
            break
    return curve, new, meta


def _curve_subset_fold(curve_anchors, offset_anchors, rollout_scores, measured, cfg) -> dict:
    """One core-size sweep row: the CURVE is fit on `curve_anchors` (a subset of problems),
    but every evaluated problem still gets its own offset from `offset_anchors` (every
    problem's own anchors) and evaluation still covers every prefix in `measured` -- never a
    held-out problem. Answers "how few problems' anchors does the curve need", not "does the
    method generalize to an unseen problem" (production never sees one: every level-6 problem
    contributes anchors from its own already-evaluated v1 candidates).
    """
    try:
        curve, offsets, meta = _fit_curve_then_offsets(curve_anchors, offset_anchors, cfg)
    except ValueError as e:
        return _fit_error_row({m.pkey for m in measured.values()}, str(e))
    return _measure_fit(curve, offsets, meta, rollout_scores, measured)


def _stability(anchors, rollout_scores, measured, cfg, n: int, seed: int, n_reps: int) -> dict:
    """Independent re-draws of the CURVE's fit subset at this size (offsets, as always, use
    every problem's own anchors): is the diagnostic's number the population's, or one core
    sample's luck? Spawned off (seed, n) via numpy's SeedSequence so every core_size gets its
    own reproducible substream, independent of how many sizes ran before it in the sweep.
    """
    problems = sorted({a.pkey for a in anchors})
    cals, agrees = [], []
    for child in np.random.SeedSequence([seed, n]).spawn(n_reps):
        rng = np.random.default_rng(child)
        core = set(rng.choice(problems, size=min(n, len(problems)), replace=False).tolist())
        curve_anchors = [a for a in anchors if a.pkey in core]
        fold = _curve_subset_fold(curve_anchors, anchors, rollout_scores, measured, cfg)
        if fold["cal_error"] is not None:
            cals.append(fold["cal_error"])
        if fold["pair_agreement"] is not None:
            agrees.append(fold["pair_agreement"])
    return {"n_reps": n_reps, "n_usable": len(cals),
            "cal_error_mean": float(np.mean(cals)) if cals else None,
            "cal_error_std": float(np.std(cals)) if cals else None,
            "pair_agreement_mean": float(np.mean(agrees)) if agrees else None,
            "pair_agreement_std": float(np.std(agrees)) if agrees else None}


def validate(anchors, rollout_scores, measured, cfg, core_sizes=(4, 8, 12, 20, 40), seed=42,
             n_reps=3) -> dict:
    """The headline: fit_joint on EVERY level-1 anchor (no problem split -- production's own
    composition, run here against measured truth instead of nothing), impute every level-1
    prefix from its own rollouts, compare to its own measured V̂. `pair_agreement` from this
    run is the number gated at >= 0.65; `cal_error` (overall, by round, by depth) is reported
    from the same run. On level 1 every anchor is ORM-unseen -- the ORM trained on level 6
    only -- so `fit_isotonic`'s contamination filter is a no-op here, unlike level 6 where
    only a fraction of anchors are clean.

    The core-size sweep is a separate DIAGNOSTIC, not the gate: how few problems' anchors does
    the CURVE need before it stops improving, while every evaluated problem still gets its own
    offset from its own anchors and evaluation always covers every level-1 prefix -- never a
    held-out one, since production never holds one out either.

    `cfg` is `prm_rollout` (matches `fit_joint`'s `conf`, not the whole `RerankerConfig`).
    `measured` maps prefix_id -> `Measured`; `rollout_scores` maps the same prefix_id -> the
    raw ORM scores of that prefix's own rollouts.
    """
    headline = _fold(anchors, rollout_scores, measured, cfg)
    rng = np.random.default_rng(seed)
    problems = sorted({a.pkey for a in anchors})
    sweep: list[dict] = []
    for n in core_sizes:
        core = set(rng.choice(problems, size=min(n, len(problems)), replace=False).tolist())
        curve_anchors = [a for a in anchors if a.pkey in core]
        fold = _curve_subset_fold(curve_anchors, anchors, rollout_scores, measured, cfg)
        fold["core_size"] = len(core)
        fold["core_problems"] = sorted(core)
        fold["stability"] = _stability(anchors, rollout_scores, measured, cfg, n, seed, n_reps)
        sweep.append(fold)
    ablations = {
        name: _fold(anchors, rollout_scores, measured, dataclasses.replace(cfg, **kw))
        for name, kw in (("no_offset", {"use_anchors": False}),
                         ("curve_iters_1", {"curve_iters": 1}),
                         ("fixed_kappa_8", {"offset_kappa": 8.0}))
    }
    return {**headline, "sweep": sweep, "ablations": ablations,
            "orm_vs_length": _length_control(anchors)}


def main(argv=None) -> None:
    manifest = calibrate(load_config(None if argv is None else list(argv)))
    lines, warn = summary(manifest)
    for line in lines:
        print(line)
    for line in warn:
        print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
