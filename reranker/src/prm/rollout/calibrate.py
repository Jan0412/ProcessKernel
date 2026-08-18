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
        out.append(Anchor(s["id"], level, problem_id, float(s["orm_score"]), target, is_seen))
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


def main(argv=None) -> None:
    manifest = calibrate(load_config(None if argv is None else list(argv)))
    lines, warn = summary(manifest)
    for line in lines:
        print(line)
    for line in warn:
        print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
