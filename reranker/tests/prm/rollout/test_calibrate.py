import json
import math
import os

import numpy as np
import pytest

from reranker.src.config import PRMRolloutConfig, RerankerConfig
from reranker.src.encoding import SequenceEncoder
from reranker.src.prm import build
from reranker.src.prm.rollout import calibrate, orm_score


def _cfg(**over) -> PRMRolloutConfig:
    conf = PRMRolloutConfig(run_tags={"a_run": "ar"})
    for k, v in over.items():
        setattr(conf, k, v)
    return conf


def _anchors(pairs, pid=1, seen=False):
    return [calibrate.Anchor(f"k{i}", 6, pid, s, t, seen) for i, (s, t) in enumerate(pairs)]


def _row(run="a_run", shard="shard_00", rnd=0, level=6, pid=1, sid=0, correct=True,
         speedup_min=2.0, speedup=2.0):
    """One v1 part row, carrying exactly the fields build.py writes that job D reads."""
    return {"run_name": run, "shard": shard, "round": rnd, "level": level, "problem_id": pid,
            "sample_id": sid, "stem": f"level_{level}_problem_{pid}_sample_{sid}_kernel",
            "raw": "```python\nimport torch\n```", "compiled": True, "correct": correct,
            "speedup": speedup, "speedup_min": speedup_min}


def _score_row(row, score, sha="ckpt"):
    return {"kind": "anchor", "id": orm_score.anchor_item(row).id, "level": row["level"],
            "problem_id": row["problem_id"], "code_sha1": "a" * 40, "orm_score": score,
            "orm_checkpoint_sha": sha, "n_code_tokens": 3}


def _ramp(bins=11):
    """A curve that is exactly y = x/10 on x in [0, 10] -- every offset below is hand-solved."""
    return calibrate.fit_isotonic(_anchors([(x, x / 10) for x in range(11)]), {},
                                  _cfg(curve_bins=bins))


# --- the curve ---------------------------------------------------------------------------


def test_isotonic_is_monotone_on_an_inverting_band():
    a = _anchors([(0, 0.0), (1, 0.9), (2, 0.1), (3, 1.0)])          # band 2 inverts
    c = calibrate.fit_isotonic(a, {}, _cfg(curve_bins=4))
    ys = [c.predict(x) for x in (0, 1, 2, 3)]
    assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(3))


def test_interpolation_is_continuous_between_knots():
    c = calibrate.fit_isotonic(_anchors([(0, 0.0), (2, 1.0)]), {}, _cfg(curve_bins=2))
    mid = c.predict(1.0)
    assert c.predict(0.0) < mid < c.predict(2.0)     # a bucket would return an endpoint


def test_out_of_range_clips_to_the_terminal_knot_and_is_counted():
    c = calibrate.fit_isotonic(_anchors([(0, 0.0), (1, 1.0)]), {}, _cfg(curve_bins=2))
    assert c.predict(99.0) == pytest.approx(c.knots_y[-1])
    assert c.predict(-99.0) == pytest.approx(c.knots_y[0])
    assert c.n_clipped == 2


def test_curve_is_fit_only_on_orm_unseen_anchors():
    # A memorized score is sharper than a fresh one; including it lifts the whole curve.
    clean = _anchors([(x, 0.0) for x in range(20)], seen=False)
    seen = _anchors([(x, 1.0) for x in range(20)], seen=True)
    c = calibrate.fit_isotonic(clean + seen, {}, _cfg(curve_bins=5))
    assert max(c.knots_y) < 0.05
    assert c.n_fit == 20


def test_fit_isotonic_refuses_an_all_seen_anchor_set():
    # Silently fitting nothing is impossible; fitting the memorized 20 would be the bug.
    with pytest.raises(ValueError, match="unseen"):
        calibrate.fit_isotonic(_anchors([(x, 1.0) for x in range(20)], seen=True), {}, _cfg())


def test_offset_corrected_scores_are_what_the_knots_are_placed_on():
    a = _anchors([(0.0, 0.0), (1.0, 1.0)], pid=3)
    c = calibrate.fit_isotonic(a, {"6:3": 5.0}, _cfg(curve_bins=2))
    assert list(c.knots_x) == pytest.approx([5.0, 6.0])


def test_a_one_point_band_borrows_the_pooled_residual_variance():
    # 0.0 would claim a variance the band never measured; the pooled estimate is honest.
    c = calibrate.fit_isotonic(_anchors([(0, 0.0), (1, 0.2), (2, 1.0)]), {}, _cfg(curve_bins=2))
    assert c.n_by_band == [1, 2]
    assert c.resid_var_by_band == pytest.approx([0.16, 0.16])


def test_every_band_including_the_last_is_reachable_from_band():
    c = calibrate.fit_isotonic(_anchors([(0, 0.0), (1, 0.5), (2, 1.0)]), {}, _cfg(curve_bins=3))
    assert len(c.resid_var_by_band) == 3
    assert [c.band(x) for x in (-99.0, 0.5, 1.5, 99.0)] == [0, 0, 1, 2]


def test_one_distinct_score_still_gives_a_usable_single_knot_curve():
    c = calibrate.fit_isotonic(_anchors([(3.0, 0.0), (3.0, 1.0)]), {}, _cfg(curve_bins=4))
    assert len(c.knots_x) == 1
    assert c.predict(-5.0) == pytest.approx(0.5) and c.predict(5.0) == pytest.approx(0.5)
    assert c.band(5.0) == 0


def test_curve_survives_a_json_round_trip():
    c = _ramp()
    back = calibrate.curve_from_dict(json.loads(json.dumps(c.to_dict())))
    assert [back.predict(x) for x in (-1.0, 2.5, 11.0)] == pytest.approx(
        [c.predict(-1.0), c.predict(2.5), c.predict(11.0)])
    assert back.n_fit == c.n_fit and back.resid_var_by_band == c.resid_var_by_band


# --- anchor targets ----------------------------------------------------------------------


def test_anchor_target_regrades_from_the_stored_speedup():
    row = {"correct": True, "compiled": True, "speedup_min": 2.0, "speedup": 2.0}
    hi = calibrate.anchor_target(row, _cfg(calib_speed_quant=0.0))
    assert 0.5 < hi <= 1.0
    assert calibrate.anchor_target({**row, "correct": False}, _cfg()) == 0.0


def test_anchor_target_reads_the_stat_the_config_names():
    row = {"correct": True, "compiled": True, "speedup_min": 0.25, "speedup": 4.0}
    assert calibrate.anchor_target(row, _cfg(speedup_stat="min")) < 0.6
    assert calibrate.anchor_target(row, _cfg(speedup_stat="mean")) == pytest.approx(1.0)


def test_a_correct_anchor_with_no_usable_speedup_is_none_not_zero():
    # 0.0 is the grade of a wrong kernel; an ungradable correct one is a drop, not a failure.
    row = {"correct": True, "compiled": True, "speedup_min": None, "speedup": None}
    assert calibrate.anchor_target(row, _cfg()) is None
    assert calibrate.anchor_target({**row, "speedup_min": -1.0}, _cfg()) is None


# --- offsets -----------------------------------------------------------------------------


def test_offset_solves_the_moment_equation():
    c = _ramp()
    a = _anchors([(2.0, 0.5), (3.0, 0.6)], pid=7)
    off, _ = calibrate.fit_offsets(a, c, _cfg())
    got = sum(c.predict(x.score + off["6:7"]["c"]) for x in a)
    assert got == pytest.approx(sum(x.target for x in a), abs=0.02)
    assert off["6:7"]["clamped"] is False


def test_the_offset_is_applied_inside_the_lookup_not_added_to_the_value():
    # curve.predict(score + c), never curve.predict(score) + c: the second leaves [0, 1].
    c = _ramp()
    off, _ = calibrate.fit_offsets(_anchors([(2.0, 0.5)], pid=7), c, _cfg())
    assert 0.0 <= c.predict(2.0 + off["6:7"]["c"]) <= 1.0


def test_clamp_fires_and_is_counted_when_every_anchor_is_wrong():
    off, meta = calibrate.fit_offsets(_anchors([(9.0, 0.0)] * 4, pid=9), _ramp(),
                                      _cfg(offset_clamp=4.0))
    assert off["6:9"]["clamped"] is True
    # `raw` is the clamped quantity. `c` is raw*shrink, and in a real campaign this same
    # problem is shrunk most of the way back to 0 -- asserting -4.0 there would pin a value
    # only the single-problem branch can produce.
    assert off["6:9"]["raw"] == pytest.approx(-4.0)
    assert meta["n_clamped"] == 1


def test_c_is_raw_times_shrink_for_every_problem():
    a = (_anchors([(2.0, 0.5)] * 3, pid=1) + _anchors([(5.0, 0.2)] * 8, pid=2)
         + _anchors([(7.0, 0.9)] * 5, pid=3))
    off, meta = calibrate.fit_offsets(a, _ramp(), _cfg())
    assert 0.0 < meta["shrink_mean"] < 1.0            # the composition is actually exercised
    for o in off.values():
        assert o["c"] == pytest.approx(o["raw"] * o["shrink"])
        assert abs(o["c"]) < abs(o["raw"])


def test_no_anchors_gives_zero_and_is_counted_not_dropped():
    c = calibrate.fit_isotonic(_anchors([(0, 0.0), (1, 1.0)]), {}, _cfg(curve_bins=2))
    off, meta = calibrate.fit_offsets([], c, _cfg())
    assert off == {} and meta["n_no_anchors"] == 0
    assert calibrate.offset_for(off, 6, 41) == 0.0        # missing problem -> 0.0, not KeyError


def test_offsets_use_every_anchor_including_the_orm_seen_ones():
    # The curve drops memorized anchors; the offsets must not -- the contamination bias
    # points the other way there and partly cancels.
    pairs = [(2.0, 0.5), (3.0, 0.6), (4.0, 0.9), (5.0, 0.9)]
    mixed = _anchors(pairs[:2], pid=7) + _anchors(pairs[2:], pid=7, seen=True)
    all_fresh = _anchors(pairs, pid=7)
    off, _ = calibrate.fit_offsets(mixed, _ramp(), _cfg())
    same, _ = calibrate.fit_offsets(all_fresh, _ramp(), _cfg())
    assert off["6:7"]["n_anchors"] == 4
    assert off["6:7"]["c"] == pytest.approx(same["6:7"]["c"])


def test_a_one_anchor_problem_is_shrunk_harder_than_a_many_anchor_one():
    a = _anchors([(2.0, 0.5)], pid=1) + _anchors([(5.0, 0.2)] * 20, pid=2)
    off, meta = calibrate.fit_offsets(a, _ramp(), _cfg())
    assert 0.0 < off["6:1"]["shrink"] < off["6:2"]["shrink"] < 1.0
    assert off["6:1"]["var"] > off["6:2"]["var"]
    assert meta["tau2"] > 0.0


def test_one_no_information_problem_does_not_zero_every_other_offset():
    # The failure this guards: var=1e6 averaged into the tau2 moment estimator drives tau2 to
    # its floor, every shrink to 0 and every offset to exactly 0.0 -- the pooled fit, reported
    # as a converged success. A whole-campaign kill from one all-incorrect problem.
    ramp = _ramp()
    a = _anchors([(2.0, 0.5)] * 3, pid=1) + _anchors([(5.0, 0.2)] * 8, pid=2)
    a += _anchors([(7.0, 0.9)] * 5, pid=3)
    a += _anchors([(0.0, 0.0)] * 4, pid=99)            # bottom of the curve: q=0, no information
    off, meta = calibrate.fit_offsets(a, ramp, _cfg())
    assert off["6:99"]["var"] == calibrate.NO_INFO_VAR
    assert off["6:99"]["c"] == pytest.approx(0.0, abs=1e-3)   # unidentified, so shrunk away
    assert meta["n_no_info"] == 1 and meta["n_informative"] == 3
    assert meta["tau2"] > 0.0
    assert all(abs(off[f"6:{p}"]["c"]) > 0.5 for p in (1, 2, 3))


def test_an_unestimable_spread_keeps_the_identified_offset_and_zeroes_the_rest():
    # The identified problem keeps its solved offset -- there is no ensemble to shrink it
    # toward. The unidentified ones go to 0 rather than being applied at full magnitude,
    # which would push the largest offsets available into exactly the problems that carry no
    # information. The flag, not a quiet tau2 of 0.0, is what a reader keys off.
    out = calibrate.eb_shrink(c_hat={1: 1.0, 2: -1.0, 3: 2.0},
                              var={1: calibrate.NO_INFO_VAR, 2: calibrate.NO_INFO_VAR, 3: 0.1})
    assert out["tau2_estimable"] is False and out["tau2"] is None
    assert out["shrink"] == {1: 0.0, 2: 0.0, 3: 1.0}
    assert out["n_informative"] == 1


def test_an_offset_the_curve_carries_no_information_about_is_shrunk_away():
    # Every problem unidentified: none of them may carry an offset, least of all the clamped
    # -offset_clamp the solver hands back for a curve it cannot move.
    flat = calibrate.fit_isotonic(_anchors([(x, 0.0) for x in range(10)]), {}, _cfg(curve_bins=5))
    a = _anchors([(1.0, 0.0)] * 3, pid=1) + _anchors([(2.0, 0.0)] * 3, pid=2)
    off, meta = calibrate.fit_offsets(a, flat, _cfg())
    assert off["6:1"]["var"] == calibrate.NO_INFO_VAR
    assert off["6:1"]["raw"] == pytest.approx(-4.0)      # the solver's censored answer
    assert off["6:1"]["c"] == 0.0 and off["6:2"]["c"] == 0.0
    assert meta["tau2_estimable"] is False and meta["n_informative"] == 0


def test_the_lone_identified_offset_survives_when_the_spread_is_unestimable():
    # One problem solvable, one all-incorrect on the curve's floor: the identified offset
    # stands, the unidentified one is zeroed rather than applied at the clamp.
    a = _anchors([(2.0, 0.5), (3.0, 0.6)], pid=1) + _anchors([(0.0, 0.0)] * 3, pid=2)
    off, meta = calibrate.fit_offsets(a, _ramp(), _cfg())
    assert meta["tau2_estimable"] is False and meta["n_informative"] == 1
    assert off["6:1"]["c"] == pytest.approx(off["6:1"]["raw"]) == pytest.approx(3.0)
    assert off["6:2"]["c"] == 0.0 and off["6:2"]["raw"] == pytest.approx(-4.0)


def test_fixed_kappa_counts_informative_problems_not_every_problem():
    out = calibrate.eb_shrink(c_hat={1: 1.0, 2: 2.0}, var={1: 0.25, 2: calibrate.NO_INFO_VAR},
                              kappa=8.0, n_anchors={1: 4, 2: 4})
    assert out["n_informative"] == 1


def test_a_no_information_problem_is_kept_out_of_tau2_but_still_shrunk():
    huge = calibrate.NO_INFO_VAR
    out = calibrate.eb_shrink(c_hat={1: -2.0, 2: 2.0, 3: -4.0}, var={1: 0.05, 2: 0.05, 3: huge})
    assert out["n_informative"] == 2
    assert out["tau2"] == pytest.approx(np.var([-2.0, 2.0]) - 0.05)
    assert out["shrink"][1] > 0.9 and out["shrink"][3] < 1e-5


# --- the empirical-Bayes shrink ----------------------------------------------------------


def test_eb_shrink_goes_to_one_at_wide_tau_and_to_zero_at_no_spread():
    wide = calibrate.eb_shrink(c_hat={1: -2.0, 2: 2.0, 3: -1.8, 4: 1.9},
                               var={i: 0.05 for i in range(1, 5)})
    none = calibrate.eb_shrink(c_hat={1: 0.02, 2: -0.01, 3: 0.0, 4: 0.01},
                               var={i: 0.5 for i in range(1, 5)})
    assert wide["shrink"][1] > 0.9 and wide["tau2"] > 1.0
    assert none["shrink"][1] < 0.1 and none["tau2"] < 0.05


def test_a_negative_tau2_estimate_is_floored_but_the_raw_value_is_kept():
    # A floored 0.0 and a measured 0.0 both zero every offset; only tau2_raw tells them apart.
    out = calibrate.eb_shrink(c_hat={1: 0.1, 2: -0.1}, var={1: 9.0, 2: 9.0})
    assert out["tau2"] == 0.0 and out["shrink"] == {1: 0.0, 2: 0.0}
    assert out["tau2_raw"] == pytest.approx(0.01 - 9.0)


def test_eb_shrink_will_not_invent_a_spread_from_a_single_problem():
    # Var over one c is 0 by construction, not by evidence: shrinking on it would throw
    # away the only offset the campaign measured.
    out = calibrate.eb_shrink(c_hat={1: -3.0}, var={1: 0.5})
    assert out["tau2"] is None and out["shrink"] == {1: 1.0}


def test_fixed_kappa_reproduces_the_old_form_exactly():
    out = calibrate.eb_shrink(c_hat={1: 1.0}, var={1: 0.25}, kappa=8.0, n_anchors={1: 4})
    assert out["shrink"][1] == pytest.approx(4 / (4 + 8))


def test_fixed_kappa_without_n_anchors_raises_rather_than_shrinking_by_nothing():
    with pytest.raises(ValueError, match="n_anchors"):
        calibrate.eb_shrink(c_hat={1: 1.0}, var={1: 0.25}, kappa=8.0)
    # Named before any variance is read, so a missing var cannot mask the real mistake.
    with pytest.raises(ValueError, match="n_anchors"):
        calibrate.eb_shrink(c_hat={1: 1.0, 2: 2.0}, var={1: 0.25}, kappa=8.0)


# --- the alternating fit ------------------------------------------------------------------


def _two_offset_problems():
    rng = np.random.default_rng(0)
    truth = lambda x: 1 / (1 + math.exp(-x))                       # noqa: E731
    a = []
    for pid, c_true in ((1, -2.0), (2, +2.0)):            # a large known offset between them
        for i, s in enumerate(rng.uniform(-4, 4, 400)):
            a.append(calibrate.Anchor(f"{pid}:{i}", 6, pid, s - c_true, truth(s), False))
    return a


def test_alternating_fit_recovers_the_truth_the_pooled_fit_flattens():
    # Independent ground truth: the targets are sigma(s) in closed form and the scores are
    # s - c_true, so the fit is checked against numbers no fitting code produced.
    a = _two_offset_problems()
    flat, _, _ = calibrate.fit_joint(a, _cfg(curve_iters=1))
    good, off, meta = calibrate.fit_joint(a, _cfg(curve_iters=3))
    # Not the span: the extreme knots come from whichever problem reaches furthest and are
    # unflattened either way. The slope through the overlap is what pooling averages out.
    slope = lambda c: c.predict(1.0) - c.predict(-1.0)             # noqa: E731
    truth = 2 / (1 + math.exp(-1.0)) - 1.0                         # sigma(1) - sigma(-1) = 0.4621
    assert slope(good) == pytest.approx(truth, abs=0.03)           # recovered, not just larger
    assert slope(flat) < truth - 0.15                              # pooling halves it
    assert off["6:1"]["c"] == pytest.approx(-2.0, abs=0.15)
    assert off["6:2"]["c"] == pytest.approx(+2.0, abs=0.15)
    assert 3.5 < meta["tau2"] < 4.5                                # Var([-2, +2]) = 4
    assert meta["converged"] is True and meta["max_offset_delta"] < calibrate.TOL


def test_a_fit_that_did_not_converge_says_so():
    good, _, meta = calibrate.fit_joint(_two_offset_problems(), _cfg(curve_iters=1))
    assert meta["converged"] is False
    assert meta["n_iters"] == 1 and meta["max_offset_delta"] > calibrate.TOL


def test_the_returned_offsets_were_solved_against_the_returned_curve():
    a = _two_offset_problems()
    curve, off, _ = calibrate.fit_joint(a, _cfg(curve_iters=2))
    for pkey in ("6:1", "6:2"):
        group = [x for x in a if x.pkey == pkey]
        got = sum(curve.predict(x.score + off[pkey]["raw"]) for x in group)
        assert got == pytest.approx(sum(x.target for x in group), rel=0.02)


def test_use_anchors_off_fits_a_pooled_curve_and_no_offsets():
    curve, off, meta = calibrate.fit_joint(_two_offset_problems(), _cfg(use_anchors=False))
    assert off == {} and meta["converged"] is True and meta["n_iters"] == 1
    assert len(curve.knots_x) > 1
    # Same meta shape as a real fit, or main() KeyErrors after the artifacts are written.
    full = calibrate.fit_joint(_two_offset_problems(), _cfg(curve_iters=1))[2]
    assert set(meta) == set(full)


# --- what the ORM memorized ----------------------------------------------------------------


def _lists_file(path, entries):
    with open(path, "w") as f:
        for level, pid, cands in entries:
            f.write(json.dumps({"level": level, "problem_id": pid, "candidates": [
                {"run_name": run, "sample_id": sid, "rel": 1.0} for run, sid in cands]}) + "\n")


def test_load_orm_seen_expands_braces_and_keys_on_run_problem_sample(tmp_path):
    # The shipped configs point at lists_{train,val}_*.jsonl and glob does not expand braces:
    # an unexpanded pattern matches nothing, and an empty seen set fits the curve on the
    # kernels the ORM memorized.
    _lists_file(tmp_path / "lists_train_x.jsonl", [(6, 18, [("run__shard_00__round0", 3)])])
    _lists_file(tmp_path / "lists_val_x.jsonl", [(6, 19, [("run__shard_01__round2", 7)])])
    seen = calibrate.load_orm_seen(str(tmp_path / "lists_{train,val}_x.jsonl"))
    assert seen == {("run__shard_00__round0", 18, 3), ("run__shard_01__round2", 19, 7)}


def test_load_orm_seen_raises_when_the_pattern_matches_nothing(tmp_path):
    with pytest.raises(FileNotFoundError, match="orm_lists_glob"):
        calibrate.load_orm_seen(str(tmp_path / "nothing_here_*.jsonl"))


def test_orm_seen_key_rebuilds_the_composite_run_name_the_lists_use(tmp_path):
    # The lists name a source row by its *unit* -- "<run>__<shard>__round<N>" -- not by the
    # bare run_name a v1 row carries. Matching on the bare name would find nothing.
    row = _row(run="gpt-oss-120b_kb6", shard="shard_03", rnd=2, pid=18, sid=3)
    assert calibrate.orm_seen_key(row) == ("gpt-oss-120b_kb6__shard_03__round2", 18, 3)
    _lists_file(tmp_path / "lists_train_x.jsonl",
                [(6, 18, [("gpt-oss-120b_kb6__shard_03__round2", 3)])])
    assert calibrate.orm_seen_key(row) in calibrate.load_orm_seen(
        str(tmp_path / "lists_train_x.jsonl"))


def test_anchor_key_is_exactly_the_id_orm_score_writes():
    row = _row(run="run", shard="shard_07", rnd=1, pid=18, sid=3)
    assert calibrate.anchor_key(row) == orm_score.anchor_item(row).id


# --- loading the anchors --------------------------------------------------------------------


def _campaign(tmp_path, rows, scores, seen=(), **over):
    """A campaign on disk: v1 parts, the ORM's anchor score part, the ORM's own lists."""
    parts = tmp_path / "parts"
    parts.mkdir(exist_ok=True)
    with open(parts / "p0.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "out"
    path = orm_score.unit_score_path(str(out_dir), orm_score.ANCHORS_UNIT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s in scores:
            f.write(json.dumps(s) + "\n")

    _lists_file(tmp_path / "lists_train_x.jsonl", [(6, pid, [(run, sid)])
                                                   for run, pid, sid in seen])
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}")

    cfg = RerankerConfig()
    conf = cfg.prm_rollout
    conf.parts_glob = str(parts / "*.jsonl")
    conf.out_dir = str(out_dir)
    conf.run_tags = {"a_run": "ar"}
    conf.anchor_rounds = [0]
    conf.label_source = "imputed"
    conf.orm_checkpoint = "ckpt"
    conf.orm_lists_glob = str(tmp_path / "lists_{train,val}_x.jsonl")
    conf.baseline_timing_json = str(baseline)
    conf.calib_speed_quant = 0.0
    for k, v in over.items():
        setattr(conf, k, v)
    return cfg


def test_iter_anchor_rows_selects_exactly_what_orm_score_anchors(tmp_path):
    rows = [_row(sid=0), _row(sid=1, rnd=1), _row(sid=2, run="other_run"), _row(sid=3)]
    cfg = _campaign(tmp_path, rows, [])
    mine = [calibrate.anchor_key(r) for r in calibrate.iter_anchor_rows(cfg)]
    theirs = [i.id for i in orm_score.iter_anchor_items(cfg)]
    assert mine == theirs and len(mine) == 2


def test_load_anchors_joins_scores_targets_and_the_seen_flag(tmp_path):
    rows = [_row(sid=0, speedup_min=4.0), _row(sid=1, correct=False, speedup_min=None,
                                               speedup=None)]
    scores = [_score_row(rows[0], 1.5), _score_row(rows[1], -2.0)]
    cfg = _campaign(tmp_path, rows, scores, seen=[("a_run__shard_00__round0", 1, 0)])
    anchors, ledger = calibrate.load_anchors(cfg)
    by_key = {a.key: a for a in anchors}
    hot = by_key[orm_score.anchor_item(rows[0]).id]
    cold = by_key[orm_score.anchor_item(rows[1]).id]
    assert hot.target == pytest.approx(1.0) and hot.score == 1.5 and hot.orm_seen is True
    assert cold.target == 0.0 and cold.orm_seen is False
    assert ledger["orm_seen"] == 1 and ledger["orm_unseen"] == 1
    assert ledger["orm_checkpoint_sha"] == "ckpt"


def test_load_anchors_counts_an_ungradable_correct_anchor_instead_of_grading_it(tmp_path):
    rows = [_row(sid=0, correct=True, speedup_min=None, speedup=None), _row(sid=1)]
    scores = [_score_row(rows[0], 0.5), _score_row(rows[1], 0.5)]
    cfg = _campaign(tmp_path, rows, scores)
    anchors, ledger = calibrate.load_anchors(cfg)
    assert len(anchors) == 1 and ledger["no_target"] == 1


def test_load_anchors_counts_a_score_whose_v1_row_is_gone(tmp_path):
    rows = [_row(sid=0)]
    scores = [_score_row(rows[0], 0.5), _score_row(_row(sid=9), 0.5)]
    cfg = _campaign(tmp_path, rows, scores)
    anchors, ledger = calibrate.load_anchors(cfg)
    assert len(anchors) == 1 and ledger["no_row"] == 1


def test_anchors_scored_under_two_checkpoints_raise(tmp_path):
    rows = [_row(sid=0), _row(sid=1)]
    scores = [_score_row(rows[0], 0.5, sha="one"), _score_row(rows[1], 0.5, sha="two")]
    cfg = _campaign(tmp_path, rows, scores)
    with pytest.raises(ValueError, match="checkpoint"):
        calibrate.load_anchors(cfg)


# --- the job ---------------------------------------------------------------------------------


def _graded_campaign(tmp_path, **over):
    """Two problems, 12 anchors each, score = target + a per-problem offset.

    Only *correct* anchors are memorized, as in the real corpus: a compile-fail never reaches
    the ORM's lists, so the unseen set the curve is fit on is enriched in 0.0 targets.
    """
    rows, scores, seen = [], [], []
    for pid, shift in ((1, -1.0), (2, +1.0)):
        for sid in range(12):
            frac = sid / 11
            correct = sid % 3 != 0
            row = _row(pid=pid, sid=sid, correct=correct,
                       speedup_min=0.2 + frac * 3.8, speedup=0.2 + frac * 3.8)
            rows.append(row)
            # A failing kernel scores low, as it does in the corpus, so the curve the driver
            # tests exercise is a real monotone one rather than a flat PAVA collapse.
            scores.append(_score_row(row, (frac * 4 - 2 if correct else -3.0) - shift))
            if correct and sid % 4 == 0:
                seen.append(("a_run__shard_00__round0", pid, sid))
    return _campaign(tmp_path, rows, scores, seen=seen, curve_bins=4, **over)


def test_calibrate_writes_a_curve_and_offsets_and_shas_both_into_the_manifest(tmp_path):
    cfg = _graded_campaign(tmp_path)
    manifest = calibrate.calibrate(cfg)
    out_dir = cfg.prm_rollout.out_dir

    curve = json.loads(open(os.path.join(out_dir, calibrate.CURVE)).read())
    assert set(curve) == {"knots_x", "knots_y", "resid_var_by_band", "n_by_band", "n_fit",
                          "curve_bins", "orm_checkpoint_sha", "target_dist"}
    assert curve["knots_y"] == sorted(curve["knots_y"])
    assert curve["orm_checkpoint_sha"] == "ckpt"
    assert curve["n_fit"] == 20                       # 24 anchors, 4 of them memorized

    offsets = json.loads(open(os.path.join(out_dir, calibrate.OFFSETS)).read())
    assert set(offsets) == {"meta", "offsets"}
    assert set(offsets["offsets"]) == {"6:1", "6:2"}
    assert set(offsets["offsets"]["6:1"]) == {"c", "raw", "var", "shrink", "n_anchors",
                                              "clamped"}
    assert offsets["offsets"]["6:1"]["n_anchors"] == 12   # offsets keep the memorized ones

    assert manifest["curve_sha1"] == build._sha1(os.path.join(out_dir, calibrate.CURVE))
    assert manifest["offsets_sha1"] == build._sha1(os.path.join(out_dir, calibrate.OFFSETS))
    assert manifest["anchors"]["orm_seen"] == 4
    assert manifest["fit"]["converged"] in (True, False)


def test_the_written_curve_and_offsets_reproduce_the_fit(tmp_path):
    cfg = _graded_campaign(tmp_path)
    calibrate.calibrate(cfg)
    out_dir = cfg.prm_rollout.out_dir
    curve = calibrate.read_curve(os.path.join(out_dir, calibrate.CURVE))
    offsets, meta = calibrate.read_offsets(os.path.join(out_dir, calibrate.OFFSETS))

    anchors, _ = calibrate.load_anchors(cfg)
    fitted, off, _ = calibrate.fit_joint(anchors, cfg.prm_rollout)
    assert list(curve.knots_y) == pytest.approx(list(fitted.knots_y))
    assert calibrate.offset_for(offsets, 6, 1) == pytest.approx(off["6:1"]["c"])
    assert calibrate.offset_for(offsets, 6, 404) == 0.0
    assert meta["kappa_mode"] == "auto"


def test_calibrate_refuses_to_fit_without_the_orm_training_lists(tmp_path):
    cfg = _graded_campaign(tmp_path, orm_lists_glob=None)
    with pytest.raises(ValueError, match="orm_lists_glob"):
        calibrate.calibrate(cfg)


def test_calibrate_runs_on_a_measured_campaign_for_validation(tmp_path):
    # prm_rollout_l1.yaml is label_source=measured and carries the ORM keys precisely so this
    # sweep can check the curve against measured truth -- level 1 is the only corpus where it
    # can be checked at all. Refusing here would make that validation impossible.
    cfg = _graded_campaign(tmp_path, label_source="measured")
    manifest = calibrate.calibrate(cfg)
    assert manifest["label_source"] == "measured"
    assert os.path.isfile(os.path.join(cfg.prm_rollout.out_dir, calibrate.CURVE))
    _, warn = calibrate.summary(manifest)
    assert any("nothing may be labelled from this curve" in w for w in warn)


def test_the_offsets_off_ablation_runs_end_to_end(tmp_path):
    cfg = _graded_campaign(tmp_path, use_anchors=False)
    manifest = calibrate.calibrate(cfg)
    offsets, meta = calibrate.read_offsets(
        os.path.join(cfg.prm_rollout.out_dir, calibrate.OFFSETS))
    assert offsets == {} and meta["converged"] is True
    lines, _ = calibrate.summary(manifest)          # KeyErrors here if the meta is short
    assert any("offsets: tau2=None" in line for line in lines)


def test_the_curve_records_how_biased_its_fit_set_is(tmp_path):
    # The unseen slice is enriched in 0.0 targets by construction; it has to be visible
    # before 1.1M labels are written, not after.
    cfg = _graded_campaign(tmp_path)
    manifest = calibrate.calibrate(cfg)
    curve = json.loads(open(os.path.join(cfg.prm_rollout.out_dir, calibrate.CURVE)).read())
    fit, whole = curve["target_dist"]["fit"], curve["target_dist"]["all"]
    assert fit["n"] == 20 and whole["n"] == 24
    assert set(fit) == {"n", "frac_zero", "mean", "p25", "p50", "p75"}
    assert fit["frac_zero"] > whole["frac_zero"]      # the skew the curve is actually fit on
    assert fit["mean"] < whole["mean"]
    assert calibrate.summary(manifest)[0][2].startswith("targets: fit set")


# --- the level-1 validation sweep, through orm_score's own writer ----------------------------


class _StubTokenizer:
    """One id per character -- deterministic, no vocab needed. Code length becomes the score."""

    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 1000 for c in text]


class _StubScorer:
    def __call__(self, encoded):
        return [float(len(e)) / 20.0 for e in encoded]


def test_the_measured_validation_sweep_runs_from_orm_score_to_a_curve(tmp_path):
    """The level-1 path end to end: orm_score writes ORM_SCORES/_anchors.jsonl itself and
    calibrate joins to it. Every other driver test hand-writes that part, so this is the only
    one that would catch anchor_key drifting from the id orm_score actually puts on disk, or
    the label_source guard making the sweep unreachable."""
    rows, seen = [], []
    for pid in (1, 2):
        for sid in range(10):
            frac = sid / 9
            correct = sid % 3 != 0
            row = _row(pid=pid, sid=sid, correct=correct,
                       speedup_min=0.2 + frac * 3.8, speedup=0.2 + frac * 3.8)
            # Length drives the stub's logit, so the anchors get a real score spread.
            row["raw"] = "```python\n" + "x" * (10 + sid * 9 + pid * 5) + "\n```"
            rows.append(row)
            if correct and sid % 4 == 0:
                seen.append(("a_run__shard_00__round0", pid, sid))
    cfg = _campaign(tmp_path, rows, [], seen=seen, curve_bins=4, label_source="measured")

    kb = tmp_path / "kb" / "KernelBench" / "level6"
    kb.mkdir(parents=True)
    for pid in (1, 2):
        (kb / f"{pid}_x.py").write_text("class Model:\n    pass\n")
    written = orm_score.score_anchors(
        cfg, _StubScorer(), SequenceEncoder(_StubTokenizer(), 6144, 1024),
        str(tmp_path / "kb" / "KernelBench"), "ckpt")
    assert written == orm_score.unit_score_path(cfg.prm_rollout.out_dir, orm_score.ANCHORS_UNIT)

    manifest = calibrate.calibrate(cfg)
    assert manifest["anchors"]["no_row"] == 0        # the ids orm_score wrote all joined
    assert manifest["anchors"]["kept"] == 20
    assert manifest["anchors"]["orm_seen"] == 4
    assert manifest["label_source"] == "measured"
    curve = calibrate.read_curve(os.path.join(cfg.prm_rollout.out_dir, calibrate.CURVE))
    assert len(curve.knots_x) > 1 and curve.n_fit == 16


def test_a_campaign_with_no_correct_anchor_is_told_it_is_the_pooled_fit(tmp_path):
    """The warning has to describe the state it fires on. It once said "no offset was shrunk
    at all and each one stands as solved", which after the zeroing fix was the exact opposite
    of what happens -- and this is the run where saying it right matters most: with nothing
    identified, the campaign IS the pooled fit v3 exists to avoid."""
    rows, scores = [], []
    for pid in (1, 2):
        for sid in range(6):
            row = _row(pid=pid, sid=sid, correct=False, speedup_min=None, speedup=None)
            rows.append(row)
            scores.append(_score_row(row, sid * 0.5 - pid))
    cfg = _campaign(tmp_path, rows, scores, curve_bins=3)
    manifest = calibrate.calibrate(cfg)

    fit = manifest["fit"]
    assert fit["tau2_estimable"] is False and fit["n_informative"] == 0 and fit["n_problems"] == 2
    offsets, _ = calibrate.read_offsets(os.path.join(cfg.prm_rollout.out_dir, calibrate.OFFSETS))
    assert all(o["c"] == 0.0 for o in offsets.values())
    assert all(o["raw"] == pytest.approx(-4.0) for o in offsets.values())

    msg = next(w for w in calibrate.summary(manifest)[1] if "tau2 could not be estimated" in w)
    assert "0 of 2 problems" in msg and "2 offsets were zeroed" in msg
    assert "this IS the pooled fit" in msg
    assert "stands as solved" not in msg      # the round-2 wording, now false
