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


def test_curve_fit_orm_seen_puts_the_memorized_anchors_back_in():
    # On level 6 the exclusion is selected on the label, so it collapses the curve instead of
    # protecting it. The flag has to reach both the fit count and the knots.
    clean = _anchors([(x, 0.0) for x in range(20)], seen=False)
    seen = _anchors([(x, 1.0) for x in range(20)], seen=True)
    c = calibrate.fit_isotonic(clean + seen, {}, _cfg(curve_bins=5, curve_fit_orm_seen=True))
    assert c.n_fit == 40 and max(c.knots_y) > 0.4


def test_curve_fit_orm_seen_makes_an_all_seen_anchor_set_fittable():
    a = _anchors([(x, 1.0) for x in range(20)], seen=True)
    assert calibrate.fit_isotonic(a, {}, _cfg(curve_fit_orm_seen=True)).n_fit == 20


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
    good, off, meta = calibrate.fit_joint(a, _cfg(curve_iters=6))
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


def test_tau2_is_frozen_after_the_first_pass():
    # The fix for the period-2 cycle. tau2 asks how much problems still differ; applying the
    # offsets is what makes them stop, so a pass-2 estimate reads its own effect and collapses.
    a = _two_offset_problems()
    first = calibrate.fit_joint(a, _cfg(curve_iters=1))[2]["tau2"]
    for iters in (2, 3, 4, 5, 8):
        assert calibrate.fit_joint(a, _cfg(curve_iters=iters))[2]["tau2"] == first


def test_the_fit_converges_geometrically_instead_of_cycling():
    # Before damping + freezing this alternated between two states forever, and the reported
    # delta never fell: on level 6 it sat at ~1.39 for twenty passes.
    a = _two_offset_problems()
    deltas = [calibrate.fit_joint(a, _cfg(curve_iters=i))[2]["max_offset_delta"]
              for i in range(1, 7)]
    assert deltas == sorted(deltas, reverse=True)                  # monotone, never a cycle
    for prev, cur in zip(deltas[1:], deltas[2:]):                  # halving, per DAMP = 0.5
        assert cur == pytest.approx(prev * calibrate.DAMP, rel=0.3)
    assert calibrate.fit_joint(a, _cfg(curve_iters=6))[2]["converged"] is True


def test_an_even_and_an_odd_iteration_count_now_agree():
    # The bug this replaced: curve_iters was a parity switch. Odd landed on offsets at full
    # spread, even on offsets zeroed against a curve fit as though they were not -- so an
    # even value silently shipped an incoherent fit.
    a = _two_offset_problems()
    _, even, m_even = calibrate.fit_joint(a, _cfg(curve_iters=4))
    _, odd, m_odd = calibrate.fit_joint(a, _cfg(curve_iters=5))
    assert m_even["tau2"] == m_odd["tau2"]
    for pkey in ("6:1", "6:2"):
        assert even[pkey]["c"] == pytest.approx(odd[pkey]["c"], abs=0.02)


def test_the_convergence_test_ignores_the_worst_one_percent_but_not_on_a_small_fit():
    # A few problems sit on the clamp boundary and flip forever without moving the fit.
    big = {f"p{i}": {"c": 0.0} for i in range(200)}
    big["p0"]["c"] = 99.0                                          # the one wild mover
    _, worst, trimmed = calibrate._offset_step({}, big, damp=False)
    assert worst == 99.0 and trimmed == 0.0
    # Under 100 problems there is no 1% to trim, so nothing is hidden.
    small = {"p0": {"c": 99.0}, "p1": {"c": 0.0}}
    _, worst, trimmed = calibrate._offset_step({}, small, damp=False)
    assert worst == 99.0 and trimmed == 99.0


def test_damping_moves_the_state_halfway_and_leaves_the_returned_offsets_alone():
    # Damping is on the iteration state only: the offsets fit_joint returns stay the exact
    # moment-match against the curve it returns, which is what N7's consumers apply.
    nxt, _, _ = calibrate._offset_step({"p": 0.0}, {"p": {"c": 1.0}}, damp=True)
    assert nxt["p"] == pytest.approx(0.5)
    nxt, _, _ = calibrate._offset_step({"p": 0.0}, {"p": {"c": 1.0}}, damp=False)
    assert nxt["p"] == pytest.approx(1.0)


def test_eb_shrink_uses_a_frozen_tau2_verbatim_instead_of_estimating_one():
    out = calibrate.eb_shrink(c_hat={1: 0.01, 2: -0.01}, var={1: 0.25, 2: 0.25}, tau2=4.0)
    assert out["tau2"] == 4.0 and out["tau2_raw"] == 4.0 and out["tau2_estimable"] is True
    assert out["shrink"][1] == pytest.approx(4.0 / 4.25)
    # Same inputs, estimating: these c_hat have no spread at all, so it would shrink to ~0.
    free = calibrate.eb_shrink(c_hat={1: 0.01, 2: -0.01}, var={1: 0.25, 2: 0.25})
    assert free["shrink"][1] < 0.01


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
    assert set(offsets) == {"meta", "offsets", "dead"}
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


# --- validate: the level-1 full-corpus calibration check + core-size diagnostic --------------


def _synthetic(noise=0.3, n_problems=30, anchors_per_problem=15, seed=0):
    """A validate() fixture with closed-form ground truth. Each problem gets a true offset
    c_true; anchor score = x - c_true (clean), target = sigma(x). Each problem also gets two
    lists (one per round), each with two prefixes at different depths; a prefix's own rollout
    scores are x2 - c_true + N(0, noise). noise=0 makes imputation exact up to fit error;
    noise=99 swamps the rollout scores so the imputed order is a coin flip -- the two ends
    test_validate_is_pinned_at_both_ends checks.

    `c_true` is drawn from `fit_offsets`' usual wide spread (unlike an earlier version of this
    fixture, which had to narrow it: under the corrected design every evaluated problem gets
    its own offset fit from its own anchors in the headline run, never a default 0.0, so a
    wide, realistic c_true no longer breaks the pinned cal_error bound). Deliberately returns
    only the keys `validate` itself takes (no core_sizes/seed/n_reps), so every test can
    freely override those without a duplicate-keyword collision against `**_synthetic(...)`.
    """
    rng = np.random.default_rng(seed)
    sigma = lambda x: 1.0 / (1.0 + np.exp(-x))                       # noqa: E731
    anchors, rollout_scores, measured = [], {}, {}
    for pid in range(n_problems):
        c_true = float(rng.uniform(-1.5, 1.5))
        for i in range(anchors_per_problem):
            x = float(rng.uniform(-4, 4))
            anchors.append(calibrate.Anchor(f"a{pid}_{i}", 1, pid, x - c_true, sigma(x), False,
                                            n_code_tokens=int(rng.integers(50, 500))))
        for rnd in range(2):
            list_key = f"synth:1:{pid}:{rnd}:0"
            for depth in range(4):     # 4 prefixes/list -> up to 6 pairs, not 1 -- tight n_pairs
                                        # is what keeps the noise=99 pair_agreement bound stable
                x2 = float(rng.uniform(-4, 4))
                pfx = f"p{pid}_{rnd}_{depth}"
                measured[pfx] = calibrate.Measured(list_key, sigma(x2), 1, pid, rnd, depth)
                rollout_scores[pfx] = [float(x2 - c_true + rng.normal(0, noise)) for _ in range(4)]
    return {"anchors": anchors, "rollout_scores": rollout_scores, "measured": measured,
            "cfg": _cfg(curve_bins=20, offset_clamp=6.0)}


def test_validate_is_pinned_at_both_ends():
    # Perfect imputation -> ~0 calibration error. Pure noise -> ~0.5 pair agreement. Both read
    # off the full-corpus headline (top-level cal_error/pair_agreement), not the sweep.
    #
    # noise=0's cal_error bound is tight (< 0.02), so every problem's own EB-shrunk offset has
    # to be recovered close to its true c_true, not just its curve well-fit: with too few
    # anchors per problem, `var_i` (fit_offsets' proxy variance) stays large relative to tau2
    # and EB shrinkage biases every offset toward 0 even off noiseless anchors. 80/problem
    # keeps that bias an order of magnitude under the bound.
    #
    # noise=99's bound (0.4-0.6) needs no such precision -- pure noise swamps everything
    # regardless -- but DOES need enough pairs that the observed proportion cannot drift past
    # 0.6 by chance alone: with too few pairs a single unlucky seed can land the Binomial(n,
    # 0.5) draw outside the window even though the underlying process is exactly chance. More
    # problems (cheap: anchors_per_problem stays at the default) tightens that.
    perfect = calibrate.validate(**_synthetic(noise=0.0, anchors_per_problem=80))
    noise = calibrate.validate(**_synthetic(noise=99.0, n_problems=80))
    assert perfect["cal_error"] < 0.02 and perfect["pair_agreement"] > 0.95
    assert 0.4 < noise["pair_agreement"] < 0.6


def test_every_evaluated_problem_gets_its_own_offset_regardless_of_the_curve_core():
    # The corrected contract: the core-size sweep restricts which problems' anchors feed the
    # CURVE, but every evaluated problem still gets its own offset from its own anchors and
    # is still imputed and compared -- never held out. A core far smaller than the full
    # problem pool should still leave no_anchor_rate at 0.
    out = calibrate.validate(**_synthetic(noise=0.3), core_sizes=[4])
    fold = out["sweep"][0]
    assert fold["core_size"] == 4 and len(fold["core_problems"]) == 4
    assert fold["no_anchor_rate"] == pytest.approx(0.0)
    assert set(fold["eval_problems"]) == set(out["eval_problems"])   # same scope as the headline


def test_sweep_reports_one_row_per_core_size():
    out = calibrate.validate(**_synthetic(noise=0.3), core_sizes=[2, 4, 8])
    assert [r["core_size"] for r in out["sweep"]] == [2, 4, 8]
    assert all(r["n_pairs"] > 0 for r in out["sweep"])


def test_ablations_are_reported():
    out = calibrate.validate(**_synthetic(noise=0.3))
    for k in ("no_offset", "curve_iters_1", "fixed_kappa_8"):
        assert k in out["ablations"] and "cal_error" in out["ablations"][k]


def test_ablations_are_measured_against_the_full_corpus_like_the_headline():
    # No "core" for an ablation any more -- it is fit_joint on every anchor, same as the
    # headline, just with one config knob flipped, so the two numbers are directly comparable.
    out = calibrate.validate(**_synthetic(noise=0.3), core_sizes=[6])
    for name in ("no_offset", "curve_iters_1", "fixed_kappa_8"):
        assert out["ablations"][name]["eval_problems"] == out["eval_problems"]
        assert "core_size" not in out["ablations"][name]


def test_sweep_reports_error_by_round_and_by_depth_bucket():
    # A good average can hide a bad bucket; both slices have to exist and cover real data.
    out = calibrate.validate(**_synthetic(noise=0.3), core_sizes=[20])
    fold = out["sweep"][0]
    assert set(fold["cal_error_by_round"]) == {"0", "1"}
    assert set(fold["cal_error_by_depth"]) == {"0", "1", "2", "3"}
    for bucket in fold["cal_error_by_round"].values():
        assert bucket["n"] > 0 and bucket["cal_error"] >= 0.0


# --- n_clipped: fitting's own predict() calls must not pollute the imputation signal ----------
#
# Task 7 gates on "clipped lookups < 2%" as the distribution-shift check -- did the rollouts
# land outside the range the curve was fit on. curve.n_clipped increments on every predict(),
# including the many _solve makes while bisecting an offset, so that counter has to be
# snapshotted before _measure_fit's own impute loop or the gate reads solver noise.


def test_n_clipped_excludes_the_offset_solvers_own_lookups():
    # pid=1's anchors ARE the curve (ramp y = x/10 on [0, 10]); pid=2's anchors sit at score
    # 50, far outside that range, so _solve's own f(lo) probe while fitting pid=2's offset
    # clips repeatedly -- and pid=1's own bisection also explores c values that push x outside
    # [0, 10] before it converges. None of that is a rollout landing out of range.
    ramp_anchors = _anchors([(x, x / 10) for x in range(11)], pid=1)
    far_anchors = _anchors([(50.0, 1.0)] * 4, pid=2)
    measured = {"p1": calibrate.Measured("L", 0.5, 6, 1, 0, 0)}
    rollout_scores = {"p1": [5.0]}          # squarely inside [0, 10] -- no clip at impute time
    out = calibrate.validate(ramp_anchors + far_anchors, rollout_scores, measured,
                             _cfg(curve_bins=4), core_sizes=[1], n_reps=1)
    assert out["n_clipped"] == 0


def test_n_clipped_counts_a_genuinely_out_of_range_imputation():
    ramp_anchors = _anchors([(x, x / 10) for x in range(11)], pid=1)
    measured = {"p1": calibrate.Measured("L", 0.5, 6, 1, 0, 0)}
    rollout_scores = {"p1": [500.0]}        # far outside [0, 10]: a real distribution shift
    out = calibrate.validate(ramp_anchors, rollout_scores, measured, _cfg(curve_bins=4),
                             core_sizes=[1], n_reps=1)
    assert out["n_clipped"] == 1


def test_curve_subset_fit_mirrors_fit_joints_use_anchors_off_branch():
    # _fit_curve_then_offsets is meant to do what fit_joint does, just off two anchor sets
    # instead of one. No caller varies use_anchors through it today, but a silent divergence
    # from fit_joint's own early return would be a trap for the next one that does.
    a = _two_offset_problems()
    curve, offsets, meta = calibrate._fit_curve_then_offsets(a, a, _cfg(use_anchors=False))
    assert offsets == {} and meta["converged"] is True and meta["n_iters"] == 1
    assert len(curve.knots_x) > 1


# --- degenerate cases ------------------------------------------------------------------------


def test_a_curve_core_too_small_to_fit_is_reported_not_a_crash():
    # core_size=0 -> zero anchors to fit the CURVE on -> fit_isotonic's ValueError. The row
    # still lands in `sweep`, named, rather than the sweep silently coming up one entry short.
    out = calibrate.validate(**_synthetic(noise=0.3), core_sizes=[0], n_reps=1)
    fold = out["sweep"][0]
    assert fold["fit_error"] is not None
    assert fold["cal_error"] is None and fold["converged"] is False
    assert len(out["sweep"]) == 1


def test_a_prefix_with_no_partner_in_its_list_reports_no_pairs_not_a_crash():
    # One prefix with no sibling in its list: cal_error is still computable, but there is no
    # pair to agree or disagree on -- None, not a ZeroDivisionError or a NaN.
    anchors = _anchors([(x, x / 10) for x in range(11)], pid=1)
    measured = {"lonely": calibrate.Measured("only_list", 0.5, 6, 1, 0, 0)}
    rollout_scores = {"lonely": [5.0]}
    out = calibrate.validate(anchors, rollout_scores, measured, _cfg(curve_bins=4),
                             core_sizes=[1], n_reps=1)
    assert out["n_pairs"] == 0 and out["pair_agreement"] is None
    assert out["cal_error"] is not None


def test_non_convergence_is_visible_at_the_top_level():
    # The headline is fit_joint over the whole corpus, so non-convergence there shows up
    # directly at the top of validate()'s output, with no need to dig into the sweep.
    a = _two_offset_problems()
    out = calibrate.validate(a, {}, {}, _cfg(curve_iters=1), core_sizes=[2], n_reps=1)
    assert out["converged"] is False


def test_no_anchor_rate_is_zero_when_every_evaluated_problem_has_anchors():
    anchors = (_anchors([(x, x / 10) for x in range(11)], pid=1)
              + _anchors([(x, x / 10) for x in range(11)], pid=2))
    measured = {
        "p1_a": calibrate.Measured("L1", 0.2, 6, 1, 0, 0),
        "p1_b": calibrate.Measured("L1", 0.8, 6, 1, 0, 0),
        "p2_a": calibrate.Measured("L2", 0.1, 6, 2, 0, 0),
        "p2_b": calibrate.Measured("L2", 0.9, 6, 2, 0, 0),
    }
    rollout_scores = {k: [5.0] for k in measured}
    out = calibrate.validate(anchors, rollout_scores, measured, _cfg(curve_bins=4),
                             core_sizes=[1], n_reps=1)
    assert out["no_anchor_rate"] == pytest.approx(0.0)


def test_no_anchor_rate_is_nonzero_when_one_evaluated_problem_has_no_anchors():
    # Only problem "6:1" has anchors; "6:2" is evaluated (it is in `measured`) but contributes
    # nothing to the fit -- the honest model for the ~2.1% of level-6 problems in that spot.
    anchors = _anchors([(x, x / 10) for x in range(11)], pid=1)
    measured = {
        "p1_a": calibrate.Measured("L1", 0.2, 6, 1, 0, 0),
        "p1_b": calibrate.Measured("L1", 0.8, 6, 1, 0, 0),
        "p2_a": calibrate.Measured("L2", 0.1, 6, 2, 0, 0),
        "p2_b": calibrate.Measured("L2", 0.9, 6, 2, 0, 0),
    }
    rollout_scores = {k: [5.0] for k in measured}
    out = calibrate.validate(anchors, rollout_scores, measured, _cfg(curve_bins=4),
                             core_sizes=[1], n_reps=1)
    assert out["eval_problems"] == ["6:1", "6:2"]
    assert out["no_anchor_rate"] == pytest.approx(0.5)      # 1 of 2 evaluated problems
    assert out["n_pairs"] == 2         # both problems' two prefixes still pair within their list


# --- stability: is the number the population's, or one core sample's luck? -------------------


def test_stability_reports_spread_across_independent_core_draws():
    out = calibrate.validate(**_synthetic(noise=0.5, n_problems=30), core_sizes=[10], n_reps=4)
    stab = out["sweep"][0]["stability"]
    assert stab["n_reps"] == 4 and stab["n_usable"] > 0
    assert stab["cal_error_mean"] is not None and stab["cal_error_std"] is not None
    assert stab["pair_agreement_mean"] is not None


def test_stability_draws_differ_from_each_other_and_from_the_canonical_core():
    # Real independent sampling, not the same core replayed four times under a new name.
    fixture = _synthetic(noise=0.5, n_problems=30)
    out = calibrate.validate(**fixture, core_sizes=[10], n_reps=4, seed=7)
    canonical = set(out["sweep"][0]["core_problems"])
    problems = sorted({a.pkey for a in fixture["anchors"]})
    draws = []
    for child in np.random.SeedSequence([7, 10]).spawn(4):
        core = set(np.random.default_rng(child).choice(problems, size=10, replace=False).tolist())
        draws.append(core)
    assert len({frozenset(d) for d in draws}) > 1        # the reps are not all identical
    assert any(d != canonical for d in draws)             # nor just the canonical core again


# --- the ORM-vs-length diagnostic -------------------------------------------------------------


def test_length_control_flags_a_length_only_orm():
    # score == length, target ~ f(length-bucket) + noise. Overall accuracy is high (length
    # tracks the bucket); within a length decile the bucket is ~constant, so only noise is
    # left to order pairs -- accuracy there should collapse toward chance.
    rng = np.random.default_rng(1)
    anchors = []
    for i in range(400):
        length = float(rng.uniform(0, 1000))
        bucket = int(length // 250)
        target = float(np.clip(bucket / 3.0 + rng.normal(0, 0.15), 0.0, 1.0))
        anchors.append(calibrate.Anchor(f"x{i}", 6, 1, length, target, False,
                                        n_code_tokens=int(length)))
    out = calibrate._length_control(anchors)
    assert out["overall"] is not None and out["decile_mean"] is not None
    assert out["overall"] > out["decile_mean"] + 0.15


def test_length_control_handles_a_single_length_without_crashing():
    anchors = [calibrate.Anchor(f"x{i}", 6, 1, float(i), float(i % 2), False, n_code_tokens=10)
               for i in range(5)]
    out = calibrate._length_control(anchors)
    assert out["decile_mean"] is None and out["n_deciles_used"] == 0


def test_load_anchors_carries_the_score_rows_code_length(tmp_path):
    rows = [_row(sid=0)]
    scores = [_score_row(rows[0], 0.5)]           # _score_row sets n_code_tokens=3
    cfg = _campaign(tmp_path, rows, scores)
    anchors, _ = calibrate.load_anchors(cfg)
    assert anchors[0].n_code_tokens == 3


# --- the dead set ---------------------------------------------------------------------------
#
# A problem no anchor of which ever passed. Every V̂ in its lists is then the ORM's invention
# and so is their order -- and lists.py's all-equal drop cannot catch it, because imputed V̂
# are floats and never tie. Decided here rather than in lists.py because this is the only job
# that loads the anchors.


def _dead_campaign(tmp_path, **over):
    """Problem 1 never passes; problem 2 passes twice."""
    rows, scores = [], []
    for pid, passing in ((1, ()), (2, (1, 2))):
        for sid in range(3):
            correct = sid in passing
            row = _row(pid=pid, sid=sid, correct=correct,
                       speedup_min=2.0 if correct else None,
                       speedup=2.0 if correct else None)
            rows.append(row)
            scores.append(_score_row(row, 1.0 if correct else -2.0))
    return _campaign(tmp_path, rows, scores, curve_bins=2, **over)


def test_dead_problems_names_only_the_problem_whose_anchors_all_failed():
    anchors = (_anchors([(-3.0, 0.0), (-2.0, 0.0)], pid=1)
               + _anchors([(-3.0, 0.0), (1.0, 0.6)], pid=2))
    assert calibrate.dead_problems(anchors) == ["6:1"]


def test_one_passing_anchor_is_enough_to_keep_a_problem_alive():
    # The predicate is evidence of failure, not weight of it: 11 failures and one pass is a
    # problem the policy can solve, and its lists rank something real.
    anchors = _anchors([(-3.0, 0.0)] * 11 + [(2.0, 0.05)], pid=7)
    assert calibrate.dead_problems(anchors) == []


def test_calibrate_writes_the_dead_set_and_counts_it_in_the_manifest(tmp_path):
    cfg = _dead_campaign(tmp_path)
    manifest = calibrate.calibrate(cfg)
    path = os.path.join(cfg.prm_rollout.out_dir, calibrate.OFFSETS)
    assert calibrate.read_dead(path) == {"6:1"}
    assert manifest["dead_problems"] == 1


def test_the_dead_set_survives_the_offsets_off_ablation(tmp_path):
    # Why it sits beside `offsets` rather than inside it: use_anchors=false writes offsets {},
    # which would take every per-problem field with it -- including one that is not an offset.
    cfg = _dead_campaign(tmp_path, use_anchors=False)
    calibrate.calibrate(cfg)
    path = os.path.join(cfg.prm_rollout.out_dir, calibrate.OFFSETS)
    assert calibrate.read_offsets(path)[0] == {}
    assert calibrate.read_dead(path) == {"6:1"}


def test_an_offsets_file_written_before_the_dead_set_existed_reads_as_empty(tmp_path):
    # Not a default that hides a bug: lists.py only consults this when drop_dead_problems is
    # on, and an empty set there means "drop nothing", which is the pre-existing behaviour.
    path = tmp_path / "offsets.json"
    path.write_text(json.dumps({"meta": {}, "offsets": {}}))
    assert calibrate.read_dead(str(path)) == set()
