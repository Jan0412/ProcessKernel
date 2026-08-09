"""``prm.targets``: the graded scale, its empty band, and the knobs that regrade it."""

from __future__ import annotations

import random

import pytest

from reranker.src.data.labels import speed_p
from reranker.src.prm.targets import (
    BINARY,
    GRADED,
    MEAN,
    MIN,
    graded_target,
    speedups,
    target_for,
)

LO, HI, QUANT = 0.2, 4.0, 0.1
LADDER = [round(0.5 + 0.05 * i, 2) for i in range(11)]  # 0.50, 0.55, ... 1.00
SEED = 20260809


def grade(speedup: float = 1.0, **over):
    """One evaluated sample whose baseline/runtime ratio is exactly ``speedup``."""
    kw = dict(
        compiled=True,
        correct=True,
        runtime=1.0,
        runtime_stats={"min": 1.0},
        baseline={"mean": speedup, "min": speedup},
        mode=GRADED,
        stat=MEAN,
        lo=LO,
        hi=HI,
        quant=QUANT,
    )
    kw.update(over)
    return target_for(**kw)


# --- the graded endpoints (PLAN §5) --------------------------------------------------


def test_a_wrong_kernel_scores_zero():
    assert grade(correct=False).value == 0.0
    assert grade(compiled=False, correct=False).value == 0.0


@pytest.mark.parametrize(
    ("speedup", "expected"),
    [(0.2, 0.50), (1.0, 0.75), (4.0, 1.00)],
)
def test_the_three_named_speedups_grade_as_the_plan_says(speedup, expected):
    assert grade(speedup).value == pytest.approx(expected)


def test_speedups_outside_the_range_clamp_to_the_endpoints():
    assert grade(0.001).value == pytest.approx(0.50)
    assert grade(1000.0).value == pytest.approx(1.00)


def _speedups(rng, n=2000):
    """Log-uniform over 1/16x .. 16x -- the scale speedups actually live on."""
    return [2 ** rng.uniform(-4, 4) for _ in range(n)]


def test_nothing_ever_lands_in_the_open_band_below_a_half():
    rng = random.Random(SEED)
    for speedup in _speedups(rng):
        t = grade(speedup, correct=rng.random() < 0.5)
        assert t.value == 0.0 or t.value >= 0.5


def test_correct_kernels_land_only_on_the_eleven_value_ladder():
    seen = set()
    for speedup in _speedups(random.Random(SEED)):
        value = grade(speedup).value
        assert round(value, 6) in LADDER
        seen.add(round(value, 2))
    assert seen == set(LADDER)


def test_the_grade_is_speed_p_shifted_into_the_upper_half():
    # ARCHITECTURE S5: one chain grades everything, and speed_p is its only scale.
    for speedup in (0.05, 0.31, 1.0, 2.7, 9.0):
        p = speed_p(speedup, LO, HI, QUANT)
        assert graded_target(speedup, lo=LO, hi=HI, quant=QUANT) == pytest.approx((1.0 + p) / 2)
        assert grade(speedup).value == pytest.approx((1.0 + p) / 2)


def test_the_grade_never_decreases_as_the_kernel_gets_faster():
    values = [grade(su).value for su in (0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0)]
    assert values == sorted(values)


def test_without_quantization_the_scale_is_continuous():
    off = grade(1.0, quant=0.0).value
    assert off == pytest.approx(0.7686, abs=1e-4)
    assert round(off, 6) not in LADDER


# --- the knobs (PLAN §7: regrading is a pass over the parts, not a rebuild) ----------


def test_binary_mode_returns_only_zero_and_one():
    assert grade(4.0, mode=BINARY).value == 1.0
    assert grade(4.0, mode=BINARY, correct=False).value == 0.0


def test_binary_mode_needs_no_baseline_at_all():
    t = grade(mode=BINARY, baseline=None)
    assert t is not None and t.value == 1.0 and t.speedup is None


def test_speedup_stat_switches_which_ratio_is_graded():
    kw = dict(baseline={"mean": 1.0, "min": 4.0}, runtime=1.0, runtime_stats={"min": 1.0})
    assert grade(stat=MEAN, **kw).value == pytest.approx(0.75)
    assert grade(stat=MIN, **kw).value == pytest.approx(1.00)


def test_the_bounds_move_the_whole_scale():
    assert grade(1.0, lo=1.0, hi=4.0).value == pytest.approx(0.50)
    assert grade(1.0, lo=0.25, hi=1.0).value == pytest.approx(1.00)


@pytest.mark.parametrize(("key", "bad"), [("mode", "raw"), ("stat", "median")])
def test_an_unknown_mode_or_stat_raises_rather_than_grading_something_else(key, bad):
    with pytest.raises(ValueError, match=bad):
        grade(**{key: bad})


@pytest.mark.parametrize(("lo", "hi"), [(4.0, 0.2), (1.0, 1.0), (0.0, 4.0), (-1.0, 4.0)])
def test_bounds_that_would_invert_or_collapse_the_scale_raise(lo, hi):
    with pytest.raises(ValueError, match="speedup_lo"):
        grade(1.0, lo=lo, hi=hi)
    with pytest.raises(ValueError, match="speedup_lo"):
        graded_target(1.0, lo=lo, hi=hi, quant=QUANT)


def test_bad_bounds_raise_on_the_first_row_even_when_it_is_a_wrong_kernel():
    # ~87% of rows are wrong; if only those reached the grader the build would write most
    # of its output before the bad config surfaced.
    with pytest.raises(ValueError, match="speedup_lo"):
        grade(correct=False, lo=4.0, hi=0.2)


@pytest.mark.parametrize("bad", [-0.1, -1.0, 1.5, 10.0, float("nan")])
def test_a_quant_outside_zero_to_one_raises_rather_than_regrading_in_silence(bad):
    # speed_p quantizes only when quant > 0: a negative one does not fail, it turns the
    # ladder off (11 values -> 1039 over 2000 log-uniform speedups). `speed_quant: 10`
    # for "10%" is the other direction -- every correct kernel collapses onto 0.50.
    with pytest.raises(ValueError, match="speed_quant"):
        grade(1.0, quant=bad)
    with pytest.raises(ValueError, match="speed_quant"):
        graded_target(1.0, lo=LO, hi=HI, quant=bad)


def test_a_bad_quant_raises_on_the_first_row_even_when_it_is_a_wrong_kernel():
    with pytest.raises(ValueError, match="speed_quant"):
        grade(correct=False, quant=-0.1)


def test_both_quant_endpoints_are_legal():
    assert grade(1.0, quant=0.0).value == pytest.approx(0.7686, abs=1e-4)  # off
    assert grade(1.0, quant=1.0).value == 1.0  # one step for the whole range


def test_binary_mode_does_not_care_about_the_bounds_or_quant_it_never_uses():
    assert grade(1.0, mode=BINARY, lo=4.0, hi=0.2).value == 1.0
    assert grade(1.0, mode=BINARY, quant=-0.1).value == 1.0


# --- missing inputs are drops, never substituted values ------------------------------


def test_a_correct_kernel_with_no_baseline_is_dropped():
    assert grade(baseline=None) is None
    assert grade(baseline={}) is None


def test_a_correct_kernel_whose_baseline_lacks_the_chosen_stat_is_dropped():
    assert grade(stat=MIN, baseline={"mean": 1.0}) is None
    assert grade(stat=MEAN, baseline={"mean": 1.0}) is not None


def test_a_correct_kernel_with_no_runtime_for_the_chosen_stat_is_dropped():
    assert grade(stat=MIN, runtime_stats={}) is None
    assert grade(stat=MIN, runtime_stats=None) is None
    assert grade(stat=MEAN, runtime=None) is None


def test_a_wrong_kernel_survives_a_missing_baseline():
    t = grade(correct=False, baseline=None)
    assert t is not None and t.value == 0.0


INF = float("inf")


@pytest.mark.parametrize(
    ("base", "ours", "expected", "was"),
    [
        (1.0, INF, None, "crash: speed_p took log2 of 0.0 -> ValueError math domain error"),
        (INF, 1.0, None, "graded 1.00 -- garbage scored a perfect kernel"),
        (INF, INF, None, "graded 0.50 -- inf/inf is nan, and speed_p clamps nan to p=0"),
        (2.0, 1.0, 0.90, "graded 0.90 -- an ordinary row, and it must not move"),
    ],
    ids=["ratio_zero", "ratio_inf", "ratio_nan", "ordinary"],
)
def test_a_ratio_that_is_not_an_ordinary_number_is_a_drop_not_a_grade(base, ours, expected, was):
    # Every row here passes the `base > 0 and kernel > 0` guard on the *inputs*; only the
    # quotient separates them, which is why the check has to be on the ratio itself.
    t = target_for(
        compiled=True,
        correct=True,
        runtime=ours,
        runtime_stats={"min": ours},
        baseline={"mean": base, "min": base},
        mode=GRADED,
        stat=MEAN,
        lo=LO,
        hi=HI,
        quant=QUANT,
    )
    if expected is None:
        assert t is None, was
    else:
        assert t.value == pytest.approx(expected), was


# --- the fields the row carries for regrading ----------------------------------------


def test_both_speedups_are_reported_for_a_correct_kernel():
    t = grade(baseline={"mean": 2.0, "min": 8.0}, runtime=1.0, runtime_stats={"min": 2.0})
    assert (t.speedup, t.speedup_min) == (2.0, 4.0)
    assert t.label == 1


def test_a_wrong_kernel_reports_no_speedup_even_with_a_runtime():
    t = grade(correct=False, runtime=1.0)
    assert (t.speedup, t.speedup_min, t.label) == (None, None, 0)


def test_a_kernel_that_did_not_compile_is_labelled_zero():
    assert grade(compiled=False, correct=False).label == 0


def ratio(correct=True, runtime=1.0, runtime_stats=None, baseline=None):
    return speedups(
        correct=correct,
        runtime=runtime,
        runtime_stats={"min": 1.0} if runtime_stats is None else runtime_stats,
        baseline={"mean": 1.0, "min": 1.0} if baseline is None else baseline,
    )


NEITHER = {MEAN: None, MIN: None}


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_runtime_or_baseline_yields_no_speedup(bad):
    assert ratio(runtime=bad, runtime_stats={"min": bad}) == NEITHER
    assert ratio(baseline={"mean": bad, "min": bad}) == NEITHER


def test_speedups_of_a_wrong_kernel_are_none_whatever_the_runtimes():
    assert ratio(correct=False, baseline={"mean": 4.0, "min": 4.0}) == NEITHER


def test_neither_stat_survives_an_ungradeable_ratio():
    assert ratio(runtime=INF, runtime_stats={"min": INF}) == NEITHER  # 1.0 / inf -> 0.0
    assert ratio(baseline={"mean": INF, "min": INF}) == NEITHER  # inf / 1.0 -> inf


def test_integer_runtimes_from_json_still_divide_as_floats():
    assert ratio(runtime=2, runtime_stats={"min": 2}, baseline={"mean": 1, "min": 3}) == {
        MEAN: 0.5,
        MIN: 1.5,
    }
