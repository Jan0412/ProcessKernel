"""compare_arms: correctness difference between one A/B arm and the control."""

from __future__ import annotations

import math

from kernel_gen.arm_stats import compare

# {problem_id: [bool, ...]} -- one entry per sample slot
CONTROL = {1: [True, False, False, False], 2: [False] * 4, 3: [True, True, False, False]}
ARM = {1: [True, True, False, False], 2: [False] * 4, 3: [True, True, True, False]}


def test_counts_every_slot_when_nothing_is_excluded():
    r = compare(CONTROL, ARM, exclude=set())
    assert r["n_control"] == 12
    assert r["n_arm"] == 12


def test_reports_the_correctness_rate_of_each_side():
    r = compare(CONTROL, ARM, exclude=set())
    assert r["p_control"] == 3 / 12
    assert r["p_arm"] == 5 / 12


def test_excluded_problems_are_dropped_from_both_sides():
    r = compare(CONTROL, ARM, exclude={2})
    assert r["n_control"] == 8
    assert r["p_control"] == 3 / 8


def test_difference_is_arm_minus_control():
    r = compare(CONTROL, ARM, exclude=set())
    assert math.isclose(r["diff"], 5 / 12 - 3 / 12)


def test_standard_error_matches_the_two_proportion_formula():
    # Pins the variance formula itself against hand-computed numbers; asserting only
    # sigma == diff/se would pass for any se whatsoever.
    r = compare(CONTROL, ARM, exclude=set())
    p_c, p_a, n = 3 / 12, 5 / 12, 12
    expected = math.sqrt(p_c * (1 - p_c) / n + p_a * (1 - p_a) / n)
    assert math.isclose(r["se"], expected)


def test_sigma_is_the_difference_over_its_standard_error():
    r = compare(CONTROL, ARM, exclude=set())
    assert math.isclose(r["sigma"], r["diff"] / r["se"])


def test_zero_variance_does_not_divide_by_zero():
    allwrong = {1: [False, False]}
    r = compare(allwrong, allwrong, exclude=set())
    assert r["diff"] == 0.0
    assert r["sigma"] == 0.0


def test_only_problems_present_in_both_arms_are_compared():
    r = compare({1: [True, False]}, {1: [True, True], 9: [True, True]}, exclude=set())
    assert r["n_control"] == 2
    assert r["n_arm"] == 2
