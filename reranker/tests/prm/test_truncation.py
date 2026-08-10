"""The rule that protects label validity — and the fence heuristic that would destroy it.

A generation stopped by `max_new_tokens` is a fragment, so it cannot compile, so it scores 0
*regardless of how good its prefix was*. That is systematic error correlated with length, not
the unbiased noise the whole method rests on, so the entire sample goes (PLAN §2).
"""

from __future__ import annotations

from collections import Counter

import pytest

from reranker.src.prm.corpus import OK, TRUNCATED, UNKNOWN, iter_labeled, truncation_state
from reranker.tests.prm.runfixture import (
    LENGTH,
    SINGLE_LENGTH,
    SINGLE_STOP,
    STOP,
    attempt,
    one_unit,
    verdict,
)

# gpt-oss: the two-pass sampler injects the opening fence and the model runs to EOS, so 99%
# of that run ends mid-fence with nothing wrong. 7,514 of them are correct kernels.
UNTERMINATED = "## Plan\nfuse the reduction\n\n```python\nimport triton\nx = 1\n"
TERMINATED = UNTERMINATED + "```\n"


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        (STOP, OK),
        (LENGTH, TRUNCATED),
        ({"passes": 2, "plan_finish_reason": "length", "code_finish_reason": "stop"}, TRUNCATED),
        ({"passes": 2, "plan_finish_reason": "length", "code_finish_reason": "length"}, TRUNCATED),
        (None, UNKNOWN),
        ({}, UNKNOWN),
        ({"passes": 2, "plan_finish_reason": "stop"}, UNKNOWN),
        ({"passes": 2, "code_finish_reason": "stop"}, UNKNOWN),
        ({"passes": 2, "plan_finish_reason": "stop", "code_finish_reason": None}, UNKNOWN),
    ],
    ids=[
        "both_stop",
        "code_length",
        "plan_length",
        "both_length",
        "no_trace",
        "empty_trace",
        "code_field_absent",
        "plan_field_absent",
        "code_field_null",
    ],
)
def test_the_state_table(trace, expected):
    # Both passes are checked because generation is two-pass and either can hit the budget.
    assert truncation_state(trace) == expected


# --- the other generation shape: --think-temperature 0 writes no plan reason -----------


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        (SINGLE_STOP, OK),
        (SINGLE_LENGTH, TRUNCATED),
        ({"passes": 1}, UNKNOWN),
        ({"passes": 1, "code_finish_reason": None}, UNKNOWN),
    ],
    ids=["stop", "length", "code_field_absent", "code_field_null"],
)
def test_a_single_pass_record_is_read_off_its_one_reason(trace, expected):
    # sampling.py::_single_pass emits code_finish_reason only. Demanding a plan reason it
    # never writes marked 100% of such a run `unknown` -- including this real shape, where
    # vLLM said "length" outright, so the ledger blamed the tracer instead of the budget.
    assert truncation_state(trace) == expected


def test_the_shortcut_does_not_leak_into_two_pass_records():
    # The branch is on `passes`, not on which fields happen to be present: a two-pass
    # record whose code reason failed to write must not read `ok` off the plan half.
    assert truncation_state({"passes": 2, "plan_finish_reason": "stop"}) == UNKNOWN
    assert truncation_state({"plan_finish_reason": "stop"}) == UNKNOWN


def test_a_record_with_no_passes_field_is_treated_as_two_pass():
    # Predates the field; behaves exactly as before, so no old record changes state.
    assert truncation_state({"plan_finish_reason": "stop", "code_finish_reason": "stop"}) == OK
    assert truncation_state({"code_finish_reason": "stop"}) == UNKNOWN


@pytest.mark.parametrize(
    "passes",
    [3, 0, None, "2", True, False, 1.0, 2.0],
    ids=["three", "zero", "null", "string", "true", "false", "float_one", "float_two"],
)
def test_a_pass_count_this_module_does_not_know_is_unknown(passes):
    # A third pass that hit the budget would otherwise read `ok` off the two reasons named
    # here, and train a fragment at target 0.0 -- the length-correlated error §2 excludes.
    # `True` and `1.0` are here because both hash equal to 1 and so would silently pick the
    # single-pass table, which is the shortcut this record has not earned.
    trace = {
        "passes": passes,
        "plan_finish_reason": "stop",
        "code_finish_reason": "stop",
        "critic_finish_reason": "length",
    }
    assert truncation_state(trace) == UNKNOWN


def test_an_unrecognised_finish_reason_is_truncated_not_ok():
    # Only "stop" means the model finished on its own; anything else is a forced end.
    bad = {"plan_finish_reason": "stop", "code_finish_reason": "abort"}
    assert truncation_state(bad) == TRUNCATED


# --- the trap: an unterminated fence is formatting, never truncation -------------------


@pytest.mark.parametrize("raw", [UNTERMINATED, TERMINATED], ids=["unterminated", "terminated"])
def test_the_verdict_is_the_same_whatever_the_text_does_with_its_fence(tmp_path, raw):
    unit = one_unit(tmp_path, [attempt(1, 0, raw=raw)], {"1": [verdict(0)]})
    (row,) = iter_labeled(unit, Counter())
    assert row.truncation == OK
    assert row.raw == raw


def test_an_unterminated_fence_that_finished_on_its_own_is_kept(tmp_path):
    # The regression guard. This shape is 99% of gpt-oss; a future "detect truncation from
    # the fence" refactor flags 38,612 attempts and destroys 7,575 correct kernels, and it
    # must fail here rather than in production.
    unit = one_unit(tmp_path, [attempt(1, 0, raw=UNTERMINATED)], {"1": [verdict(0)]})
    rows = list(iter_labeled(unit, Counter()))
    assert [(r.truncation, r.correct) for r in rows] == [(OK, True)]


def test_a_terminated_fence_that_hit_the_budget_is_still_truncated(tmp_path):
    # The other half of the same point: closing the fence proves nothing either.
    attempts = [attempt(1, 0, raw=TERMINATED, trace=LENGTH)]
    unit = one_unit(tmp_path, attempts, {"1": [verdict(0)]})
    rows = list(iter_labeled(unit, Counter()))
    assert rows[0].truncation == TRUNCATED


# --- the state reaches the row build.py drops on --------------------------------------


def test_every_row_carries_the_state_so_build_can_drop_on_it_first(tmp_path):
    attempts = [
        attempt(1, 0),
        attempt(1, 1, trace=LENGTH),
        attempt(1, 2, trace=None),
    ]
    verdicts = {"1": [verdict(i) for i in range(3)]}
    rows = list(iter_labeled(one_unit(tmp_path, attempts, verdicts), Counter()))
    assert [r.truncation for r in rows] == [OK, TRUNCATED, UNKNOWN]
    # Dropping is build.py's job; corpus reports the state and keeps the row.
    assert len(rows) == 3
