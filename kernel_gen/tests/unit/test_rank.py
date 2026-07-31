"""Which attempt a slot ships.

``Trajectory.final()`` is the loop's non-regression guard: refinement can make a kernel
worse, so the slot writes the first clean attempt, or failing that the best one by
``_rank``. KGEN-14 adds a term to that ordering -- an attempt the evaluator cannot load
loses to one it can, mirroring the "does not parse loses to parses" term above it.

This is the one change in the fix that alters which file ships for a trajectory that is
not otherwise broken, so it gets its own tests.
"""

from __future__ import annotations

from kernel_gen.core.model import Attempt, Problem, Review, Trajectory

PROBLEM = Problem(level=1, problem_id=1, name="1_Add.py", ref_arch_src="")


def review(*, loadable: bool = True, n_fail: int = 0, n_warn: int = 0, clean: bool = False):
    return Review(
        text="" if clean else "feedback",
        clean=clean,
        data={
            "parse_status": "ok",
            "n_fail": n_fail,
            "n_warn": n_warn,
            "check_ids": [],
            "submission_ok": loadable,
        },
    )


def trajectory(*attempts: Attempt) -> Trajectory:
    return Trajectory(problem=PROBLEM, sample_id=0, attempts=list(attempts))


class TestLoadabilityOutranksFindingCount:
    def test_a_loadable_attempt_beats_an_unloadable_one(self):
        broken = Attempt(round=0, raw="", code="x", review=review(loadable=False))
        works = Attempt(round=1, raw="", code="y", review=review(loadable=True, n_fail=3))

        assert trajectory(broken, works).final() is works

    def test_even_when_the_unloadable_one_lints_perfectly(self):
        """A file that cannot be imported scores zero however clean it lints, so no
        number of avoided findings can make it the better answer."""
        broken = Attempt(round=0, raw="", code="x", review=review(loadable=False))
        works = Attempt(
            round=1, raw="", code="y", review=review(loadable=True, n_fail=9, n_warn=9)
        )

        assert trajectory(broken, works).final() is works

    def test_even_when_the_unloadable_one_came_first(self):
        works = Attempt(round=2, raw="", code="y", review=review(loadable=True))
        broken = Attempt(round=0, raw="", code="x", review=review(loadable=False))

        assert trajectory(broken, works).final() is works


class TestTheRestOfTheOrderingIsUnchanged:
    def test_fewer_fails_still_wins_between_two_loadable_attempts(self):
        worse = Attempt(round=0, raw="", code="x", review=review(n_fail=3))
        better = Attempt(round=1, raw="", code="y", review=review(n_fail=1))

        assert trajectory(worse, better).final() is better

    def test_fails_still_outrank_warns(self):
        many_warns = Attempt(round=0, raw="", code="x", review=review(n_warn=9))
        one_fail = Attempt(round=1, raw="", code="y", review=review(n_fail=1))

        assert trajectory(one_fail, many_warns).final() is many_warns

    def test_ties_still_break_toward_the_earliest_round(self):
        first = Attempt(round=0, raw="", code="x", review=review(n_fail=1))
        second = Attempt(round=1, raw="", code="y", review=review(n_fail=1))

        assert trajectory(first, second).final() is first

    def test_a_clean_attempt_still_wins_outright(self):
        clean = Attempt(round=0, raw="", code="x", review=review(clean=True))
        later = Attempt(round=1, raw="", code="y", review=review(n_fail=0))

        assert trajectory(clean, later).final() is clean

    def test_an_attempt_with_no_review_is_ranked_on_having_code(self):
        empty = Attempt(round=0, raw="", code="   ", review=None)
        has_code = Attempt(round=1, raw="", code="x = 1", review=None)

        assert trajectory(empty, has_code).final() is has_code


class TestBackwardCompatibility:
    def test_a_review_without_the_key_is_treated_as_loadable(self):
        """Journals written before this change carry no `submission_ok`. Defaulting it to
        False would rank every historical attempt as unloadable."""
        old = Attempt(
            round=0,
            raw="",
            code="x",
            review=Review(text="", clean=False, data={"parse_status": "ok", "n_fail": 0}),
        )
        worse = Attempt(round=1, raw="", code="y", review=review(n_fail=5))

        assert trajectory(old, worse).final() is old


def test_an_attempt_whose_critic_never_ran_is_not_presumed_unloadable():
    """A crashed or skipped critic is missing evidence, not evidence of a defect. Ranking
    it below an attempt we positively know cannot be loaded would let a real failure win
    on a tie-break."""
    unloadable = Attempt(round=0, raw="", code="x", review=review(loadable=False))
    unreviewed = Attempt(round=1, raw="", code="x = 1", review=None)

    assert trajectory(unloadable, unreviewed).final() is unreviewed
