"""``select``: PRM scores over live candidates, and the top-B cut that keeps parents diverse."""

from __future__ import annotations

import random

from processkernel.generation.core.model import Problem
from processkernel.config import SEL_RANDOM, PRMSearchConfig
from processkernel.prm.search import select
from processkernel.prm.search.candidate import Candidate

PROBLEM = Problem(level=6, problem_id=12, name="12_Thing.py", ref_arch_src="")


def cand(parent: str, cid: str) -> Candidate:
    return Candidate(problem=PROBLEM, head="H", prompt="P", text="## Plan\nx\n",
                     cid=cid, parent=parent)


class _Encoder:
    def encode_text(self, text):
        return [len(text)]


def test_the_prm_selector_scores_through_the_encoder():
    seen = []

    def scorer(ids):
        seen.extend(ids)
        return [1.0] * len(ids)

    cands = [cand("p", "a"), cand("p", "b")]
    out = select.score(cands, PRMSearchConfig(prm_checkpoint="/c"), scorer, _Encoder(),
                       random.Random(0))
    assert out == [1.0, 1.0]
    assert len(seen) == 2


def test_the_random_selector_never_touches_the_model():
    def scorer(ids):
        raise AssertionError("the random control must not call the PRM")

    cands = [cand("p", "a"), cand("p", "b")]
    out = select.score(cands, PRMSearchConfig(selector=SEL_RANDOM), scorer, _Encoder(),
                       random.Random(0))
    assert len(out) == 2


def test_the_random_selector_is_reproducible_from_the_seed():
    cands = [cand("p", str(i)) for i in range(6)]
    conf = PRMSearchConfig(selector=SEL_RANDOM)
    a = select.score(cands, conf, None, None, random.Random(7))
    b = select.score(cands, conf, None, None, random.Random(7))
    assert a == b


def test_top_b_keeps_the_best_and_returns_the_rest():
    cands = [cand("p", "a"), cand("p", "b"), cand("p", "c")]
    kept, dropped = select.top_b(cands, [0.1, 0.9, 0.5],
                                 PRMSearchConfig(beam_width=2, min_distinct_parents=0,
                                                 prm_checkpoint="/c"))
    assert [c.cid for c in kept] == ["b", "c"]
    assert [c.cid for c in dropped] == ["a"]


def test_the_diversity_floor_swaps_the_weakest_for_an_unrepresented_parent():
    cands = [cand("p1", "a"), cand("p1", "b"), cand("p2", "c")]
    kept, _ = select.top_b(cands, [0.9, 0.8, 0.1],
                           PRMSearchConfig(beam_width=2, min_distinct_parents=2,
                                           prm_checkpoint="/c"))
    assert {c.cid for c in kept} == {"a", "c"}


def test_the_floor_is_skipped_when_the_top_already_spans_enough_parents():
    cands = [cand("p1", "a"), cand("p2", "b"), cand("p3", "c")]
    kept, _ = select.top_b(cands, [0.9, 0.8, 0.1],
                           PRMSearchConfig(beam_width=2, min_distinct_parents=2,
                                           prm_checkpoint="/c"))
    assert {c.cid for c in kept} == {"a", "b"}


def test_the_floor_gives_up_rather_than_failing_when_there_is_only_one_parent():
    cands = [cand("p1", "a"), cand("p1", "b"), cand("p1", "c")]
    kept, _ = select.top_b(cands, [0.9, 0.8, 0.1],
                           PRMSearchConfig(beam_width=2, min_distinct_parents=2,
                                           prm_checkpoint="/c"))
    assert {c.cid for c in kept} == {"a", "b"}


def test_kept_and_dropped_together_are_every_candidate_exactly_once():
    cands = [cand(f"p{i % 3}", str(i)) for i in range(9)]
    scores = [i / 10 for i in range(9)]
    kept, dropped = select.top_b(cands, scores,
                                 PRMSearchConfig(beam_width=4, prm_checkpoint="/c"))
    assert len(kept) == 4
    assert sorted(c.cid for c in kept + dropped) == sorted(c.cid for c in cands)


def test_ties_break_deterministically():
    cands = [cand("p", str(i)) for i in range(5)]
    conf = PRMSearchConfig(beam_width=2, min_distinct_parents=0, prm_checkpoint="/c")
    a, _ = select.top_b(cands, [0.5] * 5, conf)
    b, _ = select.top_b(cands, [0.5] * 5, conf)
    assert [c.cid for c in a] == [c.cid for c in b]


def test_a_beam_smaller_than_the_width_keeps_everything():
    cands = [cand("p", "a")]
    kept, dropped = select.top_b(cands, [0.5],
                                 PRMSearchConfig(beam_width=4, min_distinct_parents=0,
                                                 prm_checkpoint="/c"))
    assert len(kept) == 1 and dropped == []


def grouped(parent: str, cid: str, group: int) -> Candidate:
    c = cand(parent, cid)
    c.group = group
    return c


def test_each_sub_beam_keeps_its_own_best_even_when_another_group_scores_higher():
    cands = [grouped("p0", "a", 0), grouped("p0", "b", 0), grouped("p1", "c", 1), grouped("p1", "d", 1)]
    kept, dropped = select.top_b(cands, [0.9, 0.8, 0.2, 0.1],
                                 PRMSearchConfig(beam_width=2, beam_groups=2, prm_checkpoint="/c"))
    assert {c.cid for c in kept} == {"a", "c"}
    assert {c.cid for c in dropped} == {"b", "d"}


def test_one_group_is_the_plain_beam():
    cands = [cand("p1", "a"), cand("p1", "b"), cand("p2", "c")]
    conf = PRMSearchConfig(beam_width=2, min_distinct_parents=2, beam_groups=1, prm_checkpoint="/c")
    kept, _ = select.top_b(cands, [0.9, 0.8, 0.1], conf)
    assert {c.cid for c in kept} == {"a", "c"}


def test_a_finished_group_leaves_its_slots_empty_instead_of_lending_them():
    cands = [grouped("p0", "a", 0), grouped("p0", "b", 0), grouped("p0", "c", 0)]
    kept, _ = select.top_b(cands, [0.9, 0.8, 0.7],
                           PRMSearchConfig(beam_width=4, beam_groups=4, prm_checkpoint="/c"))
    assert [c.cid for c in kept] == ["a"]
