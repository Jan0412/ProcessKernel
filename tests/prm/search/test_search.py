"""``search``: the step-major loop -- every problem's beam advances in one batched pass."""

from __future__ import annotations

import random
from collections import Counter

from processkernel.generation.core.backend import FakeBackend
from processkernel.generation.core.model import Problem
from processkernel.generation.core.sampling import CODE_FENCE
from processkernel.config import SEL_RANDOM, PRMRolloutConfig, PRMSearchConfig
from processkernel.prm.search import search as S
from processkernel.prm.search.candidate import pkey

PROBLEMS = [
    Problem(level=6, problem_id=i, name=f"{i}_T.py", ref_arch_src="import torch\n")
    for i in range(3)
]
NEVER_ENDS = "reasoning\n" + CODE_FENCE + "\n" + "".join(f"x{i} = {i}\n" for i in range(200))
ENDS = "reasoning\n" + CODE_FENCE + "\nimport torch\nclass ModelNew: pass\n```\n"


def count(text: str) -> int:
    return (len(text) + 3) // 4


class _Encoder:
    def encode_text(self, text):
        return [len(text)]


class Finishing(FakeBackend):
    """FakeBackend with a scripted finish_reason -- plain FakeBackend fixes it at "stop", so
    a NEVER_ENDS candidate would look naturally finished at step 0 and never reach a later
    step (see test_segment.py's copy of this class)."""

    def __init__(self, reason, **kw):
        super().__init__(**kw)
        self.reason = reason

    def complete_traced(self, prompts, **kw):
        out = super().complete_traced(prompts, **kw)
        for completion in out:
            completion.finish_reason = self.reason
        return out


def run(backend, **over):
    conf = PRMSearchConfig(selector=SEL_RANDOM, **over)
    rc = PRMRolloutConfig(max_new_tokens=4096, temperature=0.8, think_temperature=1.0)
    return S.search(backend, PROBLEMS, conf, rc, count, None, _Encoder(), "SYS",
                    lambda p: f"solve {p.problem_id}", random.Random(0))


def test_one_generation_call_per_step_covers_every_problem():
    backend = Finishing("length", default=NEVER_ENDS)
    run(backend, max_steps=3, beam_width=4, expand=4)
    # step 0 is one plan call plus one code call; later steps are code only. Never 3x that
    # for the three problems.
    assert len(backend.batches) <= 2 * 3
    assert len(backend.batches[0]) == 3 * 4      # one root each, beam_width children


def test_later_steps_expand_every_survivor():
    backend = Finishing("length", default=NEVER_ENDS)
    run(backend, max_steps=2, beam_width=4, expand=4)
    assert max(len(b) for b in backend.batches) == 3 * 4 * 4


def test_the_beam_never_exceeds_its_width():
    backend = Finishing("length", default=NEVER_ENDS)
    result = run(backend, max_steps=3, beam_width=2, expand=4, min_distinct_parents=0)
    for step in result.steps:
        per = Counter(r["pkey"] for r in step["rows"] if r["kept"])
        assert all(v <= 2 for v in per.values())


def test_finished_candidates_land_in_the_pool_and_stop_generating():
    backend = FakeBackend(default=ENDS)
    result = run(backend, max_steps=4, beam_width=4, expand=4)
    assert set(result.pool) == {pkey(p) for p in PROBLEMS}
    assert all(c.done for cs in result.pool.values() for c in cs)
    assert all(c.stop == "eos" for cs in result.pool.values() for c in cs)


def test_a_beam_still_live_at_the_step_cap_is_retired_into_the_pool():
    backend = Finishing("length", default=NEVER_ENDS)
    result = run(backend, max_steps=2, beam_width=4, expand=4)
    assert all(c.stop == S.segment.STEPS for cs in result.pool.values() for c in cs)


def test_every_child_names_a_parent_kept_by_the_previous_step():
    backend = Finishing("length", default=NEVER_ENDS)
    result = run(backend, max_steps=3, beam_width=4, expand=4)
    for prev, step in zip(result.steps, result.steps[1:]):
        kept = {r["cid"] for r in prev["rows"] if r["kept"]}
        for r in step["rows"]:
            assert r["parent"] in kept


def test_steps_hold_plain_rows_not_candidates():
    backend = Finishing("length", default=NEVER_ENDS)
    result = run(backend, max_steps=2, beam_width=2, expand=2, min_distinct_parents=0)
    for step in result.steps:
        assert all(isinstance(r, dict) for r in step["rows"])
        assert all(r["score"] is not None for r in step["rows"] if not r["done"])


def test_a_retired_candidate_carries_one_score_per_step_it_survived():
    backend = Finishing("length", default=NEVER_ENDS)
    result = run(backend, max_steps=3, beam_width=4, expand=4)
    pool = [c for cs in result.pool.values() for c in cs]
    assert pool and all(len(c.scores) == 3 for c in pool)
    assert all(len(c.cut_chars) == len(c.scores) for c in pool)


def test_keep_pruned_collects_the_dropped_candidates():
    backend = Finishing("length", default=NEVER_ENDS)
    result = run(backend, max_steps=2, beam_width=2, expand=4, min_distinct_parents=0,
                 keep_pruned=True)
    assert sum(len(v) for v in result.pruned.values()) > 0


def test_keep_pruned_off_collects_nothing():
    backend = Finishing("length", default=NEVER_ENDS)
    result = run(backend, max_steps=2, beam_width=2, expand=4, min_distinct_parents=0)
    assert sum(len(v) for v in result.pruned.values()) == 0


def test_supplied_heads_and_caps_reach_their_problem_only():
    backend = FakeBackend(default=ENDS)
    conf = PRMSearchConfig(selector=SEL_RANDOM, max_steps=1)
    rc = PRMRolloutConfig(max_new_tokens=4096, temperature=0.8, think_temperature=1.0)
    k0 = pkey(PROBLEMS[0])
    result = S.search(backend, PROBLEMS, conf, rc, count, None, _Encoder(), "SYS",
                      lambda p: f"solve {p.problem_id}", random.Random(0),
                      heads={k0: "<CHAT0>"}, caps={k0: 77})
    for key, cands in result.pool.items():
        for c in cands:
            assert c.prompt == f"solve {key[1]}"
            assert (c.head == "<CHAT0>") == (key == k0)
            assert c.cap == (77 if key == k0 else None)


def test_sub_beams_stay_separate_to_the_last_step():
    backend = Finishing("length", default=NEVER_ENDS)
    result = run(backend, max_steps=6, beam_width=4, expand=4, beam_groups=4)
    for step in result.steps:
        groups = {}
        for r in step["rows"]:
            if r["kept"]:
                groups.setdefault(r["pkey"], []).append(r["group"])
        assert all(sorted(g) == [0, 1, 2, 3] for g in groups.values())
    assert all({c.group for c in cs} == {0, 1, 2, 3} for cs in result.pool.values())


def test_a_child_inherits_its_parents_group():
    backend = Finishing("length", default=NEVER_ENDS)
    result = run(backend, max_steps=4, beam_width=4, expand=4, beam_groups=2)
    group_of = {r["cid"]: r["group"] for step in result.steps for r in step["rows"]}
    for step in result.steps[1:]:
        for r in step["rows"]:
            assert r["group"] == group_of[r["parent"]]
