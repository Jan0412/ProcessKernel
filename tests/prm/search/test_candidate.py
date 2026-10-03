"""``Candidate``: what a live generation carries, and what a child inherits from its parent."""

from __future__ import annotations

from processkernel.generation.core.backend import FakeBackend
from processkernel.generation.core.model import Problem
from processkernel.generation.core.sampling import PLAN_PREFIX
from processkernel.config import PRMSearchConfig
from processkernel.prm.data.chunks import CODE, PROSE
from processkernel.prm.search.candidate import Candidate, pkey, root_of

PROBLEM = Problem(level=6, problem_id=12, name="12_Thing.py", ref_arch_src="import torch\n")
TEXT = "## Plan\nDo it.\n```python\nimport torch\nx = 1\n"


def cand(**over) -> Candidate:
    base = dict(problem=PROBLEM, head="<HEAD>", prompt="<PROMPT>", text=TEXT, mode=CODE)
    return Candidate(**{**base, **over})


def test_context_is_what_the_model_continues_from():
    assert cand().context() == "<HEAD>" + TEXT


def test_scored_prefix_uses_the_bare_prompt_not_the_chat_header():
    text = cand().scored_prefix(PRMSearchConfig())
    assert text.startswith("<PROMPT>")
    assert "<HEAD>" not in text


def test_scored_prefix_ends_at_a_line_end():
    assert cand().scored_prefix(PRMSearchConfig()).endswith("\n")


def test_score_at_cut_false_keeps_the_raw_tail():
    ragged = cand(text=TEXT + "y = foo(")
    assert ragged.scored_prefix(PRMSearchConfig(score_at_cut=False)).endswith("y = foo(")
    assert not ragged.scored_prefix(PRMSearchConfig()).endswith("y = foo(")


def test_a_child_inherits_the_text_and_records_its_parent():
    parent = cand(scores=[0.5])
    child = parent.child()
    assert child.text == parent.text
    assert child.parent == parent.cid
    assert child.cid != parent.cid


def test_a_childs_history_is_a_copy_not_the_parents_list():
    parent = cand(scores=[0.5])
    child = parent.child()
    child.scores.append(0.9)
    assert parent.scores == [0.5]


def test_two_children_of_one_parent_get_different_ids():
    parent = cand()
    assert parent.child().cid != parent.child().cid


def test_the_root_opens_with_the_plan_prefill_when_the_two_pass_sampler_is_on():
    root = root_of(PROBLEM, FakeBackend(), "SYS", "USER", think=True)
    assert root.text == PLAN_PREFIX
    assert root.mode == PROSE
    assert root.parent is None


def test_a_single_pass_root_is_empty_and_already_in_code_mode():
    root = root_of(PROBLEM, FakeBackend(), "SYS", "USER", think=False)
    assert root.text == ""
    assert root.mode == CODE


def test_pkey_separates_levels():
    assert pkey(PROBLEM) != pkey(Problem(level=1, problem_id=12, name="x", ref_arch_src=""))


def test_a_supplied_head_replaces_the_rendered_chat_but_not_the_prm_prompt():
    c = root_of(PROBLEM, FakeBackend(), "SYS", "<PROMPT>", think=False, head="<CHAT>", cap=100)
    assert (c.head, c.prompt, c.cap) == ("<CHAT>", "<PROMPT>", 100)


def test_a_child_inherits_the_cap():
    assert cand(cap=300).child().cap == 300
