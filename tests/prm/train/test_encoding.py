"""``prm.rollout.encoding``: a prefix -> the ids a PRM scores it by (PLAN_v2 §5).

The one piece of §5's trainer contract this plan builds, and it exists so `rank_eval.py` and
whatever trainer comes later encode *identically*. Two properties carry that: the text is the
stored prompt plus the prefix verbatim -- never `SequenceEncoder`'s instruction/separator
layout, because at inference there is no candidate kernel to put after a separator -- and
over-length truncation drops the head, so the cut point being scored always survives.
"""

from __future__ import annotations

import pytest

from processkernel.prm.train import encoding
from processkernel.prm.rollout import prefixes
from processkernel.prm.rollout.rollout import Source


class CharTokenizer:
    """One id per character, so a test can say exactly where a truncation lands."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False, "the v2 encoder assembles its own sequence"
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


def pre(**over) -> prefixes.Prefix:
    fields = dict(
        prefix_id="ar__shard00__p37__s1__k020",
        source="cut",
        run_name="a_run",
        run_tag="ar",
        shard="shard_00",
        level=2,
        problem_id=37,
        sample_id=1,
        stem="level_2_problem_37_sample_1_kernel",
        cut_char=5,
        cut_index=20,
        cut_kind="code",
        n_cuts_total=80,
        rel_depth=0.25,
        list_key="ar:2:37:20",
        split="train",
        selection="random",
        selection_score=None,
        K=4,
        min_rollouts=2,
    )
    fields.update(over)
    return prefixes.Prefix(**fields)


def src(prompt="PROMPT", raw="0123456789", system_prompt="SYS") -> Source:
    return Source(system_prompt=system_prompt, prompt=prompt, raw=raw)


# --- the text ----------------------------------------------------------------------------


def test_scored_text_is_the_stored_prompt_then_the_prefix_verbatim():
    assert encoding.scored_text(pre(cut_char=4), src()) == "PROMPT0123"


def test_scored_text_appends_a_beams_branch_when_the_prefix_carries_one():
    assert encoding.scored_text(pre(cut_char=4, beam_text="BEAM"), src()) == "PROMPT0123BEAM"


def test_scored_text_leaves_out_the_system_prompt_the_generator_rendered_a_chat_header_from():
    # §5: the PRM scores what v1 froze, not what `render_chat` built for the sampler -- at
    # inference there is no chat header around the half-written generation being judged.
    assert "SYS" not in encoding.scored_text(pre(), src())


def test_scored_text_refuses_a_cut_char_its_v1_row_cannot_hold():
    with pytest.raises(ValueError, match="cut_char"):
        encoding.scored_text(pre(cut_char=999), src())


# --- the ids -----------------------------------------------------------------------------


def test_a_sequence_inside_max_length_is_encoded_whole():
    enc = encoding.PrefixEncoder(CharTokenizer(), max_length=64)
    ids = enc.encode(pre(cut_char=4), src())
    assert CharTokenizer().decode(ids) == "PROMPT0123"


def test_over_length_truncation_drops_the_head_of_the_prompt_and_keeps_every_prefix_token():
    enc = encoding.PrefixEncoder(CharTokenizer(), max_length=60)
    ids = enc.encode(pre(cut_char=50), src(prompt="P" * 100, raw="R" * 50))
    assert CharTokenizer().decode(ids) == "P" * 10 + "R" * 50


def test_a_prefix_longer_than_max_length_keeps_its_tail_because_the_cut_point_is_the_thing_scored():
    enc = encoding.PrefixEncoder(CharTokenizer(), max_length=40)
    ids = enc.encode(pre(cut_char=93), src(prompt="P" * 10, raw="R" * 90 + "CUT"))
    assert len(ids) == 40
    assert CharTokenizer().decode(ids) == "R" * 37 + "CUT"


def test_a_beams_branch_is_the_tail_that_survives_truncation():
    enc = encoding.PrefixEncoder(CharTokenizer(), max_length=8)
    ids = enc.encode(pre(cut_char=10, beam_text="BEAM"), src(prompt="P" * 40))
    assert CharTokenizer().decode(ids) == "6789BEAM"


def test_max_length_below_one_is_refused_rather_than_encoding_the_whole_sequence():
    # ids[-0:] is the whole list, so a zero budget would silently encode everything.
    with pytest.raises(ValueError, match="max_length"):
        encoding.PrefixEncoder(CharTokenizer(), max_length=0)


# --- the layering constraint (§5: "must not be folded back into SequenceEncoder") --------


def test_the_v2_encoder_does_not_reach_for_the_pointwise_cross_encoder():
    # The pointwise and pairwise paths depend on `SequenceEncoder`'s layout; v2 scoring a
    # half-written generation has no candidate kernel to place after its separator. Folding
    # the two together silently changes what every existing checkpoint was trained on.
    #
    # Read off the syntax tree rather than the file's text: the module docstring names both,
    # and it should -- what it must not do is import or call either.
    import ast

    with open(encoding.__file__) as f:
        tree = ast.parse(f.read())
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "processkernel.orm.encoding" not in imported
    named = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    assert "SequenceEncoder" not in named


def test_encoding_still_works_when_sequence_encoder_would_explode():
    import processkernel.orm.encoding as pointwise

    class Bomb:
        def __init__(self, *a, **k):
            raise AssertionError("the v2 encoder called SequenceEncoder")

    original, pointwise.SequenceEncoder = pointwise.SequenceEncoder, Bomb
    try:
        enc = encoding.PrefixEncoder(CharTokenizer(), max_length=64)
        assert CharTokenizer().decode(enc.encode(pre(cut_char=4), src())) == "PROMPT0123"
    finally:
        pointwise.SequenceEncoder = original
