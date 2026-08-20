"""The two primitives the search adds to existing modules: live cut points, and scored text."""

from __future__ import annotations

from reranker.src.prm.chunks import CODE, cut_points, live_cuts
from reranker.src.prm.rollout.encoding import PrefixEncoder, scored_text_of

WHOLE = "## Plan\nDo it.\n```python\nimport torch\nx = 1\ny = 2\n```\n"
OPEN_BRACKET = "## Plan\nDo it.\n```python\nimport torch\nx = 1\ny = foo(\n"
OPEN_STRING = '## Plan\nDo it.\n```python\nimport torch\ns = """abc\n'


def test_on_text_that_tokenizes_it_is_exactly_cut_points():
    assert live_cuts(WHOLE) == cut_points(WHOLE)


def test_it_recovers_the_cuts_before_an_unterminated_bracket():
    assert cut_points(OPEN_BRACKET) is None
    cuts = live_cuts(OPEN_BRACKET)
    assert [c.char for c in cuts] == [c.char for c in cut_points(OPEN_BRACKET[: OPEN_BRACKET.rfind("y = foo(")])]
    assert cuts[-1].char <= OPEN_BRACKET.index("y = foo(")


def test_it_recovers_the_cuts_before_an_unterminated_string():
    assert cut_points(OPEN_STRING) is None
    assert [c.char for c in live_cuts(OPEN_STRING)]


def test_it_returns_an_empty_list_rather_than_none_when_there_is_nothing_to_cut():
    assert live_cuts("") == []
    assert live_cuts("no newline anywhere") == []


def test_the_kept_set_of_a_prefix_is_a_prefix_of_the_kept_set_of_the_whole():
    """I3 over live text -- what makes a cut offset stable as the generation grows."""
    grown = WHOLE + "```python\nz = 3\n```\n"
    short = [c.char for c in live_cuts(WHOLE, code_steps=2)]
    long = [c.char for c in live_cuts(grown, code_steps=2)]
    assert long[: len(short)] == short


def test_coarse_chunking_keeps_every_nth_cut():
    fine = live_cuts(WHOLE)
    coarse = live_cuts(WHOLE, code_steps=2, prose_lines=2)
    assert len(coarse) < len(fine)
    assert {c.char for c in coarse} <= {c.char for c in fine}


def test_scored_text_of_is_the_layout_scored_text_builds():
    assert scored_text_of("PROMPT", "GENERATED") == "PROMPTGENERATED"


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def test_encode_text_truncates_from_the_head_so_the_cut_point_survives():
    enc = PrefixEncoder(_Tok(), max_length=3)
    assert enc.encode_text("abcdef") == [ord("d"), ord("e"), ord("f")]
