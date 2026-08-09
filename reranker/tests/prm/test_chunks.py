"""``prm.chunks``: the four invariants, and the cutting rule itself."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import random

import pytest

from reranker.src.prm.chunks import CODE, PROSE, Cut, cut_points

# The 36 real completions that pin kernel_gen's extractor, reused as realistic PRM input.
# Read by path, not shared with kernel_gen's four readers of the same file: a fixture
# cannot feed ``parametrize``, and kernel_gen/tests is deliberately not an importable
# package (its conftest must not reach the global path -- see pyproject's testpaths note).
_CORPUS = (
    pathlib.Path(__file__).parents[3]
    / "kernel_gen" / "tests" / "fixtures" / "completions" / "corpus.jsonl"
)
GOLDEN = [json.loads(x) for x in _CORPUS.read_text().splitlines() if x.strip()]
GOLDEN_RAW = [c["raw"] for c in GOLDEN]
GOLDEN_IDS = [c["id"] for c in GOLDEN]

# One of the 36 is a real tokenize failure -- a KGEN-11 completion whose fence opener
# precedes PROSE, so the "code" span is English and its apostrophe reads as an
# unterminated string. Whole completions drop at 0.21% (28/13,353 real attempts); an
# arbitrary mid-line prefix returns None far more often, 21.6%, which is why the I3 tests
# below skip those. Cut-aligned prefixes -- the only kind the builder emits -- never do.
CUTTABLE = [(r, i) for r, i in zip(GOLDEN_RAW, GOLDEN_IDS) if cut_points(r) is not None]
CUTTABLE_RAW = [r for r, _ in CUTTABLE]
CUTTABLE_IDS = [i for _, i in CUTTABLE]

TRUNCATIONS = 50
SEED = 20260808


def chars(cuts):
    return [c.char for c in cuts]


# -- I3: prefix-closure ----------------------------------------------------------------


@pytest.mark.parametrize("raw", CUTTABLE_RAW, ids=CUTTABLE_IDS)
def test_i3_cuts_are_prefix_closed_on_real_completions(raw):
    full = cut_points(raw)
    rng = random.Random(SEED)
    checked = 0
    for _ in range(TRUNCATIONS):
        c = rng.randrange(0, len(raw) + 1)
        got = cut_points(raw[:c])
        if got is None:
            continue  # truncated mid-bracket: no correct partial answer exists
        assert got == [x for x in full if x.char <= c], f"truncation at {c}"
        checked += 1
    assert checked, "every truncation returned None -- the property was never exercised"


@pytest.mark.parametrize("raw", CUTTABLE_RAW, ids=CUTTABLE_IDS)
def test_i3_holds_under_grouping_too(raw):
    # Grouping renumbers cuts, so it is the step most likely to break prefix-closure --
    # counting the ordinal backward from the end would fail exactly here.
    full = cut_points(raw, prose_lines=4, code_steps=4)
    rng = random.Random(SEED + 1)
    checked = 0
    for _ in range(TRUNCATIONS):
        c = rng.randrange(0, len(raw) + 1)
        got = cut_points(raw[:c], prose_lines=4, code_steps=4)
        if got is None:
            continue
        assert got == [x for x in full if x.char <= c], f"truncation at {c}"
        checked += 1
    assert checked, "every truncation returned None -- the property was never exercised"


def test_i3_cut_index_is_positional_so_a_prefix_keeps_its_numbering():
    raw = "one\ntwo\nthree\nfour\n"
    full = cut_points(raw)
    head = cut_points(raw[:8])
    assert [c.index for c in full] == [0, 1, 2, 3]
    assert head == full[:2]  # same chars AND same indices


def test_truncating_mid_bracket_returns_none_rather_than_a_partial_answer():
    raw = "```python\nx = f(\n    1,\n"
    assert cut_points(raw) is None


def test_exactly_one_golden_completion_fails_to_tokenize():
    # Pins the drop, so a change that starts swallowing unparseable spans (or starts
    # rejecting parseable ones) is visible here rather than as a corpus-size surprise.
    assert len(CUTTABLE) == len(GOLDEN) - 1
    assert set(GOLDEN_IDS) - set(CUTTABLE_IDS) == {
        "kgen11_level_1_problem_33_sample_4_kernel_r0"
    }


def test_a_nul_byte_is_a_drop_not_a_crash():
    # The indented body is the point: with a pending DEDENT 3.12's C tokenizer raises
    # SystemError, which is not a SyntaxError and escaped the guard. Flat code raises
    # TokenError and never reproduced it. 0 NULs in 285,149 attempts, but a JSON
    # escape carries one through json.loads untouched, so the class is reachable.
    assert cut_points("```python\ndef f():\n    return 1\n\x00\n```\n") is None
    assert cut_points("```python\nx = 1\n\x00\n```\n") is None


def test_a_prefix_can_cut_where_the_whole_text_cannot():
    # The other half of the same fact: `None` is a verdict on the text seen so far, so a
    # completion whose tail is unparseable still cuts cleanly at every earlier prefix.
    raw = "```python\nx = 1\ns = 'unterminated\n"
    assert cut_points(raw) is None
    assert chars(cut_points(raw[:16])) == [10, 16]  # ```python , x = 1


@pytest.mark.parametrize("raw", CUTTABLE_RAW, ids=CUTTABLE_IDS)
def test_a_prefix_taken_at_a_cut_point_always_cuts(raw):
    # The builder emits `raw[:cut.char]`, never an arbitrary offset, and a cut always sits
    # after a completed logical line -- so `None` cannot reach a training row.
    cuts = cut_points(raw)
    assert cuts
    for cut in cuts:
        assert cut_points(raw[: cut.char]) is not None, f"cut at {cut.char}"


def test_a_prefix_that_ends_mid_statement_yields_no_cut_for_that_line():
    # The synthetic-NEWLINE trap: tokenize terminates an unfinished last line for us, one
    # char past the text. Honouring it would invent a cut the full text does not have.
    full = cut_points("```python\nx = 1\ny = 2\n```\n")
    head = cut_points("```python\nx = 1\ny")
    assert chars(head) == [10, 16]  # the ```python line and `x = 1`, not the bare `y`
    assert chars(head) == [c for c in chars(full) if c <= 17]


# -- I2: determinism -------------------------------------------------------------------


@pytest.mark.parametrize("raw", GOLDEN_RAW[:6], ids=GOLDEN_IDS[:6])
def test_i2_repeated_calls_agree(raw):
    assert cut_points(raw) == cut_points(raw) == cut_points(raw)


def test_i2_holds_across_configs_independently():
    raw = "## Plan\nfirst\n```python\nx = 1\ny = 2\n```\n"
    assert cut_points(raw, code_steps=2) == cut_points(raw, code_steps=2)
    assert cut_points(raw, code_steps=1) != cut_points(raw, code_steps=2)


# -- I1 / I4: one config, no cross-text state ------------------------------------------


def test_i1_i4_two_texts_of_very_different_length_do_not_influence_each_other():
    short = "```python\nx = 1\n```\n"
    long = "```python\n" + "".join(f"x{i} = {i}\n" for i in range(200)) + "```\n"

    alone_short, alone_long = cut_points(short), cut_points(long)
    interleaved = [cut_points(short), cut_points(long), cut_points(short)]

    assert interleaved == [alone_short, alone_long, alone_short]
    # I4: the counts are simply different, and nothing pads or equalizes them.
    assert len(alone_long) > 10 * len(alone_short)


def test_i4_cut_index_is_a_depth_not_a_fraction():
    # Depth 2 of a short completion and depth 2 of a long one are different states; the
    # index says nothing about how far through the text they are.
    short = cut_points("a\nb\nc\n")
    long = cut_points("a\nb\nc\n" + "d\n" * 50)
    assert short[2].index == long[2].index == 2
    assert short[2].char == long[2].char  # same depth, same char -- no rescaling


# -- the code rule: one cut per logical line -------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "def f(\n    a,\n    b,\n):\n    pass\n",
        "d = {\n    'a': 1,\n    'b': 2,\n}\n",
        "func(\n    x,\n    y,\n)\n",
        "total = (1 +\n         2 +\n         3)\n",
        "s = '''\nmulti\nline\n'''\n",
        "x = 1 + \\\n    2\n",
    ],
    ids=["def", "dict", "call", "parens", "string", "continuation"],
)
def test_one_connected_coding_step_is_one_cut(body):
    cuts = cut_points(f"```python\n{body}```\n")
    code = [c for c in cuts if c.kind == CODE]
    expected = 2 if body.startswith("def") else 1  # `def` line + its indented body
    assert len(code) == expected, chars(code)
    assert code[-1].char == len("```python\n") + len(body)


def test_blank_and_comment_only_lines_produce_no_code_cut():
    raw = "```python\nx = 1\n\n# a comment\n\ny = 2\n```\n"
    code = [c for c in cut_points(raw) if c.kind == CODE]
    assert [raw[:c.char].splitlines()[-1] for c in code] == ["x = 1", "y = 2"]


def test_a_code_cut_lands_after_the_newline_so_the_prefix_is_a_whole_line():
    raw = "```python\nx = 1\ny = 2\n```\n"
    code = [c for c in cut_points(raw) if c.kind == CODE]
    assert [raw[:c.char] for c in code][0].endswith("x = 1\n")


def test_an_unterminated_fence_still_yields_cuts():
    # The max_tokens tail, and 99% of the gpt-oss run: no closing fence ever arrives.
    raw = "## Plan\nDo it.\n```python\nimport torch\nx = 1\n"
    cuts = cut_points(raw)
    assert [c.kind for c in cuts] == [PROSE, PROSE, PROSE, CODE, CODE]
    assert raw[: cuts[-1].char].endswith("x = 1\n")


def test_code_outside_any_fence_is_prose_not_code():
    # PRM-1, known and not fixed: an unfenced kernel is cut by line, so a cut can land
    # mid-statement. 0 of 285,149 real attempts lack a fence; gating the gaps instead
    # drops 22.3% of them. This pins today's behaviour, not a desirable one.
    cuts = cut_points("import torch\nx = f(\n")
    assert all(c.kind == PROSE for c in cuts)
    assert cut_points("import torch\nx = f(\n") is not None  # no tokenize gate on prose


# -- the prose rule --------------------------------------------------------------------


def test_prose_cuts_skip_blank_lines():
    raw = "first\n\n\nsecond\n"
    assert chars(cut_points(raw)) == [6, 15]


def test_prose_cuts_need_a_real_newline():
    assert cut_points("no newline here") == []


def test_the_fence_marker_lines_are_prose():
    raw = "```python\nx = 1\n```\n"
    kinds = [c.kind for c in cut_points(raw)]
    assert kinds == [PROSE, CODE, PROSE]  # ```python , x = 1 , ```


# -- grouping --------------------------------------------------------------------------


def test_grouping_keeps_every_nth_same_kind_cut():
    raw = "```python\n" + "".join(f"x{i} = {i}\n" for i in range(12)) + "```\n"
    all_code = [c for c in cut_points(raw) if c.kind == CODE]
    every4 = [c for c in cut_points(raw, code_steps=4) if c.kind == CODE]
    assert chars(every4) == chars(all_code)[3::4]


def test_grouping_counts_each_kind_separately():
    raw = "p1\np2\np3\np4\n```python\nx = 1\ny = 2\n"
    cuts = cut_points(raw, prose_lines=4, code_steps=2)
    # 5 prose lines (4 + the ```python marker) -> the 4th kept; 2 code -> the 2nd kept.
    assert [(c.kind, c.index) for c in cuts] == [(PROSE, 0), (CODE, 1)]


def test_grouping_renumbers_index_contiguously_from_zero():
    raw = "```python\n" + "".join(f"x{i} = {i}\n" for i in range(12)) + "```\n"
    cuts = cut_points(raw, prose_lines=4, code_steps=4)
    assert [c.index for c in cuts] == list(range(len(cuts)))


def test_grouping_at_one_keeps_everything():
    raw = "## Plan\n```python\nx = 1\ny = 2\n```\n"
    assert cut_points(raw, prose_lines=1, code_steps=1) == cut_points(raw)


@pytest.mark.parametrize("bad", [{"prose_lines": 0}, {"code_steps": 0}, {"code_steps": -1}])
def test_a_chunk_size_below_one_is_rejected(bad):
    with pytest.raises(ValueError):
        cut_points("x\n", **bad)


# -- the returned shape ----------------------------------------------------------------


def test_cuts_are_ascending_and_inside_the_text():
    for raw in CUTTABLE_RAW:
        cuts = cut_points(raw)
        assert chars(cuts) == sorted(chars(cuts))
        assert len(set(chars(cuts))) == len(cuts)
        assert all(0 < c.char <= len(raw) for c in cuts)
        assert all(c.kind in (PROSE, CODE) for c in cuts)


def test_cut_is_frozen_so_a_row_cannot_be_edited_after_the_fact():
    cut = Cut(char=1, kind=PROSE, index=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cut.char = 5  # type: ignore[misc]


def test_the_empty_text_has_no_cuts():
    assert cut_points("") == []


def test_a_cut_slices_to_a_real_boundary_on_every_golden_completion():
    # The §12 hand-check, automated: no prefix may end mid-line. Unconditional -- a cut
    # at len(raw) is not exempt, it just means the text ends on a newline.
    for raw in CUTTABLE_RAW:
        for cut in cut_points(raw):
            assert raw[cut.char - 1] == "\n"


# -- carriage returns: tokenize reports a two-wide terminator past the end of the text --


def test_a_bare_carriage_return_at_the_end_of_a_prefix_invents_no_cut():
    full = "```python\nx = 1\r\ny = 2\r\n"
    assert cut_points(full) is not None
    # [:16] ends between the \r and the \n; tokenize calls that a finished logical line
    # ending at column 7 of a 6-char line, which would place a cut at char 17 of a
    # 16-char string.
    got = cut_points(full[:16])
    assert chars(got) == [10]
    assert chars(got) == [c.char for c in cut_points(full) if c.char <= 16]


@pytest.mark.parametrize("c", range(11, 25))
def test_i3_holds_at_every_truncation_of_a_crlf_completion(c):
    full = "```python\nx = 1\r\ny = 2\r\n"
    got = cut_points(full[:c])
    if got is not None:
        assert got == [x for x in cut_points(full) if x.char <= c]


def test_no_cut_ever_lands_past_the_end_of_the_text():
    for raw in ["```python\nx = 1\r", "```python\nx = 1\r\n", "```python\nx", "x\r"]:
        for cut in cut_points(raw) or []:
            assert 0 < cut.char <= len(raw), (raw, cut)
