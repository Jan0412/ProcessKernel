"""Cut points of one completion: where a half-written attempt can be scored (PLAN §4)."""

from __future__ import annotations

import io
import tokenize
from dataclasses import dataclass

from kernel_gen.core.text import fence_spans

PROSE, CODE = "prose", "code"


@dataclass(frozen=True)
class Cut:
    char: int   # the prefix is raw[:char]
    kind: str   # PROSE | CODE
    index: int  # 0-based depth in THIS completion; never compared across them (I4)


def cut_points(raw: str, *, prose_lines: int = 1, code_steps: int = 1) -> list[Cut] | None:
    """Cut offsets into ``raw``, ascending; ``None`` when its code does not tokenize.

    Code cuts at logical-line ends, prose at non-blank newlines. ``None`` rather than a
    partial answer keeps I3 exact: truncating mid-bracket has no correct prefix of cuts.
    """
    if prose_lines < 1 or code_steps < 1:
        raise ValueError(f"chunk sizes must be >= 1, got {prose_lines=} {code_steps=}")

    # Every span is tokenized as Python whatever its info tag. Over 285,149 real attempts:
    # 292,253 `python`, 52 `text`, 2 `analysis`, 2 `c`, 1 `diff`; being tag-blind costs 5
    # completions, so reading the tag is not worth the branch.
    spans = fence_spans(raw)
    regions = [(a, b, CODE) for a, b in spans]
    # Outside a fence is prose even when it is code -- known, measured, PRM-1.
    regions += [(a, b, PROSE) for a, b in _outside(raw, spans)]
    regions.sort()

    found: list[tuple[int, str]] = []
    for a, b, kind in regions:
        cuts = _code_cuts(raw, a, b) if kind == CODE else _prose_cuts(raw, a, b)
        if cuts is None:
            return None
        found.extend(cuts)

    # Count FORWARD, never back from the end: a cut's ordinal must not move when the
    # text grows, or a prefix stops keeping a prefix of the kept set (I3).
    every = {PROSE: prose_lines, CODE: code_steps}
    seen = {PROSE: 0, CODE: 0}
    kept: list[tuple[int, str]] = []
    for char, kind in found:
        seen[kind] += 1
        if seen[kind] % every[kind] == 0:
            kept.append((char, kind))
    return [Cut(char, kind, i) for i, (char, kind) in enumerate(kept)]


def _outside(raw: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The gaps between fenced spans -- prose, including the fence marker lines."""
    gaps, pos = [], 0
    for a, b in spans:
        gaps.append((pos, a))
        pos = b
    gaps.append((pos, len(raw)))
    return gaps


def _code_cuts(raw: str, a: int, b: int) -> list[tuple[int, str]] | None:
    """One cut per logical line of ``raw[a:b]``; ``None`` if it does not tokenize."""
    text = raw[a:b]
    line_start = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_start.append(i + 1)

    out: list[tuple[int, str]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type != tokenize.NEWLINE:
                continue  # NL ends a physical line only, so a multi-line call is 1 cut
            row, col = tok.end
            pos = a + line_start[row - 1] + col
            # tokenize invents a terminator for an unfinished last line and reports it
            # ending past the text (empty, or two-wide on a bare '\r'), which would be a
            # cut the full text lacks. Demand a real newline inside the span.
            if pos <= b and raw[pos - 1] == "\n":
                out.append((pos, CODE))
    # SystemError: 3.12's C tokenizer meeting a NUL with a pending DEDENT fails out of the
    # generator, not as a SyntaxError -- an indented body is enough, flat code is not.
    except (SyntaxError, tokenize.TokenError, ValueError, SystemError):
        return None
    return out


def _prose_cuts(raw: str, a: int, b: int) -> list[tuple[int, str]]:
    """One cut per non-blank line of ``raw[a:b]``. Never fails."""
    out: list[tuple[int, str]] = []
    line = a
    while True:
        nl = raw.find("\n", line, b)
        if nl < 0:
            return out
        if raw[line:nl].strip():
            out.append((nl + 1, PROSE))
        line = nl + 1
