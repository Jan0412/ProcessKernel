"""Pure string helpers: completion -> code, CLI spec -> ids, problem name -> id.

Duplicated from ``generate_kernels_samples.py`` on purpose -- see this package's
docstring. Keep them byte-equivalent in behavior until the arms are ported.
"""

from __future__ import annotations

import ast
import os
import re
import textwrap

_LEADING_ID = re.compile(r"(\d+)")


#: The class KernelBench instantiates. A fenced block without it is not a submission --
#: see ``triton_lint.parsing.ENTRY_CLASSES``.
ENTRY_CLASS = "class ModelNew"

#: A fenced-code delimiter: ``` then an optional info string (the language) then the end
#: of the line. Three properties earn their keep. (1) The info string is the
#: discriminator this model's output reliably provides -- an *opener* carries one
#: (```python), a *closer* is bare (```). (2) Matching through end-of-line, not merely
#: the backticks, is what keeps an inline ```tl.minimum(x) written in prose from being
#: read as a fence. (3) Matching anywhere on the line, not only at column 0, is what
#: keeps the two-pass seam's injected ```python visible -- it lands mid-line ~13% of the
#: time (5,844 of 45,853 in the level 1+2 traces).
_FENCE_TOKEN = re.compile(r"```([A-Za-z0-9_+#-]*)[ \t]*(?:\r?\n|$)")


def _fenced_blocks(text: str) -> list[str]:
    """The completion's fenced code blocks, paired open->close in document order.

    Replaces a regex (```` ```...``` ````, non-greedy) that could not tell an opening
    fence from a closing one -- both read as ```` ``` ````. A stray closing fence (the
    model ending an in-reasoning code example, or writing ```` ```</think> ````) was
    mis-read as an opener, so the non-greedy match ran on to the REAL block's ```python
    opener, captured the prose between them as the "block", and swallowed the real opener
    -- so the model's complete ``ModelNew`` was never seen as a block at all (KGEN-3).

    Here a block is opened only by an info-tagged fence; a bare ```` ``` ```` seen while
    *outside* a block is a stray closer and is skipped, which is the whole fix. Back-to-
    back openers (a ```python with no bare close before the next ```python) close the
    first and open the next. A block still open at end-of-text is the ``max_tokens`` tail
    (KGEN-2), returned as a candidate so a truncated final answer is not invisible.
    """
    blocks: list[str] = []
    start: int | None = None  # char index where the open block's content begins, or None
    for m in _FENCE_TOKEN.finditer(text):
        opener = bool(m.group(1))
        if start is not None:  # inside a block: any fence ends it
            blocks.append(text[start : m.start()])
            start = m.end() if opener else None  # an opener immediately reopens
        elif opener:  # outside a block: only an info-tagged fence starts one
            start = m.end()
        # a bare ``` outside a block is a stray closer -- ignore it
    if start is not None:  # unterminated final block: the model hit max_tokens mid-answer
        blocks.append(text[start:])
    return [textwrap.dedent(b).strip() for b in blocks]


def extract_code_block(text: str) -> str:
    """The model's **final** complete kernel file, dedented; else its best effort.

    Not the first fenced block, which is what this used to return. Reasoning models
    revise in the open: they write a kernel, then "Wait, ``tl.dot`` expects ``a`` to be
    ``(BLOCK_K, 1)``…", then write it again, several times, and only the last block is
    the answer. Measured on the first traced run (Qwen3.6-27B, KernelBench level 1),
    **12 of 14 completions had a later, better block than the one taken**, and four of
    those took a bare fragment -- a loop body with no imports and no class -- because an
    interrupted illustrative snippet happened to come first. Every one of the 14 did
    contain a complete block. So the rule is:

    1. the LAST fenced block that parses *and* defines :data:`ENTRY_CLASS`;
    2. failing that, the last that parses at all;
    3. failing that, the first block -- the historical behaviour, kept so a completion
       where nothing parses degrades exactly as it always did rather than newly
       returning something different;
    4. and with no fence at all, the text from its first Python statement.

    Block boundaries come from :func:`_fenced_blocks`, which pairs fences by an open/close
    walk rather than a regex, so three failure modes are handled at the source: a
    superseded first draft (ranked below the final block, KGEN-1); a final block cut off
    by ``max_tokens`` with no closing fence (recovered as the open tail, KGEN-2); and a
    stray closing fence that used to swallow the real block (ignored when it appears
    outside a block, KGEN-3). Each candidate then flows through the same ranking, so a
    truncated or fragmentary block still loses to a complete ``ModelNew``.

    When no fence is present, leading prose ("## Plan", a numbered list) is dropped by
    resuming at the first Python statement, but only when that turns an unparseable blob
    into a parseable one -- so a bare file that merely opens with a comment is returned
    untouched.
    """

    def _parses(src: str) -> bool:
        try:
            ast.parse(src)
            return True
        except (SyntaxError, ValueError):
            return False

    blocks = _fenced_blocks(text)
    if blocks:
        parseable = [b for b in blocks if _parses(b)]
        submissions = [b for b in parseable if ENTRY_CLASS in b]
        return (submissions or parseable or blocks[:1])[-1]

    stripped = textwrap.dedent(re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip()))
    if not _parses(stripped):
        head = re.search(r"^[ \t]*(?:import\s|from\s|@|def\s|class\s)", stripped, re.MULTILINE)
        if head:
            candidate = textwrap.dedent(stripped[head.start():]).strip()
            if _parses(candidate):
                return candidate
    return stripped.strip()


def parse_int_spec(spec: str) -> list[int]:
    """Parse ``'1-49'``, ``'1,5,10'`` or ``'23'`` into a list of ints."""
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def problem_id_from_name(name: str, fallback: int) -> int:
    """The real KernelBench problem id from a problem name.

    Names look like ``19_ReLU.py``; the leading integer is the id *within the level*,
    which is not the dataset array index (the HF split is ordered lexicographically,
    so index 1 is problem 10). Fall back to the index when there is no prefix.
    """
    match = _LEADING_ID.match(os.path.basename(name))
    return int(match.group(1)) if match else fallback
