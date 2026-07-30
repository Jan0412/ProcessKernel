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
#: see ``checker.core.parsing.ENTRY_CLASSES``.
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
    # Empty blocks are dropped: an empty string is never the model's answer, and one is
    # manufactured by the two-pass sampler whenever pass 2 returns nothing -- the injected
    # CODE_FENCE then trails the text with no content after it. `ast.parse("")` succeeds,
    # so an empty block used to rank as a valid candidate and win (KGEN-9).
    return [b for b in (textwrap.dedent(b).strip() for b in blocks) if b]


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
    4. and with no fence at all, the text from its first Python statement -- falling back
       to its largest parseable prefix when even that does not parse, but only when the
       salvage contains :data:`ENTRY_CLASS` (KGEN-9).

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

    def _loads(src: str) -> bool:
        """What the evaluator actually needs: ``exec(compile(...))``, not just a parse.

        ``compile`` runs the symbol-table and codegen passes on top of the parser, so it
        rejects source ``ast.parse`` accepts -- a duplicated kernel parameter being by far
        the most common in practice. Raises wider than ``SyntaxError`` (``ValueError`` on
        a null byte, ``IndentationError``, ``MemoryError`` on pathological nesting), hence
        the broad except.
        """
        try:
            compile(src, "<block>", "exec")
            return True
        except Exception:  # noqa: BLE001 - the family is wider than SyntaxError
            return False

    blocks = _fenced_blocks(text)
    if blocks:
        # Containing ModelNew stays the dominant criterion and loadability only breaks
        # ties within it. Ranking purely by loadability looks stricter and is worse:
        # measured over 10,510 real completions it moved 111 attempts off a ModelNew block
        # that merely failed to compile and onto a compilable *fragment* with no entry
        # class -- trading a one-line fix the repair round is now told about for a file
        # that scores zero with certainty.
        parseable = [b for b in blocks if _parses(b)]
        loadable = [b for b in parseable if _loads(b)]
        submissions = [b for b in parseable if ENTRY_CLASS in b]
        loadable_submissions = [b for b in loadable if ENTRY_CLASS in b]
        return (
            loadable_submissions or submissions or loadable or parseable or blocks[:1]
        )[-1]

    # Every fence marker here is noise: no block had any content, so none of them
    # delimited code. They are replaced by a newline rather than deleted -- the two-pass
    # seam's injected ```python lands glued to the model's last line ("return f(x)```python"),
    # and deleting it outright would splice that line onto the next one.
    stripped = textwrap.dedent(_FENCE_TOKEN.sub("\n", text.strip()))
    if not _parses(stripped):
        head = re.search(r"^[ \t]*(?:import\s|from\s|@|def\s|class\s)", stripped, re.MULTILINE)
        if head:
            candidate = textwrap.dedent(stripped[head.start():]).strip()
            if _parses(candidate):
                return candidate
            salvage = _largest_parseable_prefix(candidate)
            if ENTRY_CLASS in salvage:
                return salvage
    return stripped.strip()


def _largest_parseable_prefix(src: str, max_trims: int = 60) -> str:
    """The longest leading run of ``src`` that parses, found by trimming at each error.

    Only ever reached when the whole unfenced text fails to parse, i.e. where the caller
    would otherwise return something it already knows is not Python. A model that writes
    its kernel as unfenced prose-with-code leaves a complete ``ModelNew`` followed by
    commentary; ``SyntaxError.lineno`` points at the first line that is not code, so
    trimming there and retrying converges on the code. The caller keeps the result only
    when it contains :data:`ENTRY_CLASS`, so a salvage that finds no submission is
    discarded rather than shipped.

    **Deliberately not applied to fenced blocks.** Ranking a salvaged submission above a
    complete non-submission block looks like the KGEN-1 principle, and on the 10,510-trace
    corpus it "recovers" 5 more. Four of those five are worse than what they replace: a
    block truncated mid-``ModelNew`` salvages to a class with ``__init__`` but no
    ``forward`` (instantiable, not callable), and one to a 299-char stub with no imports at
    all. Eval scores those zero either way, but they read as submissions. A truncated block
    is a model failure; only the *unfenced* case is ours.
    """
    lines = src.splitlines()
    hi = len(lines)
    for _ in range(max_trims):
        if hi <= 0:  # guards the slice below: a negative hi would trim from the FRONT
            return ""
        candidate = "\n".join(lines[:hi])
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError as exc:
            # Jump to just before the offending line, and never fewer than one line, so
            # `hi` strictly decreases whatever the parser reports.
            hi = min(exc.lineno or hi, hi) - 1
        except ValueError:
            return ""  # e.g. a lone surrogate: UnicodeEncodeError, and no lineno to use
    return ""


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
