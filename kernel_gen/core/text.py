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

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL)


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

    A newline is required after the language tag so an *indented* fence (nested under
    prose or a list item) does not have the first code line's indentation eaten while the
    rest keep theirs -- ``textwrap.dedent`` then restores a uniform block. When no fence
    is present, leading prose ("## Plan", a numbered list) is dropped by resuming at the
    first Python statement, but only when that turns an unparseable blob into a parseable
    one -- so a bare file that merely opens with a comment is returned untouched.
    """

    def _parses(src: str) -> bool:
        try:
            ast.parse(src)
            return True
        except (SyntaxError, ValueError):
            return False

    blocks = [textwrap.dedent(m.group(1)).strip() for m in _FENCE.finditer(text)]
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
