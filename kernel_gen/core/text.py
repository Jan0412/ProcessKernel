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


def extract_code_block(text: str) -> str:
    """The first ```python / ```py fenced block, dedented; else the text from its first
    Python statement, with leading prose and stray fences removed.

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

    match = re.search(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", text, re.DOTALL)
    if match:
        return textwrap.dedent(match.group(1)).strip()

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
