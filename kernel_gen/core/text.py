"""Pure string helpers: completion -> code, CLI spec -> ids, problem name -> id.

Duplicated from ``generate_kernels_samples.py`` on purpose -- see this package's
docstring. Keep them byte-equivalent in behavior until the arms are ported.
"""

from __future__ import annotations

import os
import re

_FENCED = re.compile(r"```python\s*(.*?)```", re.DOTALL)
_LEADING_ID = re.compile(r"(\d+)")


def extract_code_block(text: str) -> str:
    """The first ```python fenced block, or the whole text with fences stripped."""
    match = _FENCED.search(text)
    if match:
        return match.group(1).strip()
    return re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip())


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
