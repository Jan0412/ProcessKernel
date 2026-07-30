"""Kernel file naming: ``level_{L}_problem_{P}_sample_{S}_kernel.py``.

Same convention as ``reranker/src/data/build_dataset.py::_staged_kernel_path``, duplicated
here so ``checker`` stays dependency-free. It sits in ``core`` rather than next to either
analyzer because the generation arms and the run-dir readers address kernels by this name
without caring which analysis is about to run over them.
"""

from __future__ import annotations

import re

_KERNEL_RE = re.compile(
    r"^level_(?P<level>\d+)_problem_(?P<problem>\d+)_sample_(?P<sample>\d+)_kernel\.py$"
)


def staged_kernel_filename(level: int, problem_id: int, sample_id: int) -> str:
    return f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"


def parse_kernel_filename(name: str) -> tuple[int, int, int] | None:
    """``(level, problem_id, sample_id)`` or ``None`` if *name* is not a kernel file."""
    m = _KERNEL_RE.match(name)
    if m is None:
        return None
    return int(m["level"]), int(m["problem"]), int(m["sample"])
