"""Label logic for the reranker dataset.

A generated kernel is a positive example (label 1) iff it
  - compiled, AND
  - is numerically correct.

Everything else (failed to compile or incorrect output) is a negative example (label 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LabelResult:
    label: int
    reason: str  # human-readable explanation, useful for dataset stats


def compute_label(compiled: bool, correct: bool) -> LabelResult:
    """Apply the positive/negative rule to one evaluated kernel sample."""
    if not compiled:
        return LabelResult(0, "not_compiled")
    if not correct:
        return LabelResult(0, "incorrect")
    return LabelResult(1, "compiled_and_correct")


def speed_p(speedup: float, lo: float, hi: float, quant: float = 0.0) -> float:
    """Map an absolute speedup over the baseline to p in [0, 1] on a log2 scale.

    lo -> 0, hi -> 1, clamped.  e.g. lo=0.25x, hi=4x => 1x maps to p=0.5.

    ``quant`` (0 = off) snaps p to a grid of that size (``round(p/quant)*quant``):
    a deadband so trivial, sub-quant speedup differences collapse to the *same*
    grade and produce no spurious ranking pair. It does not drop any kernel; it
    only coarsens the grade, so it stays compatible with training on raw data.

    Shared by the listwise (graded relevance) and pairwise (speed pair_mode)
    pipelines so both grade ``fast_p`` identically (quant defaults off for pairwise).
    """
    lg, llo, lhi = math.log2(speedup), math.log2(lo), math.log2(hi)
    p = min(1.0, max(0.0, (lg - llo) / (lhi - llo)))
    if quant > 0:
        p = min(1.0, max(0.0, round(p / quant) * quant))
    return p
