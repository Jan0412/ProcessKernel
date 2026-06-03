"""Label logic for the reranker dataset.

A generated kernel is a positive example (label 1) iff it
  - compiled, AND
  - is numerically correct.

Everything else (failed to compile or incorrect output) is a negative example (label 0).
"""

from __future__ import annotations

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
