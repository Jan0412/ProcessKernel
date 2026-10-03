"""Label logic for the reranker dataset.

A generated kernel is a positive example (label 1) iff it
  - compiled, AND
  - is numerically correct.

Everything else (failed to compile or incorrect output) is a negative example (label 0).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
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


def load_baseline_times(path: str) -> dict[int, dict[int, dict[str, float]]]:
    """Baseline runtimes as ``{level: {problem_id: {"mean", "min"}}}``; ``{}`` if the file is gone.

    The KernelBench timing JSON keys levels as ``"level1"`` and problems by filename
    (``"1_Square_matrix_multiplication_.py"``), whose integer prefix is the problem id.
    ``mean`` is the ``fast_p`` convention, ``min`` the noise-robust one for launch-bound
    kernels. One parse for both the reranker and the PRM builds.
    """
    if not os.path.isfile(path):
        # Not fatal here, but the cost differs per caller: build_dataset writes a null
        # column, the PRM build drops every correct kernel it cannot grade.
        print(f"[WARN] baseline timing file not found: {path} — no speedups will be computed")
        return {}
    with open(path) as f:
        raw = json.load(f)
    out: dict[int, dict[int, dict[str, float]]] = {}
    for level_key, problems in raw.items():
        m = re.match(r"level(\d+)$", str(level_key))
        if not m or not isinstance(problems, dict):
            continue
        per_problem: dict[int, dict[str, float]] = {}
        for fname, stats in problems.items():
            pm = re.match(r"(\d+)_", str(fname))
            if pm and isinstance(stats, dict) and stats.get("mean") is not None:
                entry = {"mean": float(stats["mean"])}
                if stats.get("min") is not None:
                    entry["min"] = float(stats["min"])
                per_problem[int(pm.group(1))] = entry
        out[int(m.group(1))] = per_problem
    return out


def code_hash(src: str) -> str:
    """Stable hash of a kernel source, used to drop duplicate candidates.

    Shared by the listwise and pairwise builds so both dedup identically.
    """
    return hashlib.sha1(src.encode("utf-8", "ignore")).hexdigest()
