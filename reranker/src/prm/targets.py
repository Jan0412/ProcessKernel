"""Eval verdict + baseline runtime -> the value a prefix is trained against (PLAN §5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from reranker.src.data.labels import compute_label, speed_p

GRADED, BINARY = "graded", "binary"
MEAN, MIN = "mean", "min"


@dataclass(frozen=True)
class Target:
    value: float               # the training target, [0, 1]
    label: int                 # binary verdict, always kept for diagnostics
    speedup: float | None      # mean over mean
    speedup_min: float | None  # min over min


def check_scale(lo: float, hi: float, quant: float) -> None:
    if not 0 < lo < hi:
        raise ValueError(f"need 0 < speedup_lo < speedup_hi, got {lo=} {hi=}")
    # speed_p quantizes only when quant > 0, so a negative one silently turns the ladder
    # off; p is clamped to [0, 1], so a step wider than that collapses it to one value.
    if not 0 <= quant <= 1:
        raise ValueError(f"need 0 <= speed_quant <= 1 (0 = off), got {quant=}")


def check_knobs(mode: str, stat: str, lo: float, hi: float, quant: float) -> None:
    """Every grading knob at once, so config.validate and target_for cannot disagree."""
    if mode not in (GRADED, BINARY):
        raise ValueError(f"label_mode must be {GRADED!r} or {BINARY!r}, got {mode!r}")
    if stat not in (MEAN, MIN):
        raise ValueError(f"speedup_stat must be {MEAN!r} or {MIN!r}, got {stat!r}")
    if mode == GRADED:
        check_scale(lo, hi, quant)


def graded_target(speedup: float, *, lo: float, hi: float, quant: float) -> float:
    """Correct kernels only: ``lo`` -> 0.5, ``hi`` -> 1.0, on a ladder of ``quant``/2 steps."""
    check_scale(lo, hi, quant)
    return (1.0 + speed_p(speedup, lo, hi, quant)) / 2


def _usable(t: float | None) -> bool:
    """Present, finite and positive -- not missing, not the harness's -1.0 "never timed"."""
    return t is not None and math.isfinite(t) and t > 0


def speedups(
    *,
    correct: bool,
    ours: dict[str, float | None] | None,
    baseline: dict[str, float] | None,
) -> dict[str, float | None]:
    """Baseline over kernel runtime, mean/mean and min/min; ``None`` where a side is unusable.

    Both sides carry the same two keys so neither statistic can be taken from the other's
    source (PRM-4); corpus.py owns the mapping from KernelBench's field names. Keyword-only
    because a transposed call would then invert every speedup in silence.
    """
    out: dict[str, float | None] = {MEAN: None, MIN: None}
    if not correct:
        return out
    for stat in (MEAN, MIN):
        base, kernel = (baseline or {}).get(stat), (ours or {}).get(stat)
        if _usable(base) and _usable(kernel):
            # The quotient is checked too: two finite inputs still divide to 0.0 or inf,
            # which speed_p raises on and grades 1.00 respectively.
            ratio = float(base) / float(kernel)
            if _usable(ratio):
                out[stat] = ratio
    return out


def target_for(
    *,
    compiled: bool,
    correct: bool,
    ours: dict[str, float | None] | None,
    baseline: dict[str, float] | None,
    mode: str,
    stat: str,
    lo: float,
    hi: float,
    quant: float,
) -> Target | None:
    """Grade one evaluated sample; ``None`` when a correct kernel has no speedup to grade.

    Every knob is checked before any row is graded -- a bad one must not surface partway
    through a build. ``None`` is a drop the caller counts, never a substituted value.
    """
    check_knobs(mode, stat, lo, hi, quant)

    label = compute_label(compiled=compiled, correct=correct).label
    speed = speedups(correct=bool(label), ours=ours, baseline=baseline)

    if mode == BINARY:
        value = float(label)
    elif not label:
        value = 0.0
    elif speed[stat] is None:
        return None
    else:
        value = graded_target(speed[stat], lo=lo, hi=hi, quant=quant)
    return Target(value, label, speed[MEAN], speed[MIN])
