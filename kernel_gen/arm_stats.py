"""Compare one A/B arm's correctness against the control arm.

Paired at problem level only: a changed prompt prefix changes sampling, so slots do
not correspond one-to-one across arms.

`se` treats every slot as independent, which they are NOT -- slots cluster by problem
(measured ICC 0.434 on kb6 round 0, design effect 2.30 at 4 slots/problem). So `se` is
understated and `sigma` overstates significance by ~1.5x. Multiply `se` by sqrt(2.30)
before reading `sigma` as a z-score.
"""

from __future__ import annotations

import math


def compare(control: dict, arm: dict, exclude: set) -> dict:
    shared = (set(control) & set(arm)) - set(exclude)
    c = [ok for pid in shared for ok in control[pid]]
    a = [ok for pid in shared for ok in arm[pid]]
    n_c, n_a = len(c), len(a)
    p_c = sum(c) / n_c if n_c else 0.0
    p_a = sum(a) / n_a if n_a else 0.0
    var = (p_c * (1 - p_c) / n_c if n_c else 0.0) + (
        p_a * (1 - p_a) / n_a if n_a else 0.0
    )
    se = math.sqrt(var)
    diff = p_a - p_c
    return {
        "n_control": n_c,
        "n_arm": n_a,
        "p_control": p_c,
        "p_arm": p_a,
        "diff": diff,
        "se": se,
        "sigma": diff / se if se else 0.0,
    }
