"""Shared constants for the memory-traffic checks."""

from __future__ import annotations

#: Achievable HBM bandwidth, bytes/second. ~80% of an A100-SXM4-80GB's 2.039 TB/s peak
#: -- the runs in this repo were all evaluated on A100s.
ACHIEVABLE_BW = 1.6e12

#: Cost of one Triton kernel launch (CPU-side launch + GPU scheduling), seconds.
LAUNCH_OVERHEAD = 5e-6


def fmt_bytes(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


def fmt_time(seconds: float) -> str:
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds * 1e6:.1f} us"


def transfer_time(nbytes: int) -> float:
    return nbytes / ACHIEVABLE_BW
