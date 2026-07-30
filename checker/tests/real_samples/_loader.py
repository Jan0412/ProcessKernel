"""Loader for the audited real-sample kernels in ``data/``."""

from __future__ import annotations

from pathlib import Path

from checker import analyze_file
from checker.model import Finding

DATA = Path(__file__).parent / "data"


def findings(problem: int, sample: int, check_id: str, level: int = 5) -> list[Finding]:
    """Run one check against the copied sample and return its findings."""
    path = DATA / f"level_{level}_problem_{problem}_sample_{sample}_kernel.py"
    report = analyze_file(str(path), only={check_id})
    assert report.parse_status == "ok", report.summary
    return [f for f in report.findings if f.check_id == check_id]


def full_report(problem: int, sample: int, level: int = 5):
    path = DATA / f"level_{level}_problem_{problem}_sample_{sample}_kernel.py"
    return analyze_file(str(path))
