"""The part of a report's summary that means the same thing for every analyzer.

Deliberately mentions no check id. The version this was split out of hardcoded ``"F1.4"``
and ``("F2.1", "F2.3", "F2.4")``, which made "summarise a report" quietly a lint-only
operation -- a second analyzer would have inherited three keys that can never be non-zero
for it. Those keys now live in :mod:`checker.lint.summary`, and what is left here is true
of any set of findings whatsoever.
"""

from __future__ import annotations

from .model import Finding


def build_summary(findings: list[Finding]) -> dict:
    return {
        "n_fail": sum(1 for f in findings if f.severity == "fail"),
        "n_warn": sum(1 for f in findings if f.severity == "warn"),
        "n_info": sum(1 for f in findings if f.severity == "info"),
        "check_ids": sorted({f.check_id for f in findings}),
    }
