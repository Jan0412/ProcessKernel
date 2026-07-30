"""Check registry.

Every check lives in its own file under ``family1/`` or ``family2/`` and
registers itself with :func:`register`.  A check is a function
``(ModuleModel) -> list[Finding]``; importing this package populates ``CHECKS``.

Adding a family is a new subpackage plus one import at the bottom of this file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ...core.model import Finding, ModuleModel, Severity

CheckFn = Callable[[ModuleModel], list[Finding]]


@dataclass
class Check:
    check_id: str
    name: str
    default_severity: Severity
    fn: CheckFn


CHECKS: list[Check] = []


def register(check_id: str, name: str, default_severity: Severity = "warn"):
    def decorator(fn: CheckFn) -> CheckFn:
        CHECKS.append(Check(check_id, name, default_severity, fn))
        return fn

    return decorator


def run_checks(model: ModuleModel, only: set[str] | None = None) -> list[Finding]:
    """Run every registered check. A check that raises is skipped, not fatal:
    one malformed file must not abort a 175k-file scan."""
    findings: list[Finding] = []
    for check in CHECKS:
        if only and check.check_id not in only:
            continue
        try:
            findings.extend(check.fn(model))
        except Exception as exc:  # noqa: BLE001 - defensive by design
            model.notes.append(f"{check.check_id} raised {type(exc).__name__}: {exc}")
    findings.sort(key=lambda f: (f.check_id, f.data.get("lineno", 0)))
    return findings


from .family1 import *  # noqa: E402,F401,F403  (populates CHECKS)
from .family2 import *  # noqa: E402,F401,F403
