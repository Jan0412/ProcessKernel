"""What a check is, and what a collection of them is.

A check used to be a bare function plus a ``@register`` decorator, which left three things
by convention that are now structural.

**The id was retyped at every finding.** A check declared ``"F1.4"`` in its decorator and
then again inside each ``Finding`` it built, so the two could drift apart silently.
:meth:`Check.finding` stamps it once from the ClassVar.

**``data["lineno"]`` was load-bearing but unenforced.** ``Registry.run`` sorts on it and
``kernel_gen.inspect_trace`` joins findings to source lines through it, yet nothing made a
check remember it. It is now a keyword argument. When it is ``None`` the key is *omitted*
rather than set to 0 -- F1.1 is a whole-file finding with no line to point at, and
inventing line 1 for it would be a lie the trace join would then act on.

**The registry was a module global.** One list populated by import side effect works for
exactly one family of checks; registering a second analyzer's checks into it would fire
them on every lint call. A :class:`Registry` is an instance, so the linter and the
submission checker are independently runnable and independently ablatable.

``severity`` is a ClassVar *default*, not a constraint: five of the F1/F2 findings decide
severity from what they found (``"fail" if heavy else "warn"``), so ``run`` may override it
per finding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from .model import Finding, ModuleModel, Severity


class Check(ABC):
    check_id: ClassVar[str]
    name: ClassVar[str]
    severity: ClassVar[Severity] = "warn"

    def __init_subclass__(cls, **kwargs) -> None:
        # Fail at import, not at scan time: a check with no id would otherwise register
        # fine and only blow up once a real file reached it.
        super().__init_subclass__(**kwargs)
        for attribute in ("check_id", "name"):
            if not getattr(cls, attribute, None):
                raise TypeError(f"{cls.__name__} must set {attribute}")

    @abstractmethod
    def run(self, model: ModuleModel) -> list[Finding]:
        """Findings for this model.

        Should not raise -- ``Registry.run`` catches, but that is a backstop for malformed
        input, not a licence to let bugs through.
        """

    def finding(self, message: str, *, severity: Severity | None = None, **data) -> Finding:
        """The only way a check builds a Finding: stamps the id, defaults the severity.

        ``lineno`` is an ordinary keyword like any other payload key, and is deliberately
        *not* hoisted to the front of ``data``. Measured over the shipped corpus, no check
        puts it first -- it is last in six of them and mid-payload in three -- so hoisting
        it would reorder the serialised payload of some 13,000 findings for no gain.
        Omitting the keyword omits the key rather than defaulting it to 0, which is what
        F1.1 needs: a whole-file finding has no line, and ``inspect_trace`` joins on this
        key, so inventing line 1 would point the join at unrelated source.
        """
        return Finding(
            check_id=self.check_id,
            severity=severity or self.severity,
            message=message,
            data=data,
        )


class Registry:
    """A named, independent collection of checks."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._checks: list[Check] = []

    def add(self, check: type[Check]) -> type[Check]:
        """Decorator. Stores an instance -- checks are stateless, so one is enough, and it
        keeps ``monkeypatch.setattr(registry.checks[0], "run", ...)`` a one-line test."""
        self._checks.append(check())
        return check

    @property
    def checks(self) -> list[Check]:
        return self._checks

    @property
    def check_ids(self) -> list[str]:
        return [c.check_id for c in self._checks]

    def run(self, model: ModuleModel, only: set[str] | None = None) -> list[Finding]:
        """Run every registered check. A check that raises is skipped, not fatal:
        one malformed file must not abort a 175k-file scan."""
        findings: list[Finding] = []
        for check in self._checks:
            if only and check.check_id not in only:
                continue
            try:
                findings.extend(check.run(model))
            except Exception as exc:  # noqa: BLE001 - defensive by design
                model.notes.append(f"{check.check_id} raised {type(exc).__name__}: {exc}")
        findings.sort(key=lambda f: (f.check_id, f.data.get("lineno", 0)))
        return findings
