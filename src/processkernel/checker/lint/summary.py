"""The lint report's summary: the generic counts plus what only F1/F2 can answer.

``fallback_ops`` and ``wasted_bytes_lower_bound`` read specific check ids out of the
findings, and the kernel/launch counts read the model, so none of it generalises to a
second analyzer. Key order is load-bearing: the summary is serialised with ``json.dumps``
and run dirs on disk carry the historical ordering, so the model-derived keys stay ahead of
the generic ones exactly as they were.
"""

from __future__ import annotations

from ..core.model import Finding, ModuleModel
from ..core.summary import build_summary


def lint_summary(model: ModuleModel, findings: list[Finding]) -> dict:
    launches = model.reachable_launches
    fallback_ops = sorted(
        {op for f in findings if f.check_id == "F1.4" for op in f.data.get("ops", [])}
    )
    wasted = sum(
        f.data["bytes"]
        for f in findings
        if f.check_id in ("F2.1", "F2.3", "F2.4") and isinstance(f.data.get("bytes"), int)
    )
    return {
        "n_kernels": len(model.kernels),
        "n_launches": len(launches),
        "launches_in_loop": sum(1 for ls in launches if ls.loop_depth > 0),
        "fallback_ops": fallback_ops,
        "wasted_bytes_lower_bound": wasted,
        **build_summary(findings),
    }
