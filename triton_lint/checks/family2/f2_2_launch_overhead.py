"""F2.2 -- launch overhead, and the far worse case of launching inside a Python loop.

Two findings from one analysis:

**launch_in_loop** (fail). A kernel launched from a host-side Python ``for`` loop --
say, iterating over the batch and launching once per sample -- means N serialised
launches, N x the overhead, and a dimension the model failed to put in the grid. This
is the most severe form of the pattern and it is trivially actionable.

**launch_overhead** (info/warn). Each launch costs ~5 us of CPU-side launch plus GPU
scheduling. On a typical KernelBench Level-1 problem (a 4 MB elementwise op) the
memory time is ~3 us -- so four launches means ~20 us of overhead against ~3 us of
actual work, and kernel count is the *dominant* term in runtime.

Note we deliberately do **not** lint on the count. "You have too many kernels" is a
message a model can satisfy by fusing the wrong pair; the actionable instruction comes
from F2.1, which names a specific, provably-safe pair. What F2.2 contributes is the
*regime*: it tells the model why fusing matters here, which makes it far more likely
to do it properly rather than cosmetically.

Reference: KernelEvolve (Meta, 2025, arXiv:2512.23236) -- the Conv2d case, where the
PyTorch workaround launches four auxiliary kernels (unsqueeze, layout conversion,
conv, squeeze), each incurring a full tensor pass, and the fused Triton solution wins.
"""

from __future__ import annotations

from ...model import Finding, ModuleModel
from .. import register
from ._common import LAUNCH_OVERHEAD, fmt_bytes, fmt_time, transfer_time

#: Below this, a launch-count finding is noise.
LAUNCH_WARN_THRESHOLD = 3


@register("F2.2", "launch_overhead", "warn")
def check(model: ModuleModel) -> list[Finding]:
    launches = model.reachable_launches
    if not launches:
        return []

    findings: list[Finding] = []

    for launch in launches:
        if launch.loop_depth > 0:
            loop = " / ".join(launch.loop_vars) or "a loop"
            findings.append(
                Finding(
                    check_id="F2.2",
                    severity="fail",
                    message=(
                        f"`{launch.kernel_name}` is launched inside a Python loop over "
                        f"{loop} (line {launch.lineno}). Every iteration is a separate, "
                        f"serialised kernel launch (~{fmt_time(LAUNCH_OVERHEAD)} of pure "
                        f"overhead each). Move that dimension into the launch grid and "
                        f"launch the kernel once."
                    ),
                    data={
                        "kernel": launch.kernel_name,
                        "loop_vars": launch.loop_vars,
                        "loop_depth": launch.loop_depth,
                        "lineno": launch.lineno,
                        "kind": "launch_in_loop",
                    },
                )
            )

    n = len(launches)
    if n >= LAUNCH_WARN_THRESHOLD:
        overhead = n * LAUNCH_OVERHEAD
        regime = ""
        essential = _essential_bytes(model)
        if essential:
            mem_time = transfer_time(essential)
            if overhead > mem_time:
                regime = (
                    f" At this problem size ({fmt_bytes(essential)} of essential I/O) the "
                    f"launch overhead exceeds the memory-transfer time "
                    f"(~{fmt_time(mem_time)}), so kernel count dominates the runtime."
                )
        findings.append(
            Finding(
                check_id="F2.2",
                severity="warn" if regime else "info",
                message=(
                    f"forward() launches {n} Triton kernels "
                    f"(~{fmt_time(overhead)} of launch overhead).{regime}"
                ),
                data={
                    "n_launches": n,
                    "overhead_s": overhead,
                    "kernels": [ls.kernel_name for ls in launches],
                    "kind": "launch_count",
                    "lineno": min(ls.lineno for ls in launches),
                },
            )
        )

    return findings


def _essential_bytes(model: ModuleModel) -> int | None:
    """Bytes that any correct implementation must move: read the inputs once."""
    total = 0
    for info in model.input_shapes:
        if not info:
            return None
        shape, dtype = info
        from ...shapes import _nbytes

        total += _nbytes(shape, dtype)
    return total or None
