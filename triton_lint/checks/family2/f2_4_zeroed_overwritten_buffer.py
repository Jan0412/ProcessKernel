"""F2.4 -- a buffer is zero-initialised and then completely overwritten.

    out = torch.zeros_like(x)        # pays for a full memset kernel...
    add_kernel[grid](x, y, out, n)   # ...then overwrites every element

``torch.zeros`` launches a memset: a full write pass over the tensor. If the kernel
unconditionally stores to the buffer and never accumulates into it, that entire pass
is wasted -- ``torch.empty_like`` gives the same result for free.

The false-positive guard that matters: if the kernel uses ``tl.atomic_add`` (or
otherwise reads the buffer back as an accumulator), the zero-init is **required**, and
"fixing" this would introduce a correctness bug. So we check the parameter's role
first: fire only when the kernel stores to it, never loads it, and performs no atomic
on it.

Note that a mask like ``offs < n`` does not disqualify the finding -- the mask only
prevents out-of-bounds writes; every in-bounds element is still written.

Reference: no direct paper precedent; this is the memory-traffic principle from
Liger-Kernel (arXiv:2410.10989) -- a wasted full-tensor pass -- applied to allocation.
"""

from __future__ import annotations

from ...hostflow import _base_name, scoped
from ...model import Finding, ModuleModel
from .. import register
from ._common import fmt_bytes, fmt_time, transfer_time

ZERO_ALLOCS = {"zeros", "zeros_like"}


@register("F2.4", "zeroed_overwritten_buffer", "warn")
def check(model: ModuleModel) -> list[Finding]:
    launches = {ls.index: ls for ls in model.reachable_launches}
    findings: list[Finding] = []

    for buf in model.buffers.values():
        if buf.alloc_fn not in ZERO_ALLOCS or not buf.stored_by:
            continue

        # Is the buffer ever read back -- by a kernel, or by host code?
        if buf.loaded_by or buf.read_by_host:
            continue

        # Examine only the parameter *this* buffer is bound to, not every stored param
        # of the kernel: a sibling atomic output (`hist`) must not suppress the finding
        # for an independent, genuinely-wasted buffer (`out`). Skip when that param
        # accumulates (atomic/reads it back -- zero-init required) or writes it only
        # partially (a diagonal/strided store leaves elements at their zero value, so
        # "use empty_like" would be a correctness bug).
        accumulating = False
        partial = False
        writers: list[str] = []
        for index in buf.stored_by:
            launch = launches.get(index)
            if launch is None:
                continue
            kernel = model.kernels.get(launch.kernel_name)
            if kernel is None:
                continue
            writers.append(kernel.name)
            for param, expr in launch.arg_map.items():
                role = kernel.params.get(param)
                if role is None or not role.stored:
                    continue
                var = _base_name(expr)
                if var is None or model.canonical(scoped(launch.enclosing, var)) != buf.canonical:
                    continue  # a sibling output of the same kernel, not this buffer
                if role.atomic or role.loaded:
                    accumulating = True
                if role.partial_store:
                    partial = True

        if accumulating or partial or not writers:
            continue

        name = buf.canonical.split("::")[-1]
        cost = ""
        if buf.nbytes:
            cost = (
                f" That memset moves {fmt_bytes(buf.nbytes)} "
                f"(~{fmt_time(transfer_time(buf.nbytes))})."
            )

        findings.append(
            Finding(
                check_id="F2.4",
                severity="warn",
                message=(
                    f"`{name}` is allocated with `torch.{buf.alloc_fn}` (line "
                    f"{buf.alloc_lineno}) but every element is then written by "
                    f"`{writers[0]}`, which performs no accumulation. The zero-fill is a "
                    f"wasted full write pass -- use `torch.empty_like` instead.{cost}"
                ),
                data={
                    "buffer": name,
                    "alloc_fn": buf.alloc_fn,
                    "writers": writers,
                    "bytes": buf.nbytes,
                    "lineno": buf.alloc_lineno or 0,
                },
            )
        )

    return findings
