"""F2.1 -- an intermediate tensor is materialised in HBM between two launches.

    tmp = torch.empty_like(x)
    exp_kernel[grid](x, tmp, n)      # tmp is written to HBM...
    out = torch.empty_like(x)
    scale_kernel[grid](tmp, out, n)  # ...and immediately read back
    return out                       # tmp never escapes

``tmp`` makes a full round trip through HBM -- 2 x numel x itemsize bytes -- for no
reason. Because **Triton has no cross-launch fusion pass**, this is not a heuristic
guess about performance: the write and the read are guaranteed to happen. A fused
kernel would keep the value in registers.

References:
  * Liger-Kernel (Hsu et al., 2024, arXiv:2410.10989): fusing ops into monolithic
    Triton kernels eliminates intermediate materialisation and HBM<->SRAM traffic
    (~20% training throughput, ~60% memory reduction). This is the quantitative
    justification for treating a materialised intermediate as an anti-pattern.
  * KernelEvolve (Meta, 2025, arXiv:2512.23236): the Conv2d case, where the unfused
    path pays a full tensor pass per auxiliary kernel.

Why we gate the *suggestion* on provable fusibility: KernelBenchX (arXiv:2605.04956)
found that 72% of Fusion tasks fail across all methods, and that iterative refinement
raises the compile rate (52.3% -> 68.8%) while *lowering* average speedup (1.58x ->
1.44x). Telling a model to fuse a pair that cannot be legally fused is how you
reproduce that degradation. So we only suggest a fusion when the producer/consumer
iteration spaces are compatible; otherwise we report the cost and stay silent about
the fix.
"""

from __future__ import annotations

from ....core.check import Check
from ....core.model import Buffer, Finding, KernelDef, LaunchSite, ModuleModel
from .. import LINT_REGISTRY
from ._common import fmt_bytes, fmt_time, transfer_time

#: (producer kind, consumer kind) pairs that can always be fused into one kernel.
FUSIBLE = {
    ("elementwise", "elementwise"),  # inline the second computation
    ("elementwise", "reduction"),    # the reduction loads and transforms before reducing
    ("copy", "elementwise"),
    ("copy", "reduction"),
    ("matmul", "elementwise"),       # epilogue fusion
}

# Deliberately absent: ("reduction", "elementwise") -- fusible only if the reduced
# axis fits in a single block (the softmax case), which we do not try to prove; and
# ("reduction", "reduction") over different axes, which is not fusible at all.


def _is_dead_intermediate(buf: Buffer) -> bool:
    return (
        len(buf.stored_by) == 1
        and bool(buf.loaded_by)
        and buf.stored_by[0] not in buf.loaded_by  # not an in-place accumulator
        and not buf.returned
        and not buf.read_by_host
        and not buf.is_forward_input
        and buf.alloc_fn is not None
    )


@LINT_REGISTRY.add
class DeadIntermediate(Check):
    check_id = "F2.1"
    name = "dead_intermediate"
    severity = "warn"

    def run(self, model: ModuleModel) -> list[Finding]:
        launches = {ls.index: ls for ls in model.reachable_launches}
        if len(launches) < 2:
            return []

        intermediates: list[tuple[Buffer, LaunchSite, list[LaunchSite]]] = []
        for buf in model.buffers.values():
            if not _is_dead_intermediate(buf):
                continue
            producer = launches.get(buf.stored_by[0])
            consumers = [launches[i] for i in buf.loaded_by if i in launches]
            if producer is None or not consumers:
                continue
            intermediates.append((buf, producer, consumers))

        if not intermediates:
            return []

        # Merge producer->consumer chains (k1 -> t1 -> k2 -> t2 -> k3) into one finding,
        # so the model gets "fuse these three", not three separate suggestions.
        chains = _build_chains(intermediates)

        findings: list[Finding] = []
        for chain in chains:
            findings.append(self._finding_for(model, chain))
        return findings


    def _finding_for(self, model: ModuleModel, chain) -> Finding:
        buffers = [buf for buf, _, _ in chain]
        total_bytes = sum(2 * b.nbytes for b in buffers if b.nbytes)
        known_bytes = all(b.nbytes for b in buffers)

        pairs: list[tuple[str, str, bool]] = []
        for buf, producer, consumers in chain:
            pk = _kernel_of(model, producer)
            for consumer in consumers:
                ck = _kernel_of(model, consumer)
                if pk is None or ck is None:
                    continue
                pairs.append((pk.name, ck.name, (pk.kind, ck.kind) in FUSIBLE))

        fusible = bool(pairs) and all(ok for _, _, ok in pairs)
        names = ", ".join(f"`{b.canonical.split('::')[-1]}`" for b in buffers)
        kernel_names = []
        for pk, ck, _ in pairs:
            for name in (pk, ck):
                if name not in kernel_names:
                    kernel_names.append(name)

        cost = ""
        if known_bytes and total_bytes:
            cost = (
                f" This costs {fmt_bytes(total_bytes)} of HBM traffic "
                f"(~{fmt_time(transfer_time(total_bytes))} at achievable bandwidth)."
            )

        if fusible:
            chain_desc = " -> ".join(f"`{k}`" for k in kernel_names)
            message = (
                f"{names} {'is' if len(buffers) == 1 else 'are'} written by one kernel and "
                f"immediately read by the next, and used nowhere else -- so "
                f"{'it' if len(buffers) == 1 else 'they'} round-trip(s) through HBM for "
                f"nothing (Triton does not fuse across kernel launches). "
                f"Fuse {chain_desc} into a single kernel and keep the intermediate in "
                f"registers.{cost}"
            )
            severity = "warn"
        else:
            kinds = ", ".join(
                f"`{pk}` ({_kind(model, pk)}) -> `{ck}` ({_kind(model, ck)})" for pk, ck, _ in pairs
            )
            message = (
                f"{names} {'is' if len(buffers) == 1 else 'are'} materialised in HBM between "
                f"kernels ({kinds}).{cost} These iteration spaces are not trivially fusible, "
                f"so only fuse them if the reduction fits in a single block."
            )
            severity = "info"

        return self.finding(
            message,
            severity=severity,
            intermediates=[b.canonical.split("::")[-1] for b in buffers],
            kernels=kernel_names,
            fusible=fusible,
            bytes=total_bytes if known_bytes else None,
            lineno=min((b.alloc_lineno or 0) for b in buffers),
        )


def _build_chains(
    intermediates: list[tuple[Buffer, LaunchSite, list[LaunchSite]]],
) -> list[list[tuple[Buffer, LaunchSite, list[LaunchSite]]]]:
    """Group intermediates into connected components over the launches they touch.

    A single greedy pass cannot merge two chains once a later intermediate bridges
    them (a diamond: two producers feeding one consumer, one producer itself fed by an
    earlier launch), so the shared consumer would land in two contradictory "Fuse …"
    findings. A union-find over launch indices merges the whole component regardless of
    processing order -- one finding per connected component.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for _, producer, consumers in intermediates:
        for consumer in consumers:
            union(producer.index, consumer.index)

    groups: dict[int, list[tuple[Buffer, LaunchSite, list[LaunchSite]]]] = {}
    for item in sorted(intermediates, key=lambda t: t[1].index):
        groups.setdefault(find(item[1].index), []).append(item)

    return [groups[root] for root in sorted(groups, key=lambda r: groups[r][0][1].index)]


def _kernel_of(model: ModuleModel, launch: LaunchSite) -> KernelDef | None:
    return model.kernels.get(launch.kernel_name)


def _kind(model: ModuleModel, kernel_name: str) -> str:
    kernel = model.kernels.get(kernel_name)
    return kernel.kind if kernel else "unknown"
