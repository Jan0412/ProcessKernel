"""F1.3 -- a kernel is launched, but its result is thrown away.

The subtle sibling of F1.2. Reachability analysis passes this file happily -- the
kernel *is* launched -- and only dataflow catches it::

    def forward(self, x):
        out = torch.empty_like(x)
        relu_kernel[(n,)](x, out, x.numel())   # out is a store target...
        return torch.relu(x)                   # ...and is then discarded

Reference: this is an extension beyond the reachability gate of arXiv:2602.11000
(which would accept the code above). The runtime analogue is Dr. Kernel / KernelGYM
(arXiv:2602.05885), which instruments Triton's launch path to detect kernels that
"execute no code". Doing it statically means no GPU is needed.

Conservative by construction: we only fire when we resolved at least one store-target
buffer for the launch, so an unresolvable argument yields silence, not a false alarm.
"""

from __future__ import annotations

from ....core.model import Finding, ModuleModel
from ...hostflow import _base_name, scoped
from .. import register


@register("F1.3", "discarded_output", "fail")
def check(model: ModuleModel) -> list[Finding]:
    findings: list[Finding] = []

    for launch in model.reachable_launches:
        kernel = model.kernels.get(launch.kernel_name)
        if kernel is None:
            continue

        outputs = kernel.outputs()
        if not outputs:
            continue  # nothing is written; not this check's business

        resolved = []
        for param in outputs:
            expr = launch.arg_map.get(param)
            if expr is None:
                continue
            var = _base_name(expr)
            if var is None:
                continue
            buf = model.buffers.get(model.canonical(scoped(launch.enclosing, var)))
            if buf is not None:
                resolved.append((var, buf))

        if not resolved:
            continue  # could not resolve the output tensors -- stay quiet

        # A buffer consumed by a *different* launch (the next kernel in a pipeline) is
        # used, not discarded. Loading by this same launch does not count -- an atomic
        # accumulator is stored and loaded by its own launch, and discarding it is still
        # a real F1.3 hit.
        def used(buf) -> bool:
            return (
                buf.returned
                or buf.read_by_host
                or buf.is_forward_input
                or any(idx != launch.index for idx in buf.loaded_by)
            )

        if any(used(b) for _, b in resolved):
            continue

        names = ", ".join(f"`{v}`" for v, _ in resolved)
        findings.append(
            Finding(
                check_id="F1.3",
                severity="fail",
                message=(
                    f"`{launch.kernel_name}` (line {launch.lineno}) writes its result to "
                    f"{names}, but that tensor is never returned or used afterwards. The "
                    f"kernel runs and its output is discarded -- return the kernel's result."
                ),
                data={
                    "kernel": launch.kernel_name,
                    "outputs": [v for v, _ in resolved],
                    "lineno": launch.lineno,
                },
            )
        )

    return findings
