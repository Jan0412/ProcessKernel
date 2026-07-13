"""F1.4 -- the host wrapper computes with PyTorch instead of Triton.

The headline cheating check. The model writes a Triton kernel for the easy part
and quietly leaves the expensive part to PyTorch::

    x = torch.conv2d(x, w)        # <- cuDNN does the real work
    relu_kernel[grid](x, out, n)  # <- the "Triton solution"

This passes a correctness harness, and a naive "contains @triton.jit" rule sees a
valid solution, but what was generated is a wrapper around cuDNN.

References:
  * AutoTriton (Li et al., 2025, arXiv:2507.05687) documents exactly this failure
    mode -- a Triton kernel for the ReLU, a PyTorch fallback for the convolution.
  * TritonRL (Woo et al., 2025, arXiv:2510.17891) uses a rule-based linter that
    "detects actual calls of Triton kernels and flags reliance on PyTorch modules".
  * The FLOP-weighted severity is a static analogue of Dr. Kernel's profiling ratio
    PR = T_generated / T_total (arXiv:2602.05885), which measures the same thing at
    runtime.

Design: an allowlist, not a blocklist. Allocation and shape metadata are legitimate
wrapper code; anything that launches a compute kernel is a fallback. Ops that merely
move memory (``contiguous``, ``zeros``) are not cheating and are handled by Family 2.
"""

from __future__ import annotations

import ast

from ...model import Finding, ModuleModel
from ...hostflow import _base_name
from .. import register

#: Ops that dominate a task's FLOPs. Falling back on one of these means the model
#: did not implement the problem.
HEAVY_OPS = {
    "matmul", "mm", "bmm", "addmm", "baddbmm", "einsum", "linear", "dot", "mv",
    "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "scaled_dot_product_attention",
    "softmax", "log_softmax",
    "layer_norm", "batch_norm", "group_norm", "instance_norm", "rms_norm",
    "max_pool1d", "max_pool2d", "max_pool3d",
    "avg_pool1d", "avg_pool2d", "avg_pool3d",
    "sort", "topk", "cumsum", "cumprod",
}

#: Real compute, but cheap relative to the heavy ops above.
LIGHT_OPS = {
    "relu", "gelu", "silu", "sigmoid", "tanh", "elu", "leaky_relu", "hardtanh",
    "exp", "log", "log2", "log10", "sqrt", "rsqrt", "pow", "abs", "sign",
    "sum", "mean", "prod", "std", "var", "norm", "amax", "amin",
    "argmax", "argmin", "clamp", "clip", "where", "sigmoid_", "dropout",
    "add", "sub", "mul", "div", "neg", "reciprocal", "erf", "floor", "ceil",
    "round", "maximum", "minimum", "gather", "scatter", "scatter_add",
    "index_select", "masked_fill", "cross_entropy", "mse_loss",
}

#: Legitimate wrapper plumbing -- never a fallback.
PLUMBING_OPS = {
    # allocation (torch.zeros is legitimate here; its memset cost is F2.4's business)
    "empty", "empty_like", "empty_strided", "zeros", "zeros_like", "ones",
    "ones_like", "full", "full_like", "tensor", "as_tensor", "from_numpy",
    # shape / layout metadata (no kernel launched)
    "view", "reshape", "permute", "transpose", "squeeze", "unsqueeze", "expand",
    "t", "flatten", "ravel", "narrow", "as_strided", "detach", "item",
    "size", "stride", "shape", "numel", "dim", "data_ptr", "element_size",
    "is_contiguous", "is_cuda", "get_device",
    # memory movement -- costs a kernel, but reported by F2.3 not F1.4
    "contiguous", "clone", "to", "cuda", "cpu", "type", "float", "half", "long",
    "int", "bool", "double", "bfloat16", "cat", "stack", "repeat",
    # launch plumbing
    "cdiv", "next_power_of_2", "range", "len", "int", "min", "max",
}

BINOP_NAMES = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.MatMult: "@", ast.Pow: "**", ast.FloorDiv: "//", ast.Mod: "%",
}


def _is_functional(qualname: str) -> bool:
    """``F.foo`` / ``torch.nn.functional.foo`` is always compute."""
    return qualname.startswith("F.") or "nn.functional." in qualname


def _op_of(qualname: str) -> str:
    return qualname.rsplit(".", 1)[-1]


def _host_scopes(model: ModuleModel) -> set[str]:
    """Reachable functions, excluding ``__init__`` (weight prep there is legitimate)."""
    scopes = model.reachable if model.entry else set(model.functions)
    return {s for s in scopes if not s.endswith(".__init__")}


@register("F1.4", "torch_fallback", "fail")
def check(model: ModuleModel) -> list[Finding]:
    if model.tree is None:
        return []

    scopes = _host_scopes(model)
    heavy: list[tuple[str, int]] = []
    light: list[tuple[str, int]] = []

    for call in model.host_calls:
        if call.enclosing not in scopes:
            continue
        op = _op_of(call.qualname)
        if op in PLUMBING_OPS and not _is_functional(call.qualname):
            continue
        if op in HEAVY_OPS or (_is_functional(call.qualname) and op not in PLUMBING_OPS):
            heavy.append((call.qualname, call.lineno))
        elif op in LIGHT_OPS:
            light.append((call.qualname, call.lineno))

    findings: list[Finding] = []

    if heavy or light:
        ops = [q for q, _ in heavy] + [q for q, _ in light]
        lines = sorted({ln for _, ln in heavy + light})
        severity = "fail" if heavy else "warn"
        if heavy:
            listed = ", ".join(f"`{q}` (line {ln})" for q, ln in heavy)
            msg = (
                f"forward() computes with PyTorch instead of Triton: {listed}. "
                f"This is the dominant cost of the task -- it must be implemented as a "
                f"Triton kernel, not handed back to PyTorch."
            )
        else:
            listed = ", ".join(f"`{q}` (line {ln})" for q, ln in light)
            msg = (
                f"forward() still uses PyTorch operators: {listed}. Fold these into the "
                f"Triton kernel so no work is left to PyTorch."
            )
        findings.append(
            Finding(
                check_id="F1.4",
                severity=severity,
                message=msg,
                data={
                    "ops": sorted(set(ops)),
                    "heavy_ops": sorted({q for q, _ in heavy}),
                    "linenos": lines,
                    "lineno": lines[0] if lines else 0,
                },
            )
        )

    findings.extend(_tensor_binops(model, scopes))
    return findings


def _tensor_binops(model: ModuleModel, scopes: set[str]) -> list[Finding]:
    """Elementwise arithmetic written as an operator (``a + b``) rather than a call.

    Lower confidence than the call scan: we only fire when an operand traces back to
    a forward input or to a tensor a kernel wrote, which is as close to type inference
    as we get without running anything.
    """
    from ...hostflow import scoped

    hits: list[tuple[str, int]] = []

    for qual in sorted(scopes):
        node = model.functions.get(qual)
        if node is None:
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.BinOp):
                continue
            symbol = BINOP_NAMES.get(type(sub.op))
            if symbol is None:
                continue
            for operand in (sub.left, sub.right):
                name = _base_name(operand)
                if name is None:
                    continue
                buf = model.buffers.get(model.canonical(scoped(qual, name)))
                if buf is not None and (buf.is_forward_input or buf.stored_by):
                    hits.append((symbol, sub.lineno))
                    break

    if not hits:
        return []

    lines = sorted({ln for _, ln in hits})
    symbols = sorted({s for s, _ in hits})
    severity = "fail" if "@" in symbols else "warn"
    return [
        Finding(
            check_id="F1.4",
            severity=severity,
            message=(
                f"forward() performs tensor arithmetic in PyTorch using the "
                f"{', '.join('`' + s + '`' for s in symbols)} operator "
                f"(line{'s' if len(lines) > 1 else ''} {', '.join(map(str, lines))}). "
                f"Each of these launches a PyTorch kernel -- do the arithmetic inside "
                f"the Triton kernel instead."
            ),
            data={"ops": symbols, "linenos": lines, "lineno": lines[0], "kind": "binop"},
        )
    ]
