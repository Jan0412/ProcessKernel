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

Two axes are kept separate. The *spelling* (``F.foo`` / ``torch.nn.functional.foo``)
answers "is this compute rather than plumbing"; the op lists answer "is it heavy".
An unclassified functional op is therefore still flagged, but only as ``warn`` --
spelling alone cannot make something "the dominant cost of the task"
(``F._Reduction.get_enum``, an internal enum lookup, disproved that). And only code
the benchmark's forward() actually executes is scanned (``ModuleModel.timed_scopes``):
autograd ``backward``/``jvp``/``vmap`` never run under the timed forward, so torch
ops there are not fallbacks.
"""

from __future__ import annotations

import ast

from ...model import Finding, ModuleModel
from ...hostflow import _base_name, scoped
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
    "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d",
    "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d",
    "interpolate", "upsample", "upsample_bilinear", "upsample_nearest", "grid_sample",
    "multi_head_attention_forward",
    "sort", "topk", "cumsum", "cumprod",
}

#: Heavy only in the functional spelling: ``F.unfold`` materialises im2col, but the
#: tensor method ``x.unfold(...)`` is a view and must not match by bare op name.
HEAVY_FUNCTIONAL_ONLY = {"unfold", "fold"}

#: Real compute, but cheap relative to the heavy ops above.
LIGHT_OPS = {
    "relu", "gelu", "silu", "sigmoid", "tanh", "elu", "leaky_relu", "hardtanh",
    "exp", "log", "log2", "log10", "sqrt", "rsqrt", "pow", "abs", "sign",
    "sum", "mean", "prod", "std", "var", "norm", "amax", "amin",
    "argmax", "argmin", "clamp", "clip", "where", "sigmoid_", "dropout",
    "add", "sub", "mul", "div", "neg", "reciprocal", "erf", "floor", "ceil",
    "round", "maximum", "minimum", "gather", "scatter", "scatter_add",
    "index_select", "masked_fill", "cross_entropy", "mse_loss",
    # elementwise activations (memory-bound, one cheap pass)
    "softplus", "mish", "selu", "celu", "relu6", "hardswish", "hardsigmoid",
    "logsigmoid", "log_sigmoid", "softsign", "glu", "prelu", "rrelu", "threshold",
    "hardshrink", "softshrink", "tanhshrink",
    # elementwise math
    "log1p", "expm1", "exp2", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "erfc", "square", "logsumexp",
    # memory-shaped ops that still launch one cheap kernel
    "normalize", "pad", "one_hot", "pixel_shuffle", "pixel_unshuffle",
    # losses: one cheap reduction
    "binary_cross_entropy", "binary_cross_entropy_with_logits", "nll_loss",
    "kl_div", "l1_loss", "smooth_l1_loss", "huber_loss", "cosine_similarity",
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
    """A public op spelled through the functional namespace -- always compute.

    Requires exactly one public segment after the prefix: ``F.softplus`` is compute,
    ``F._Reduction.get_enum`` (an internal enum lookup, no kernel launched) is not.
    """
    if qualname.startswith("F."):
        rest = qualname[2:]
    elif "nn.functional." in qualname:
        rest = qualname.split("nn.functional.", 1)[1]
    else:
        return False
    return bool(rest) and "." not in rest and not rest.startswith("_")


def _op_of(qualname: str) -> str:
    return qualname.rsplit(".", 1)[-1]


#: Tensor methods / attributes that return a Python scalar, not a tensor. Arithmetic on
#: one (``x.numel() // 4``, ``x.shape[0] * 2``) is launch-grid math, not a torch kernel.
SCALAR_METHODS = {
    "numel", "size", "dim", "ndimension", "nelement", "element_size", "stride", "item",
}
SCALAR_ATTRS = {"shape", "ndim"}


def _is_scalar_expr(node: ast.expr) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in SCALAR_METHODS
    if isinstance(node, ast.Subscript):
        return _is_scalar_expr(node.value)
    if isinstance(node, ast.Attribute):
        return node.attr in SCALAR_ATTRS
    return False


#: Python builtins that collide with op tokens (``round``/``abs``/``pow`` in LIGHT_OPS,
#: ``sum`` too). A *bare* call to one (no namespace, not a tensor method) doing scalar
#: grid/shape math is not a torch op -- see BUG-31. ``min``/``max``/``int``/``float`` are
#: already PLUMBING_OPS.
PY_SCALAR_BUILTINS = {"round", "abs", "pow", "sum"}


def _operand_is_tensor(model: ModuleModel, scope: str, node: ast.expr) -> bool:
    """*node*'s leftmost name resolves to a tensor buffer (a forward input or a tensor a
    kernel wrote) -- as close to type inference as we get without running anything."""
    name = _base_name(node)
    if name is None:
        return False
    buf = model.buffers.get(model.canonical(scoped(scope, name)))
    return buf is not None and (buf.is_forward_input or bool(buf.stored_by))


def _call_touches_tensor(model: ModuleModel, scope: str, node: ast.Call) -> bool:
    """Any argument (recursing into list/tuple/generator elements) is a tensor. Used to
    tell a genuine ``sum([t1, t2])`` fallback from bare-scalar ``sum(l*l for l in levels)``."""
    for arg in list(node.args) + [kw.value for kw in node.keywords]:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Name):
                buf = model.buffers.get(model.canonical(scoped(scope, sub.id)))
                if buf is not None and (buf.is_forward_input or bool(buf.stored_by)):
                    return True
    return False


def _is_bare_scalar_builtin(model: ModuleModel, call, op: str) -> bool:
    if call.is_method or "." in call.qualname or op not in PY_SCALAR_BUILTINS:
        return False
    if call.node is None:
        return True  # a bare builtin with no resolvable args is scalar math
    return not _call_touches_tensor(model, call.enclosing, call.node)


def _nn_binding(model: ModuleModel, cls: str | None, attr: str) -> str | None:
    """The ``nn.*`` class bound to ``self.<attr>`` for a call made in class ``cls``.

    Checked per class (BUG-24): a binding in an unrelated class no longer decides how a
    same-named attribute is judged here. Falls back to the constructed entry class, where
    ``__init__`` bindings live when the forward is inherited (call is in a base method).
    """
    for candidate in (cls, model.model_class):
        binding = model.nn_modules_in_init.get(candidate or "", {})
        if attr in binding:
            return binding[attr]
    return None


def _is_local_call(model: ModuleModel, call) -> bool:
    """The call targets code the model wrote itself, not a torch op.

    A module-level helper (``def softmax(...)`` that launches a kernel) or a local
    submodule (``self.layer_norm = LayerNormTriton(...)``) may share a name with an op
    token; grading it a PyTorch fallback and telling the model to rewrite its own kernel
    is both wrong and destructive. An nn module invoked here is a genuine fallback
    (F1.5's domain) and must keep firing, so it takes precedence.
    """
    qualname = call.qualname
    if qualname in model.functions:
        return True
    if qualname.startswith("self."):
        attr = qualname.split(".", 1)[1].split(".", 1)[0]
        cls = call.enclosing.rsplit(".", 1)[0]
        if _nn_binding(model, cls, attr) is not None:
            return False  # a real nn.Module call in this class -- still a fallback
        if attr in model.attr_classes:
            return True  # local submodule assigned a file-defined class
        if model.model_class and f"{model.model_class}.{attr}" in model.functions:
            return True  # calling the model's own method
    return False


def _host_scopes(model: ModuleModel) -> set[str]:
    """Functions the timed forward() actually executes, excluding ``__init__``.

    Built on the precise ``timed_scopes`` walk, not the conservative ``reachable``
    set: autograd ``backward``/``jvp``/``vmap`` never run under the benchmark's
    forward call, so a torch op there is not a fallback. ``__init__`` is excluded
    because weight prep is legitimate even when an inline ``MyModule()(x)`` drags
    one onto the timed path.
    """
    scopes = model.timed_scopes if model.forward_entry else set(model.functions)
    return {s for s in scopes if not s.endswith(".__init__")}


@register("F1.4", "torch_fallback", "fail")
def check(model: ModuleModel) -> list[Finding]:
    if model.tree is None:
        return []

    scopes = _host_scopes(model)
    heavy: list[tuple[str, int]] = []
    light: list[tuple[str, int]] = []
    unknown: list[str] = []

    for call in model.host_calls:
        if call.enclosing not in scopes:
            continue
        if _is_local_call(model, call):
            continue  # model-authored code, not a torch fallback (F1.5 handles nn modules)
        if call.qualname.split(".", 1)[0] in ("tl", "triton"):
            continue  # BUG-26: a Triton builtin (tl.sum/tl.dot/...) in host scope is not torch
        op = _op_of(call.qualname)
        if _is_bare_scalar_builtin(model, call, op):
            continue  # BUG-31: bare round/abs/pow/sum on non-tensors is Python scalar math
        functional = _is_functional(call.qualname)
        if op in PLUMBING_OPS and not functional:
            continue
        if op in HEAVY_OPS or (functional and op in HEAVY_FUNCTIONAL_ONLY):
            heavy.append((call.qualname, call.lineno))
        elif op in LIGHT_OPS:
            light.append((call.qualname, call.lineno))
        elif functional:
            # unclassified compute: flagged, but spelling cannot make it heavy
            light.append((call.qualname, call.lineno))
            unknown.append(call.qualname)

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
        data = {
            "ops": sorted(set(ops)),
            "heavy_ops": sorted({q for q, _ in heavy}),
            "linenos": lines,
            "lineno": lines[0] if lines else 0,
        }
        if unknown:
            data["unknown_ops"] = sorted(set(unknown))
        findings.append(
            Finding(check_id="F1.4", severity=severity, message=msg, data=data)
        )

    findings.extend(_tensor_binops(model, scopes))
    return findings


def _tensor_binops(model: ModuleModel, scopes: set[str]) -> list[Finding]:
    """Elementwise arithmetic written as an operator (``a + b``) rather than a call.

    Lower confidence than the call scan: we only fire when an operand traces back to
    a forward input or to a tensor a kernel wrote, which is as close to type inference
    as we get without running anything.
    """
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
                if _is_scalar_expr(operand):
                    continue  # x.numel() // 4 is scalar grid math, not tensor arithmetic
                if _operand_is_tensor(model, qual, operand):
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
