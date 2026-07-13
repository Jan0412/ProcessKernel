"""F2.3 -- hidden kernels in the host wrapper: layout conversions and copies.

The most underrated check here, because it catches kernels that a ``@triton.jit``
counter thinks are perfectly clean. The key distinction is which shape ops are free
and which secretly launch a kernel:

  free (metadata only):   view, reshape, permute, transpose, squeeze, unsqueeze,
                          expand, slicing
  costs a full pass:      .contiguous() **on a non-contiguous tensor**, .clone(),
                          .to(dtype), torch.cat, torch.stack, .repeat

``x.permute(0, 2, 1).contiguous()`` is a hidden kernel doing 2 x numel x itemsize
bytes of traffic, sitting in host code where nobody is looking for it.

Reference: KernelEvolve (Meta, 2025, arXiv:2512.23236). Its Conv2d case study is
exactly this -- the unfused path launches unsqueeze (free), a *layout conversion* (a
real kernel, a full tensor pass), the conv, and squeeze (free). Four "kernels", only
one of which is visible as one. A decorator-counting analysis scores that file as
clean and misses the cost entirely.

The fix we suggest for the permute->contiguous case is what Triton's API is for:
kernels take stride arguments precisely so the host does not have to materialise a
transposed copy.
"""

from __future__ import annotations

import ast

from ...hostflow import LAYOUT_METHODS, _base_name, scoped
from ...model import Finding, ModuleModel
from ...parsing import _dotted
from .. import register
from ._common import fmt_bytes, fmt_time, transfer_time
from ..family1.f1_4_torch_fallback import _host_scopes

#: Host ops that launch a real kernel and cost a full pass over the tensor.
COPY_OPS = {"clone", "cat", "stack", "repeat", "repeat_interleave", "contiguous", "to"}

DTYPES = {
    "float", "float16", "float32", "float64", "half", "bfloat16", "double",
    "int", "int8", "int16", "int32", "int64", "long", "short", "bool", "uint8",
}


def _is_dtype_cast(call: ast.Call) -> bool:
    """``x.to(torch.float32)`` costs a pass; ``x.to(device)`` / ``x.to('cuda')`` does not."""
    for arg in call.args:
        name = _dotted(arg)
        if name and name.rsplit(".", 1)[-1] in DTYPES:
            return True
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return False  # a device string
    for kw in call.keywords:
        if kw.arg == "dtype":
            return True
    return False


def _receiver_is_noncontiguous(model: ModuleModel, scope: str, node: ast.expr) -> bool:
    """True when the receiver provably has a non-contiguous layout.

    Handles both the chained form ``x.permute(0,2,1).contiguous()`` -- caught
    structurally, by walking the call chain -- and the bound form
    ``xt = x.permute(0,2,1); xt.contiguous()``, which needs the layout set.

    Note the bound form must be looked up by the **raw scoped name**, not the
    canonical one. ``model.noncontiguous`` records the alias itself (``forward::xt``),
    whereas canonicalisation deliberately resolves an alias back to the storage it
    views (``forward::x``) -- which is the contiguous base, and never in the set. Asking
    for the canonical name here is therefore guaranteed to answer False.
    """
    if isinstance(node, ast.Call):
        fn = _dotted(node.func) or ""
        if fn.rsplit(".", 1)[-1] in LAYOUT_METHODS:
            return True
        return _receiver_is_noncontiguous(model, scope, node.func)
    if isinstance(node, ast.Attribute):
        if node.attr in ("T", "mT"):
            return True
        return _receiver_is_noncontiguous(model, scope, node.value)
    if isinstance(node, ast.Name):
        return scoped(scope, node.id) in model.noncontiguous
    return False


@register("F2.3", "layout_churn", "warn")
def check(model: ModuleModel) -> list[Finding]:
    scopes = _host_scopes(model)
    findings: list[Finding] = []

    for call in model.host_calls:
        if call.enclosing not in scopes or call.node is None:
            continue
        op = call.qualname.rsplit(".", 1)[-1]
        if op not in COPY_OPS:
            continue

        node = call.node
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None

        if op == "contiguous":
            if receiver is None or not _receiver_is_noncontiguous(model, call.enclosing, receiver):
                continue  # .contiguous() on an already-contiguous tensor is a no-op
            message_fix = (
                "Triton kernels take stride arguments -- pass the tensor's strides and "
                "index the permuted layout directly inside the kernel instead of "
                "materialising a transposed copy."
            )
        elif op == "to":
            if not _is_dtype_cast(node):
                continue  # a device move, not a cast
            message_fix = (
                "Do the cast inside the kernel on load (`tl.load(...).to(tl.float32)`) "
                "rather than casting the whole tensor in host code."
            )
        else:
            message_fix = (
                "This launches a kernel and moves the whole tensor; fold it into your "
                "Triton kernel's indexing if possible."
            )

        nbytes = _receiver_bytes(model, call.enclosing, receiver) if receiver else None
        cost = ""
        if nbytes:
            traffic = 2 * nbytes  # read + write
            cost = (
                f" It moves {fmt_bytes(traffic)} of HBM traffic "
                f"(~{fmt_time(transfer_time(traffic))})."
            )

        findings.append(
            Finding(
                check_id="F2.3",
                severity="warn",
                message=(
                    f"`{call.qualname}()` (line {call.lineno}) launches a hidden PyTorch "
                    f"kernel that copies the whole tensor.{cost} {message_fix}"
                ),
                data={
                    "op": op,
                    "qualname": call.qualname,
                    "bytes": 2 * nbytes if nbytes else None,
                    "lineno": call.lineno,
                },
            )
        )

    return findings


def _receiver_bytes(model: ModuleModel, scope: str, node: ast.expr) -> int | None:
    name = _base_name(node)
    if name is None:
        return None
    buf = model.buffers.get(model.canonical(scoped(scope, name)))
    return buf.nbytes if buf else None
