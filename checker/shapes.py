"""Static shape inference, so Family 2 can report wasted traffic in *bytes*.

Bytes are the whole point: "you have too many kernels" is a suggestion a model can
satisfy by fusing the wrong pair, whereas "this intermediate costs 33.6 MB of HBM
traffic" names a specific, physical cost.

Sources of truth, in order:
  1. ``get_inputs()`` in the generated file gives the input tensor shapes -- the same
     AST trick used by kernel_gen/kernelbook_convert.py (``_func_name`` / ``_shape_elts``).
  2. ``B, C, H, W = x.shape`` / ``x.shape[0]`` / ``x.numel()`` bind integer names.
  3. ``torch.empty((B, H, W))`` / ``torch.empty_like(x)`` size the intermediates.

Anything we cannot resolve stays ``None``. We never guess: a finding without a byte
count is still useful, a finding with a wrong one is not.
"""

from __future__ import annotations

import ast
import math

from .hostflow import _base_name, scoped
from .model import ModuleModel
from .parsing import _dotted

ITEMSIZE = {
    "float32": 4, "float": 4, "float64": 8, "double": 8,
    "float16": 2, "half": 2, "bfloat16": 2,
    "int64": 8, "long": 8, "int32": 4, "int": 4, "int16": 2,
    "int8": 1, "uint8": 1, "bool": 1,
}

RANDOM_FNS = {"rand", "randn", "empty", "zeros", "ones", "full", "randint", "tensor"}


def _dtype_of(call: ast.Call, default: str = "float32") -> str:
    for kw in call.keywords:
        if kw.arg == "dtype":
            name = _dotted(kw.value)
            if name:
                return name.rsplit(".", 1)[-1]
    fn = _dotted(call.func) or ""
    if fn.endswith("randint"):
        return "int64"
    return default


def _const_shape(elts: list[ast.expr], env: dict[str, int]) -> tuple[int, ...] | None:
    out: list[int] = []
    for e in elts:
        val = _const_int(e, env)
        if val is None:
            return None
        out.append(val)
    return tuple(out)


def _const_int(node: ast.expr, env: dict[str, int]) -> int | None:
    """Evaluate an integer expression against known names. Returns None if unknown."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp):
        left = _const_int(node.left, env)
        right = _const_int(node.right, env)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Div):
                return left // right
        except ZeroDivisionError:
            return None
    return None


def _shape_from_call(call: ast.Call, env: dict[str, int]) -> tuple[int, ...] | None:
    """``torch.rand([4,4])`` / ``torch.rand(4, 4)`` / ``torch.randint(0, 9, [4,4])``."""
    fn = _dotted(call.func) or ""
    op = fn.rsplit(".", 1)[-1]
    if op not in RANDOM_FNS:
        return None

    for kw in call.keywords:
        if kw.arg == "size" and isinstance(kw.value, (ast.List, ast.Tuple)):
            return _const_shape(list(kw.value.elts), env)

    args = list(call.args)
    if op == "randint" and len(args) >= 3:
        args = args[2:]
    if len(args) == 1 and isinstance(args[0], (ast.List, ast.Tuple)):
        return _const_shape(list(args[0].elts), env)
    if args and all(isinstance(a, (ast.Constant, ast.Name, ast.BinOp)) for a in args):
        return _const_shape(args, env)
    return None


def infer_input_shapes(model: ModuleModel) -> None:
    """Read ``get_inputs()`` -> list of (shape, dtype)."""
    fn = model.functions.get("get_inputs")
    if fn is None:
        return
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        for elt in node.value.elts:
            call = _innermost_call(elt)
            if call is None:
                model.input_shapes.append(None)  # type: ignore[arg-type]
                continue
            shape = _shape_from_call(call, {})
            dtype = _dtype_of(call)
            model.input_shapes.append((shape, dtype) if shape else None)  # type: ignore[arg-type]
        return


def _innermost_call(node: ast.expr) -> ast.Call | None:
    """``torch.rand([4,4]).cuda()`` -> the ``torch.rand`` call."""
    current = node
    last: ast.Call | None = None
    while True:
        if isinstance(current, ast.Call):
            fn = _dotted(current.func) or ""
            if fn.rsplit(".", 1)[-1] in RANDOM_FNS:
                return current
            last = current
            current = current.func
        elif isinstance(current, ast.Attribute):
            current = current.value
        else:
            return last if isinstance(last, ast.Call) else None


def _nbytes(shape: tuple[int, ...], dtype: str) -> int:
    return math.prod(shape) * ITEMSIZE.get(dtype, 4)


def shapes_from_source(source: str) -> list:
    """Input shapes declared by ``get_inputs()`` in an arbitrary source string."""
    from .parsing import build_skeleton

    probe = build_skeleton(source, "<reference>")
    if probe.tree is None:
        return []
    infer_input_shapes(probe)
    return probe.input_shapes


def reference_input_shapes(level: int, problem_id: int) -> list:
    """Input shapes from the KernelBench reference at ``level{L}/{P}_*.py``.

    Only ~31% of generated kernels keep ``get_inputs()``, but the reference always
    has it -- and the reference is what the kernel was evaluated against, so its
    shapes are the right ones.
    """
    from .runs import reference_source

    source = reference_source(level, problem_id)
    return shapes_from_source(source) if source else []


def infer(model: ModuleModel, fallback_shapes: list | None = None) -> None:
    infer_input_shapes(model)
    if not any(model.input_shapes) and fallback_shapes:
        model.input_shapes = list(fallback_shapes)

    entry = model.entry
    if entry is None or entry not in model.functions:
        return

    # Seed forward's parameters from get_inputs().
    fn = model.functions[entry]
    params = [a.arg for a in fn.args.args if a.arg != "self"]
    for i, param in enumerate(params):
        if i >= len(model.input_shapes):
            break
        info = model.input_shapes[i]
        if not info:
            continue
        shape, dtype = info
        buf = model.buffers.get(model.canonical(scoped(entry, param)))
        if buf is not None:
            buf.shape, buf.dtype, buf.nbytes = shape, dtype, _nbytes(shape, dtype)

    # Propagate through helper calls and allocations until nothing changes.
    for _ in range(4):
        changed = False
        for qual in sorted(model.reachable or set(model.functions)):
            node = model.functions.get(qual)
            if node is not None and _walk_function(model, qual, node):
                changed = True
        if not changed:
            break


def _walk_function(model: ModuleModel, qual: str, node: ast.FunctionDef) -> bool:
    """Bind integer names from ``.shape``, size allocations, propagate into helpers."""
    changed = False
    env: dict[str, int] = {}

    def buf_of(name: str):
        return model.buffers.get(model.canonical(scoped(qual, name)))

    for stmt in ast.walk(node):
        # B, C, H, W = x.shape
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Attribute):
            if stmt.value.attr == "shape":
                src = _base_name(stmt.value.value)
                buf = buf_of(src) if src else None
                target = stmt.targets[0]
                if buf and buf.shape and isinstance(target, (ast.Tuple, ast.List)):
                    for elt, dim in zip(target.elts, buf.shape):
                        if isinstance(elt, ast.Name):
                            env[elt.id] = dim

        # n = x.numel()  /  B = x.size(0)  /  B = x.shape[0]
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                value = stmt.value
                if isinstance(value, ast.Call):
                    fname = _dotted(value.func) or ""
                    op = fname.rsplit(".", 1)[-1]
                    src = _base_name(value.func)
                    buf = buf_of(src) if src else None
                    if buf and buf.shape:
                        if op == "numel":
                            env[target.id] = math.prod(buf.shape)
                        elif op == "size" and value.args:
                            idx = _const_int(value.args[0], env)
                            if idx is not None and idx < len(buf.shape):
                                env[target.id] = buf.shape[idx]
                elif isinstance(value, ast.Subscript) and isinstance(value.value, ast.Attribute):
                    if value.value.attr == "shape":
                        src = _base_name(value.value.value)
                        buf = buf_of(src) if src else None
                        idx = _const_int(value.slice, env) if isinstance(value.slice, (ast.Constant, ast.Name)) else None
                        if buf and buf.shape and idx is not None and idx < len(buf.shape):
                            env[target.id] = buf.shape[idx]

    # Size the allocated buffers now that env is populated.
    for buf in model.buffers.values():
        if not buf.canonical.startswith(f"{qual}::") or buf.shape or buf.alloc_call is None:
            continue
        call = buf.alloc_call
        fn_name = (_dotted(call.func) or "").rsplit(".", 1)[-1]

        if fn_name.endswith("_like") and call.args:
            src = _base_name(call.args[0])
            ref = buf_of(src) if src else None
            if ref and ref.shape:
                buf.shape = ref.shape
                buf.dtype = _dtype_of(call, ref.dtype or "float32")
                buf.nbytes = _nbytes(buf.shape, buf.dtype)
                changed = True
        else:
            shape = _shape_from_call(call, env)
            if shape:
                buf.shape = shape
                buf.dtype = _dtype_of(call, _default_dtype(model))
                buf.nbytes = _nbytes(shape, buf.dtype)
                changed = True

    # forward -> helper(x): give the helper's parameters the caller's shapes, and
    # give `y = helper(x)` the shape of whatever the helper returns.
    for call in model.host_calls:
        if call.enclosing != qual or call.node is None:
            continue
        target_fn = call.qualname
        if target_fn.startswith("self.") and model.model_class:
            target_fn = f"{model.model_class}.{target_fn.split('.', 1)[1]}"
        target = model.functions.get(target_fn)
        if target is None:
            continue

        names = [a.arg for a in target.args.args if a.arg != "self"]
        for i, arg in enumerate(call.node.args):
            if i >= len(names):
                break
            src = _base_name(arg)
            ref = buf_of(src) if src else None
            dst = model.buffers.get(model.canonical(scoped(target_fn, names[i])))
            if ref and ref.shape and dst is not None and dst.shape is None:
                dst.shape, dst.dtype, dst.nbytes = ref.shape, ref.dtype, ref.nbytes
                changed = True

        if call.assigned_to:
            returned = _returned_buffer(model, target_fn)
            dst = model.buffers.get(model.canonical(scoped(qual, call.assigned_to)))
            if returned and returned.shape and dst is not None and dst.shape is None:
                dst.shape, dst.dtype, dst.nbytes = (
                    returned.shape,
                    returned.dtype,
                    returned.nbytes,
                )
                changed = True

    return changed


def _returned_buffer(model: ModuleModel, qual: str):
    prefix = f"{qual}::"
    for buf in model.buffers.values():
        if buf.canonical.startswith(prefix) and buf.returned and buf.shape:
            return buf
    return None


def _default_dtype(model: ModuleModel) -> str:
    for info in model.input_shapes:
        if info:
            return info[1]
    return "float32"
