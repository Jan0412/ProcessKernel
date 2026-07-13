"""Per-``@triton.jit`` analysis: what each parameter is used for, and what kind
of computation the kernel performs.

Roles are recovered by taint propagation *inside* the kernel: a parameter is an
output if it reaches the pointer argument of ``tl.store``, an input if it reaches
``tl.load``.  Pointer arithmetic through locals is followed, so

    row = x_ptr + pid * stride
    v = tl.load(row + offs)

still marks ``x_ptr`` as loaded.
"""

from __future__ import annotations

import ast

from .model import KernelDef, ModuleModel, ParamRole

REDUCE_FNS = {"sum", "max", "min", "argmax", "argmin", "reduce", "xor_sum"}
MATH_FNS = {
    "exp", "log", "sqrt", "sin", "cos", "sigmoid", "abs", "maximum", "minimum",
    "where", "fma", "erf", "tanh", "rsqrt", "log2", "exp2", "cdiv", "softmax",
}


def _tl_func(call: ast.Call) -> str | None:
    """``tl.load`` / ``triton.language.store`` -> ``"load"`` / ``"store"``."""
    fn = call.func
    if isinstance(fn, ast.Attribute):
        base = fn.value
        if isinstance(base, ast.Name) and base.id in ("tl", "triton"):
            return fn.attr
        if isinstance(base, ast.Attribute) and base.attr == "language":
            return fn.attr
    return None


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _pointer_operands(node: ast.expr) -> set[str]:
    """Names on the additive spine of an address expression.

    ``x_ptr + base + cur_c * (H * W)`` -> ``{x_ptr, base}``. Anything reached only
    through a multiply, a call or a subscript is offset arithmetic, not a pointer.
    """
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _pointer_operands(node.left) | _pointer_operands(node.right)
    return set()


class _BodyVisitor(ast.NodeVisitor):
    """Single pass over a kernel body collecting roles and structural facts."""

    def __init__(self, kernel: KernelDef):
        self.kernel = kernel
        # local variable -> set of kernel params it was derived from
        self.taint: dict[str, set[str]] = {}
        # local variable -> params it is a *pointer* into (see _pointer_operands)
        self.ptr_taint: dict[str, set[str]] = {}
        # locals assigned straight from a tl.load with no arithmetic applied
        self.pure_load: set[str] = set()
        self.has_dot = False
        self.has_reduce = False
        self.has_math = False
        self.loop_depth = 0
        self.accum_in_loop = False
        self.store_values: list[ast.expr] = []

    # -- taint ------------------------------------------------------------

    def derived(self, node: ast.AST) -> set[str]:
        """Kernel params that *node* is (transitively) derived from."""
        out: set[str] = set()
        for name in _names(node):
            if name in self.kernel.params:
                out.add(name)
            out |= self.taint.get(name, set())
        return out

    def pointers(self, node: ast.expr) -> set[str]:
        """Params that *node* is a pointer INTO, following only the additive spine.

        The address expression of a load is ``base_ptr + offset``. Names reached
        through multiplication or a call are offset arithmetic, not pointers -- in

            tl.load(x_ptr + base + cur_c * (H * W), ...)

        only ``x_ptr`` is a pointer; ``H`` and ``W`` are shape scalars. Taking every
        name in the expression (the naive approach) marks H and W as tensor
        parameters, which then invents phantom buffers on the host side.
        """
        out: set[str] = set()
        for name in _pointer_operands(node):
            if name in self.kernel.params and not self.kernel.params[name].is_constexpr:
                out.add(name)
            out |= self.ptr_taint.get(name, set())
        return out

    def _mark(self, node: ast.expr, *, stored=False, loaded=False, atomic=False) -> None:
        for name in self.pointers(node):
            role = self.kernel.params[name]
            role.stored |= stored
            role.loaded |= loaded
            role.atomic |= atomic

    # -- statements -------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        tainted = self.derived(node.value)
        pointed = self.pointers(node.value)
        is_pure_load = self._is_pure_load(node.value)
        for target in node.targets:
            for name in _names(target):
                self.taint[name] = set(tainted)
                if pointed:
                    self.ptr_taint[name] = set(pointed)
                else:
                    self.ptr_taint.pop(name, None)
                if is_pure_load:
                    self.pure_load.add(name)
                else:
                    self.pure_load.discard(name)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if self.loop_depth > 0:
            self.accum_in_loop = True
        for name in _names(node.target):
            self.taint.setdefault(name, set()).update(self.derived(node.value))
            self.pure_load.discard(name)

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def _is_pure_load(self, value: ast.expr) -> bool:
        if isinstance(value, ast.Call) and _tl_func(value) == "load":
            return True
        if isinstance(value, ast.Name):
            return value.id in self.pure_load
        return False

    # -- calls ------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        fn = _tl_func(node)
        if fn is not None and node.args:
            ptr = node.args[0]
            if fn == "load":
                self._mark(ptr, loaded=True)
            elif fn == "store":
                self._mark(ptr, stored=True)
                if len(node.args) > 1:
                    self.store_values.append(node.args[1])
            elif fn.startswith("atomic_"):
                # An atomic both reads and writes; zero-init is required.
                self._mark(ptr, stored=True, loaded=True, atomic=True)

            if fn == "dot":
                self.has_dot = True
            elif fn in REDUCE_FNS:
                self.has_reduce = True
            elif fn in MATH_FNS:
                self.has_math = True

        self.generic_visit(node)


def _classify(kernel: KernelDef, v: _BodyVisitor) -> str:
    if v.has_dot:
        return "matmul"
    # An accumulator updated inside a loop is a reduction even without tl.sum.
    if v.has_reduce or v.accum_in_loop:
        return "reduction"
    if v.store_values and all(_stores_raw_load(val, v) for val in v.store_values):
        return "copy"
    if not v.store_values:
        return "unknown"
    return "elementwise"


def _stores_raw_load(value: ast.expr, v: _BodyVisitor) -> bool:
    """The stored value is a loaded value with no arithmetic applied to it."""
    if isinstance(value, ast.Call) and _tl_func(value) == "load":
        return True
    if isinstance(value, ast.Name):
        return value.id in v.pure_load
    return False


def analyze_kernel(kernel: KernelDef) -> None:
    args = kernel.node.args
    positional = list(args.posonlyargs) + list(args.args)
    for i, arg in enumerate(positional + list(args.kwonlyargs)):
        annotation = ast.unparse(arg.annotation) if arg.annotation else ""
        kernel.params[arg.arg] = ParamRole(
            name=arg.arg,
            index=i,
            is_constexpr="constexpr" in annotation,
        )
    kernel.param_order = [a.arg for a in positional] + [a.arg for a in args.kwonlyargs]

    visitor = _BodyVisitor(kernel)
    for stmt in kernel.node.body:
        visitor.visit(stmt)

    kernel.kind = _classify(kernel, visitor)  # type: ignore[assignment]


def analyze_kernels(model: ModuleModel) -> None:
    for kernel in model.kernels.values():
        analyze_kernel(kernel)
