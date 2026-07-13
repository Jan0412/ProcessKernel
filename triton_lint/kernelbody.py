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

#: BinOp kinds whose operands are scalar arithmetic, never pointer bases (you never
#: multiply/divide/compare a base pointer). A kernel param seen in one of these is a
#: scalar dimension/stride, not a tensor pointer -- see :func:`_scalar_params`.
_SCALAR_BINOPS = (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult)


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


def _additive_terms(node: ast.expr) -> list[ast.expr]:
    """Flatten the additive spine of an address into its terms.

    ``out_ptr + row * n + col`` -> ``[out_ptr, row * n, col]``. Used to separate the
    base pointer term from the offset terms (for coverage and copy analysis).
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _additive_terms(node.left) + _additive_terms(node.right)
    return [node]


def _has_literal_stride(term: ast.expr) -> bool:
    """An offset term that scales an index by a literal int >= 2 (a strided scatter,
    e.g. ``offs * 2``) -- it skips elements, so the write is not full coverage."""
    if isinstance(term, ast.BinOp) and isinstance(term.op, ast.Mult):
        for side in (term.left, term.right):
            if isinstance(side, ast.Constant) and isinstance(side.value, int) and side.value >= 2:
                return True
    return False


def _is_partial_coverage(offset_terms: list[ast.expr]) -> bool:
    """True when a store's offset does not touch every element of its buffer.

    Two structural tells, both sound (a full row-major store has each index once,
    scaled symbolically):

    * a single index name appears in >= 2 additive terms -- ``i * n + i`` (diagonal),
      ``col * s1 + col * s2`` (a diagonal-covariance write);
    * an index is scaled by a literal int >= 2 -- ``offs * 2`` (stride-2 scatter).
    """
    counts: dict[str, int] = {}
    for term in offset_terms:
        for name in _names(term):
            counts[name] = counts.get(name, 0) + 1
    if any(c >= 2 for c in counts.values()):
        return True
    return any(_has_literal_stride(term) for term in offset_terms)


def _contains_tl_mem(node: ast.AST) -> bool:
    """True if *node* contains a ``tl.load``/``tl.store``/``tl.atomic_*`` call.

    Such an operand is a *loaded value* (or its address); a multiply/compare applied to
    it (``tl.load(x_ptr + o) * 2.0``) operates on the value, so the pointer names inside
    must not be read as scalar operands of that multiply.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = _tl_func(sub)
            if fn in ("load", "store") or (fn is not None and fn.startswith("atomic_")):
                return True
    return False


def _scalar_params(node: ast.FunctionDef, params: dict[str, ParamRole]) -> set[str]:
    """Kernel params that are provably scalars (dimensions / strides), never pointers.

    A base pointer is only ever *added* to an offset; it is never multiplied, divided
    or compared. So a param appearing as an operand of a multiplicative BinOp or a
    comparison is a scalar. Without this, a scalar dimension sitting bare on a store's
    additive spine (``out_ptr + b*(2*H) + H + h``) is mistaken for a stored pointer and
    surfaces as a phantom kernel output.

    Operands that load a value (``tl.load(x_ptr + o) * 2.0``) are skipped -- the pointer
    names live *inside* the load address, which is additive and handled on its own; it
    is the loaded value that is multiplied, not the pointer.
    """
    scalars: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, _SCALAR_BINOPS):
            operands = [sub.left, sub.right]
        elif isinstance(sub, ast.Compare):
            operands = [sub.left, *sub.comparators]
        else:
            continue
        for operand in operands:
            if _contains_tl_mem(operand):
                continue
            for name in _names(operand):
                if name in params:
                    scalars.add(name)
    return scalars


class _BodyVisitor(ast.NodeVisitor):
    """Single pass over a kernel body collecting roles and structural facts."""

    def __init__(self, kernel: KernelDef, scalar_params: set[str]):
        self.kernel = kernel
        self.scalar_params = scalar_params
        # local variable -> set of kernel params it was derived from
        self.taint: dict[str, set[str]] = {}
        # local variable -> params it is a *pointer* into (see _pointer_operands)
        self.ptr_taint: dict[str, set[str]] = {}
        # locals assigned straight from a tl.load with no arithmetic on the value,
        # mapped to the *offset signature* of the load address (see _offset_signature)
        self.pure_load: dict[str, frozenset[str]] = {}
        self.has_dot = False
        self.has_reduce = False
        self.has_math = False
        self.loop_depth = 0
        self.accum_in_loop = False
        # (stored value expr, is it a straight memcpy of a load at the same offset)
        self.stores: list[tuple[ast.expr, bool]] = []

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
            if self._is_pointer_param(name):
                out.add(name)
            out |= self.ptr_taint.get(name, set())
        return out

    def _is_pointer_param(self, name: str) -> bool:
        role = self.kernel.params.get(name)
        return role is not None and not role.is_constexpr and name not in self.scalar_params

    def _is_base_ptr_term(self, term: ast.expr) -> bool:
        """A term of an additive address that is the base pointer, not the offset."""
        if isinstance(term, ast.Name):
            return term.id in self.ptr_taint or self._is_pointer_param(term.id)
        return False

    def _offset_signature(self, addr: ast.expr) -> frozenset[str]:
        """The offset an address applies, with the base pointer stripped.

        Two loads/stores address the same element iff their offset signatures match.
        A straight ``x_ptr + offs`` and ``out_ptr + offs`` share ``{offs}``; a gather
        ``x_ptr + offs*2`` does not match ``out_ptr + offs``.
        """
        return frozenset(
            ast.dump(term)
            for term in _additive_terms(addr)
            if not self._is_base_ptr_term(term)
        )

    def _mark(self, node: ast.expr, *, stored=False, loaded=False, atomic=False) -> None:
        for name in self.pointers(node):
            role = self.kernel.params[name]
            role.stored |= stored
            role.loaded |= loaded
            role.atomic |= atomic

    def _mark_partial(self, addr: ast.expr) -> None:
        offset_terms = [t for t in _additive_terms(addr) if not self._is_base_ptr_term(t)]
        if not _is_partial_coverage(offset_terms):
            return
        for name in self.pointers(addr):
            self.kernel.params[name].partial_store = True

    # -- statements -------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        tainted = self.derived(node.value)
        pointed = self.pointers(node.value)
        load_sig = self._load_signature(node.value)

        # A loop-carried self-reference (`acc = tl.maximum(acc, v)`) is a reduction,
        # just like `acc += v`. A pointer advance (`p = p + BLOCK`, `pointed` non-empty)
        # is address arithmetic, not accumulation -- excluding it keeps tiled matmul
        # loops from being misread as reductions.
        if self.loop_depth > 0 and not pointed:
            rhs_names = _names(node.value)
            if any(n in rhs_names for t in node.targets for n in _names(t)):
                self.accum_in_loop = True

        for target in node.targets:
            for name in _names(target):
                self.taint[name] = set(tainted)
                if pointed:
                    self.ptr_taint[name] = set(pointed)
                else:
                    self.ptr_taint.pop(name, None)
                if load_sig is not None:
                    self.pure_load[name] = load_sig
                else:
                    self.pure_load.pop(name, None)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        target_names = _names(node.target)
        # A pointer advance (`p += BLOCK`) is not a value accumulation.
        if self.loop_depth > 0 and not any(n in self.ptr_taint for n in target_names):
            self.accum_in_loop = True
        for name in target_names:
            self.taint.setdefault(name, set()).update(self.derived(node.value))
            self.pure_load.pop(name, None)

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def _load_signature(self, value: ast.expr) -> frozenset[str] | None:
        """Offset signature if *value* is a load with no arithmetic on the loaded
        value (directly, or through a chain of pure-load names); else ``None``."""
        if isinstance(value, ast.Call) and _tl_func(value) == "load" and value.args:
            return self._offset_signature(value.args[0])
        if isinstance(value, ast.Name):
            return self.pure_load.get(value.id)
        return None

    def _is_matching_copy(self, value: ast.expr, store_sig: frozenset[str]) -> bool:
        """The stored value is a raw load addressed at the *same* offset as the store
        -- i.e. a genuine element-for-element memcpy, not a gather/scatter/select."""
        load_sig = self._load_signature(value)
        return load_sig is not None and load_sig == store_sig

    # -- calls ------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        fn = _tl_func(node)
        if fn is not None and node.args:
            ptr = node.args[0]
            if fn == "load":
                self._mark(ptr, loaded=True)
            elif fn == "store":
                self._mark(ptr, stored=True)
                self._mark_partial(ptr)
                if len(node.args) > 1:
                    value = node.args[1]
                    matching = self._is_matching_copy(value, self._offset_signature(ptr))
                    self.stores.append((value, matching))
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
    # A pure copy performs no arithmetic (has_math would mean a computed branch, e.g. a
    # tl.constexpr dispatch) and every store is a raw load at the *same* offset (a
    # gather/scatter/concat recomputes the address -- that is the task, not a decoy).
    if v.stores and not v.has_math and all(matching for _, matching in v.stores):
        return "copy"
    if not v.stores:
        return "unknown"
    return "elementwise"


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

    scalars = _scalar_params(kernel.node, kernel.params)
    visitor = _BodyVisitor(kernel, scalars)
    for stmt in kernel.node.body:
        visitor.visit(stmt)

    kernel.kind = _classify(kernel, visitor)  # type: ignore[assignment]


def analyze_kernels(model: ModuleModel) -> None:
    for kernel in model.kernels.values():
        analyze_kernel(kernel)

    # Kernel -> kernel call edges: a @triton.jit *device function* is invoked by name
    # inside another kernel's body (`v * rsqrt(x)`) and inlined by Triton, never
    # [grid]-launched. F1.2 uses these edges so an inlined helper is not "dead".
    kernel_names = set(model.kernels)
    for kernel in model.kernels.values():
        for node in ast.walk(kernel.node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                callee = node.func.id
                if callee in kernel_names and callee != kernel.name:
                    kernel.calls.add(callee)
