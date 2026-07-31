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

from ..core.model import KernelDef, ModuleModel, ParamRole

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

    # A scalar offset parameter used *purely additively* -- on an address spine
    # (`tl.store(dst_ptr + dst_offset + offs, v)`, BUG-29) or added onto a grid index in a
    # local (`out_h = h + pad_top`, a padding relocation, BUG-32) -- is never multiplied or
    # compared, so the pass above misses it and it is mistaken for a base pointer. Identify
    # it structurally over *every* additive expression: the leading bare-Name parameter term
    # is the base pointer; any further parameter term is a scalar offset. Because addresses
    # are written `base + offset`, a hoisted base pointer (`inp = inp_ptr + rev_col`) is
    # always the leading term and is protected -- so this never demotes a real pointer and
    # corrupts kernel roles (a reversed `offset + base` spelling can only drop an offset).
    base_params: set[str] = set()
    offset_params: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, (ast.Add, ast.Sub)):
            terms = _additive_terms(sub)
            param_terms = [t.id for t in terms if isinstance(t, ast.Name) and t.id in params]
            if not param_terms:
                continue
            # The base pointer is the *leading term* iff that term is a bare parameter
            # (`base_ptr + offset`); its further parameter terms are offsets. When the
            # leading term is not a parameter (`h + pad_top`: an index plus a shift), there
            # is no base here and every parameter term is a scalar offset.
            first = terms[0]
            if isinstance(first, ast.Name) and first.id in params:
                base_params.add(first.id)
                offset_params.update(param_terms[1:])
            else:
                offset_params.update(param_terms)
    scalars |= offset_params - base_params
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
        # pointer local -> the offset signature carried by the address it was bound to
        # (`inp = x_ptr + rev_col` -> {rev_col}); empty for a pure base. Preserves the
        # offset when a full base+offset address is hoisted into a local (see BUG-22).
        self.ptr_offset: dict[str, frozenset[str]] = {}
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
        # locals derived from tl.arange / tl.program_id -- the grid sweep (see coverage)
        self.index_locals: set[str] = set()
        # locals whose value depends (transitively) on a tl.load -- a data-dependent index
        # (a scatter / one-hot target), even through a `.to()` cast
        self.loaded_taint: set[str] = set()
        # locals whose additive spine carries a scalar-parameter shift or a loaded index,
        # i.e. a relocated / data-dependent address that does not cover the full buffer
        self.relocated: set[str] = set()
        # the kernel has a program_id-guarded early return (`if pid_m < pid_n: return`) --
        # a data-dependent block skip, so no store covers every tile (BUG-36)
        self.conditional_skip = False
        # local -> the expression it was assigned, so a store's `mask=local` resolves to the
        # predicate that defines it (`mask = off_m >= off_n`) -- see _is_conditional_mask
        self.mask_defs: dict[str, ast.expr] = {}

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

        A base-pointer term that is a *hoisted local* (``inp = x_ptr + rev_col``) carries
        its own offset, which is part of the address and must not be stripped away with the
        base -- otherwise the flip/gather offset vanishes and the kernel looks like a memcpy
        (BUG-22). A pure base pointer carries the empty set and contributes nothing.
        """
        sig: set[str] = set()
        for term in _additive_terms(addr):
            if self._is_base_ptr_term(term):
                if isinstance(term, ast.Name):
                    sig |= self.ptr_offset.get(term.id, frozenset())
            else:
                sig.add(ast.dump(term))
        return frozenset(sig)

    def _mark(self, node: ast.expr, *, stored=False, loaded=False, atomic=False) -> None:
        for name in self.pointers(node):
            role = self.kernel.params[name]
            role.stored |= stored
            role.loaded |= loaded
            role.atomic |= atomic

    def _is_index_expr(self, node: ast.expr) -> bool:
        """*node* is part of the grid sweep -- derived from ``tl.arange``/``tl.program_id``
        (an index tensor), as opposed to a scalar size/param or a loaded value."""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and _tl_func(sub) in ("arange", "program_id"):
                return True
        return any(n in self.index_locals for n in _names(node))

    def _has_param_shift(self, value: ast.expr) -> bool:
        """The additive spine of *value* adds a bare kernel parameter onto a grid index
        (``h + pad_top``, ``channel_offset + c``) -- a relocation that shifts the write off
        the full sweep. Only an *additive* bare-parameter term counts: a multiplicative
        stride (``idx * stride``) is not a shift, and a hoisted base pointer term is harmless
        (its local is a load/store base, never a store *offset* term)."""
        terms = _additive_terms(value)
        has_index = any(self._is_index_expr(t) for t in terms)
        has_param = any(
            isinstance(t, ast.Name) and t.id in self.kernel.params for t in terms
        )
        return has_index and has_param

    def _is_data_dependent(self, value: ast.expr) -> bool:
        return _contains_tl_mem(value) or any(n in self.loaded_taint for n in _names(value))

    def _is_conditional_mask(self, mask: ast.expr | None) -> bool:
        """A store mask comparing two grid indices (``off_m >= off_n``) restricts the write
        to a data-dependent triangle -- the rest of the buffer keeps its zeros. A bounds mask
        (``offs < n``: an index vs a scalar size) does not (BUG-36). Resolves a bound mask
        local (``mask = off_m >= off_n``) and looks through ``&``/``|`` combinations."""
        if isinstance(mask, ast.Name) and mask.id in self.mask_defs:
            return self._is_conditional_mask(self.mask_defs[mask.id])
        if isinstance(mask, ast.Compare):
            operands = [mask.left, *mask.comparators]
            return len(operands) >= 2 and all(self._is_index_expr(op) for op in operands)
        if isinstance(mask, ast.BoolOp):
            return any(self._is_conditional_mask(v) for v in mask.values)
        if isinstance(mask, ast.BinOp) and isinstance(mask.op, (ast.BitAnd, ast.BitOr)):
            return self._is_conditional_mask(mask.left) or self._is_conditional_mask(mask.right)
        if isinstance(mask, ast.UnaryOp):
            return self._is_conditional_mask(mask.operand)
        return False

    def _is_full_coverage_store(self, addr: ast.expr, mask: ast.expr | None) -> bool:
        """The store provably writes every element of its buffer: a plain grid sweep with
        no partial-coverage signal. Anything unproven defaults to *not* full -- the sound
        direction, since F2.4's "use empty_like" advice corrupts an unwritten region."""
        offset_terms = [t for t in _additive_terms(addr) if not self._is_base_ptr_term(t)]
        if _is_partial_coverage(offset_terms):
            return False  # repeated index (diagonal) or literal stride >= 2 (BUG-8)
        for term in offset_terms:
            if isinstance(term, ast.Name) and (
                term.id in self.scalar_params  # additive scalar-param shift (concat)
                or term.id in self.relocated   # a local built from such a shift (pad)
            ):
                return False
            if any(n in self.loaded_taint for n in _names(term)):
                return False  # a data-dependent (loaded) scatter / one-hot index
            if _contains_tl_mem(term):
                return False  # the term embeds a load -- a data-dependent address
            if not self._is_index_expr(term):
                return False  # not a grid-derived sweep term (an opaque / constant offset)
        if self._is_conditional_mask(mask):
            return False  # triangular / conditional store mask (BUG-36)
        if self.conditional_skip:
            return False  # a program_id-guarded early return skips whole tiles (BUG-36)
        return True

    def _mark_partial(self, addr: ast.expr, mask: ast.expr | None = None) -> None:
        if self._is_full_coverage_store(addr, mask):
            return
        for name in self.pointers(addr):
            self.kernel.params[name].partial_store = True

    # -- statements -------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        tainted = self.derived(node.value)
        pointed = self.pointers(node.value)
        load_sig = self._load_signature(node.value)
        ptr_off = self._offset_signature(node.value) if pointed else None
        is_index = self._is_index_expr(node.value)
        is_data_dep = self._is_data_dependent(node.value)
        # relocated: the value shifts the write off the full sweep. Either a parameter is
        # added onto a grid index (`h + pad_top`), or it references an already-relocated
        # local anywhere -- the shift survives any arithmetic (`(base + out_c) * stride`) --
        # or a bare scalar/loaded term sits on its additive spine. A multiplicative stride
        # (`idx * BLOCK`) is deliberately *not* a relocation, so only bare additive
        # scalar_params count, never a stride buried in a product.
        is_relocated = (
            self._has_param_shift(node.value)
            or any(n in self.relocated for n in _names(node.value))
            or any(
                isinstance(t, ast.Name) and (t.id in self.scalar_params or t.id in self.loaded_taint)
                for t in _additive_terms(node.value)
            )
        )

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
                    self.ptr_offset[name] = ptr_off or frozenset()
                else:
                    self.ptr_taint.pop(name, None)
                    self.ptr_offset.pop(name, None)
                if load_sig is not None:
                    self.pure_load[name] = load_sig
                else:
                    self.pure_load.pop(name, None)
                if is_index:
                    self.index_locals.add(name)
                else:
                    self.index_locals.discard(name)
                if is_data_dep:
                    self.loaded_taint.add(name)
                else:
                    self.loaded_taint.discard(name)
                if is_relocated:
                    self.relocated.add(name)
                else:
                    self.relocated.discard(name)
                self.mask_defs[name] = node.value

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        target_names = _names(node.target)
        # A pointer advance (`p += BLOCK`) is not a value accumulation.
        if self.loop_depth > 0 and not any(n in self.ptr_taint for n in target_names):
            self.accum_in_loop = True
        for name in target_names:
            self.taint.setdefault(name, set()).update(self.derived(node.value))
            self.pure_load.pop(name, None)

    def visit_If(self, node: ast.If) -> None:
        # A program_id-guarded early return (`if pid_m < pid_n: return`) skips whole tiles,
        # so no store covers the full buffer -- the un-written region keeps its zeros (BUG-36
        # block-skip). A comparison of two grid indices, as opposed to a bounds guard
        # (`if pid >= n_blocks: return`, an index vs a scalar).
        if any(isinstance(s, ast.Return) for s in node.body) and isinstance(node.test, ast.Compare):
            operands = [node.test.left, *node.test.comparators]
            if len(operands) >= 2 and all(self._is_index_expr(op) for op in operands):
                self.conditional_skip = True
        self.generic_visit(node)

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
                mask = None
                for kw in node.keywords:
                    if kw.arg == "mask":
                        mask = kw.value
                if mask is None and len(node.args) >= 3:
                    mask = node.args[2]  # tl.store(ptr, value, mask)
                self._mark_partial(ptr, mask)
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
