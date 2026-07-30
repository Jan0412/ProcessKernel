"""Host-side (non-kernel) analysis.

Recovers, for every function in the module:

* launch sites -- ``kernel[grid](...)`` is a ``Call`` whose ``func`` is a
  ``Subscript`` -- along with the Python loop nesting they sit in and a map from
  kernel parameter name to the host expression passed for it;
* reachability from the entry point, following calls through helper functions to
  a fixpoint (a kernel launched from a helper two hops away is still reachable) --
  computed twice with opposite approximations: ``reachable`` (conservative, for
  "is this kernel ever used" checks) and ``timed_scopes`` (precise, for "what
  does the timed forward do" checks);
* buffers and their aliases (``view``/``permute``/... produce a second name for
  the same memory, so they must be unioned before asking "is this intermediate
  read by anyone else?");
* which buffers flow into the return value, and which are read by host code.

Buffer names are scoped as ``"<function qualname>::<var>"``: an intermediate
allocated and consumed inside a helper is a local fact about that helper.
"""

from __future__ import annotations

import ast
from collections.abc import Callable

from ..core.model import Buffer, HostCall, LaunchSite, ModuleModel
from ..core.parsing import _dotted, is_triton_kernel

#: Namespaces whose attribute calls are free functions combining their arguments
#: (``torch.cat([a, b])`` forwards ``a`` and ``b``), as opposed to tensor methods
#: (``out.view(...)`` forwards its receiver).
MODULE_NAMESPACES = {"torch", "F", "nn", "np", "numpy", "tl", "triton"}

ALLOC_FNS = {
    "empty", "empty_like", "empty_strided",
    "zeros", "zeros_like",
    "ones", "ones_like",
    "full", "full_like",
    "rand", "randn", "randn_like", "rand_like",
}

#: Methods that return a view -- same storage, no kernel launched.
ALIAS_METHODS = {
    "view", "reshape", "permute", "transpose", "squeeze", "unsqueeze",
    "expand", "t", "detach", "as_strided", "flatten", "ravel", "narrow",
}

#: Alias methods that make the result non-contiguous (so a later ``.contiguous()``
#: on it really does launch a copy kernel).
LAYOUT_METHODS = {"permute", "transpose", "t", "expand", "as_strided"}

#: Allocators that read only a tensor's *metadata* (shape/dtype/device), not its data,
#: from their first argument -- ``torch.empty_like(t)`` is not a host read of ``t``.
METADATA_LIKE = {
    "empty_like", "zeros_like", "ones_like", "full_like",
    "rand_like", "randn_like", "randint_like",
}
#: The ``tensor.new_*`` family reads metadata from its *receiver* instead.
METADATA_NEW = {
    "new_empty", "new_zeros", "new_ones", "new_full", "new_tensor", "new_empty_strided",
}

#: ``nn.*`` names that are not callable compute modules -- tensor holders that launch
#: nothing and cannot be invoked, so they are never a fallback (BUG-24: ``nn.Parameter``).
NON_MODULE_NN = {
    "Parameter", "ParameterList", "ParameterDict",
    "UninitializedParameter", "UninitializedBuffer",
}

#: ``nn.*`` containers: they own no compute of their own, so what matters is their
#: contents -- a container of the file's own Triton modules is not a fallback (BUG-30).
NN_CONTAINERS = {"Sequential", "ModuleList", "ModuleDict"}


def _is_nn_module_ctor(dotted: str) -> bool:
    """``nn.Conv2d`` / ``torch.nn.Sequential`` -> True; ``nn.Parameter`` -> False."""
    if not (dotted.startswith("nn.") or dotted.startswith("torch.nn.")):
        return False
    return dotted.rsplit(".", 1)[-1] not in NON_MODULE_NN


def _builds_compute_nn_module(node: ast.expr) -> bool:
    """A construction that builds a *non-container* ``nn.*`` compute module somewhere
    inside (``nn.Sequential(TritonReLU(), nn.Linear(...))``) -- so it is a genuine fallback
    even when it also wraps a local class (BUG-30's mixed case must keep firing)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            dotted = _dotted(sub.func) or ""
            if _is_nn_module_ctor(dotted) and dotted.rsplit(".", 1)[-1] not in NN_CONTAINERS:
                return True
    return False


def scoped(func: str, name: str) -> str:
    return f"{func}::{name}"


class _FuncVisitor(ast.NodeVisitor):
    """Walks one host function, recording launches, calls, aliases and buffers."""

    def __init__(
        self,
        model: ModuleModel,
        qualname: str,
        kernels: set[str],
        param_default_classes: dict[str, list[str]] | None = None,
    ):
        self.model = model
        self.qual = qualname
        self.kernels = kernels
        # param name -> local classes named in its default value (`act_layer=GELU`)
        self.param_default_classes = param_default_classes or {}
        self.loop_stack: list[str] = []
        # Name-load bookkeeping used to decide `read_by_host`.
        self.loads: dict[str, int] = {}
        self.launch_loads: dict[str, int] = {}
        self.return_loads: dict[str, int] = {}
        # Passing a tensor into a helper that launches a kernel is plumbing, not a
        # host read -- otherwise every interprocedural intermediate looks "used".
        self.helper_loads: dict[str, int] = {}
        # A tensor read only for its shape/dtype (`torch.empty_like(t)`) is not a data
        # read either.
        self.meta_loads: dict[str, int] = {}
        self.referenced: set[str] = set()
        self.assign_target: dict[int, str] = {}  # id(Call node) -> assigned name

    # -- helpers ----------------------------------------------------------

    def buf(self, name: str) -> Buffer:
        key = scoped(self.qual, name)
        canonical = self.model.canonical(key)
        if canonical not in self.model.buffers:
            self.model.buffers[canonical] = Buffer(canonical=canonical)
        return self.model.buffers[canonical]

    def _count(self, node: ast.AST, sink: dict[str, int]) -> None:
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                sink[n.id] = sink.get(n.id, 0) + 1

    # -- loops ------------------------------------------------------------

    def visit_For(self, node: ast.For) -> None:
        var = ast.unparse(node.target) if node.target else "?"
        self.loop_stack.append(var)
        start = len(self.model.launches)
        self.generic_visit(node)
        self._detect_recurrence(node, start)
        self.loop_stack.pop()

    def visit_While(self, node: ast.While) -> None:
        self.loop_stack.append("while")
        start = len(self.model.launches)
        self.generic_visit(node)
        self._detect_recurrence(node, start)
        self.loop_stack.pop()

    def _detect_recurrence(self, loop: ast.AST, start: int) -> None:
        """Flag launches carrying a data dependency through the loop.

        A sequential recurrence rebinds one of the kernel's *input* tensors to its
        *output* each iteration (``h = h_new``, where ``h`` is read and ``h_new`` is
        written by the launch). That loop dimension cannot be moved into the launch
        grid -- see F2.2. A parallel batch loop (``for i: k(a[i], out[i])``) has no
        such reassignment.
        """
        pairs: list[tuple[str, str]] = []
        for sub in ast.walk(loop):
            if (
                isinstance(sub, ast.Assign)
                and len(sub.targets) == 1
                and isinstance(sub.targets[0], ast.Name)
            ):
                src = _base_name(sub.value)
                if src is not None:
                    pairs.append((sub.targets[0].id, src))
        for launch in self.model.launches[start:]:
            kernel = self.model.kernels.get(launch.kernel_name)
            if kernel is None:
                continue
            inputs: set[str] = set()
            outputs: set[str] = set()
            in_place = False
            for param, expr in launch.arg_map.items():
                role = kernel.params.get(param)
                base = _base_name(expr)
                if role is None or base is None:
                    continue
                # A buffer both loaded and stored by the *same* launch, passed unchanged
                # across the loop, carries state in place -- a recurrence with no host
                # `h = h_new` rebind to detect (BUG-33). Gridding it races on the shared
                # buffer, so the fix is to move the loop into the kernel, not into the grid.
                if role.stored and role.loaded:
                    in_place = True
                if role.stored:
                    outputs.add(base)
                elif role.loaded:
                    inputs.add(base)
            if in_place or any(t in inputs and s in outputs for t, s in pairs):
                launch.recurrence = True

    # -- statements -------------------------------------------------------

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            self._count(node.value, self.return_loads)
            for name in _returned_names(node.value):
                self.buf(name).returned = True
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        targets = [t for t in node.targets if isinstance(t, ast.Name)]

        # self.foo = <expr>  in __init__
        if self.qual.endswith(".__init__"):
            for t in node.targets:
                if not (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    continue

                if isinstance(node.value, ast.Call):
                    cls = _dotted(node.value.func) or ""
                    if _is_nn_module_ctor(cls):
                        # Keyed by owning class ("ModelNew.__init__" -> "ModelNew") so a
                        # binding in one class cannot decide how a same-named attribute is
                        # judged in another (BUG-24).
                        owner = self.qual.rsplit(".", 1)[0]
                        self.model.nn_modules_in_init.setdefault(owner, {})[t.attr] = cls

                # self.layernorm = LayerNormTriton(...) -- a submodule defined in this
                # file. Its forward() is where the kernel is actually launched, so
                # reachability must follow it. Also catches nn.Sequential(Mish(), ...)
                # and `self.act = act_layer()` where act_layer defaults to a local class.
                local = [
                    n.id
                    for n in ast.walk(node.value)
                    if isinstance(n, ast.Name) and n.id in self.model.local_classes
                ]
                for n in ast.walk(node.value):
                    if isinstance(n, ast.Name) and n.id in self.param_default_classes:
                        local.extend(self.param_default_classes[n.id])
                if local:
                    self.model.attr_classes.setdefault(t.attr, []).extend(local)
                    if _builds_compute_nn_module(node.value):
                        self.model.containers_with_torch.add(t.attr)

        if len(targets) == 1:
            name = targets[0].id
            self._record_binding(name, node.value)
            if isinstance(node.value, ast.Call):
                self.assign_target[id(node.value)] = name

        self.generic_visit(node)

        # A Subscript/Attribute *target* (`out[mask] = v`) writes those elements; the
        # base Name carries Load ctx in the AST but is not a host read. Undo the
        # visit_Name count for it (AugAssign targets, which genuinely read, are handled
        # in visit_AugAssign and deliberately not undone here).
        for t in node.targets:
            if not isinstance(t, ast.Name):
                base = _base_name(t)
                if base is not None and self.loads.get(base):
                    self.loads[base] -= 1

        # A plain rebind `y = x` aliases x; the source Name is plumbing (tracked as an
        # alias), not a data read -- so it must not count toward read_by_host.
        if len(node.targets) == 1 and isinstance(node.value, ast.Name):
            if self.loads.get(node.value.id):
                self.loads[node.value.id] -= 1

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # `out += x` / `out[i] += x` reads `out` before writing it; the target Name
        # has Store ctx so visit_Name misses it. Count the base as a host read.
        base = _base_name(node.target)
        if base is not None:
            self.loads[base] = self.loads.get(base, 0) + 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # A nested @triton.jit kernel is not host code -- its tl.load/tl.store/tl.dot
        # must not leak into the enclosing function's call scan.
        if is_triton_kernel(node):
            return
        self.generic_visit(node)

    def _record_binding(self, name: str, value: ast.expr) -> None:
        key = scoped(self.qual, name)

        # Rebinding kills the stale non-contiguous flag (it is a monotonic taint set;
        # without a kill, `x = x.permute(...)` then `x = torch.relu(x)` would keep `x`
        # flagged and a later no-op `x.contiguous()` would be reported as a copy). The
        # alias map is deliberately *not* cleared here: an intermediate that is aliased
        # and then transformed (`loss = full_loss; loss = loss * r`) must keep resolving
        # to its storage for interprocedural propagation.
        self.model.noncontiguous.discard(key)

        if isinstance(value, ast.Call):
            # raw pointer: p = out.data_ptr() / p = out[i].data_ptr(). A common way to hand
            # Triton a (per-slice) base address; the pointer local must alias its receiver's
            # storage or the launched output reads as a phantom buffer (BUG-35). Handled
            # before the dotted-name check: a subscript receiver (out[i]) makes _dotted give
            # up, and `_base_name` walks the Subscript so out[i].data_ptr() resolves to `out`.
            if isinstance(value.func, ast.Attribute) and value.func.attr == "data_ptr":
                src = _base_name(value.func.value)
                if src is not None:
                    self.model.aliases[key] = scoped(self.qual, src)
                    return

            fn = _dotted(value.func)
            if fn:
                op = fn.rsplit(".", 1)[-1]
                base = fn.rsplit(".", 1)[0] if "." in fn else ""

                # allocation: torch.empty_like(x), torch.zeros(...)
                if op in ALLOC_FNS and base in ("torch", ""):
                    buf = self.buf(name)
                    buf.alloc_fn = op
                    buf.alloc_lineno = value.lineno
                    buf.alloc_call = value
                    return

                # alias: y = x.view(...)  /  y = x.permute(...).contiguous()
                if op in ALIAS_METHODS and isinstance(value.func, ast.Attribute):
                    src = _base_name(value.func.value)
                    if src is not None:
                        self.model.aliases[key] = scoped(self.qual, src)
                        if op in LAYOUT_METHODS or scoped(self.qual, src) in self.model.noncontiguous:
                            self.model.noncontiguous.add(key)
                        return

        # y = x  (plain rebinding)
        if isinstance(value, ast.Name):
            self.model.aliases[key] = scoped(self.qual, value.id)
            if scoped(self.qual, value.id) in self.model.noncontiguous:
                self.model.noncontiguous.add(key)

        # subscript view: c = out[b] / out[b, 0] / out[b:b+1]. Writing a slice writes into
        # the parent's storage, so the per-slice launch idiom must alias to `out` -- else its
        # store target is a fresh buffer nobody reads and F1.3 calls the output discarded
        # (BUG-23). `_base_name` walks the Subscript to its leftmost Name.
        if isinstance(value, ast.Subscript):
            src = _base_name(value)
            if src is not None and src != name:
                self.model.aliases[key] = scoped(self.qual, src)

    # -- calls ------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        # Launch: kernel[grid](args...)
        if isinstance(node.func, ast.Subscript):
            name = _base_name(node.func.value)
            if name is not None and name in self.kernels:
                self._record_launch(name, node)
                self.generic_visit(node)
                return

        fn = _dotted(node.func)
        if fn is None and isinstance(node.func, ast.Attribute):
            # Chained method call: `x.permute(1, 0).contiguous()`. The receiver is a
            # Call, so the dotted-name walk gives up -- but this is exactly the layout
            # churn F2.3 is looking for, so name it after its leftmost tensor.
            base = _base_name(node.func.value)
            fn = f"{base}.{node.func.attr}" if base else node.func.attr

        if fn:
            receiver = None
            is_method = isinstance(node.func, ast.Attribute)
            if is_method:
                base = _base_name(node.func.value)
                if base is not None:
                    receiver = self.model.canonical(scoped(self.qual, base))
            self.model.host_calls.append(
                HostCall(
                    qualname=fn,
                    enclosing=self.qual,
                    lineno=node.lineno,
                    is_method=is_method,
                    receiver=receiver,
                    node=node,
                    assigned_to=self.assign_target.get(id(node)),
                )
            )
            self.referenced.add(fn.split(".")[0])
            self.referenced.add(fn)

            # Args handed to a module-level helper are plumbing, not host reads.
            if fn in self.model.functions:
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    self._count(arg, self.helper_loads)

            # Metadata-only allocators read shape/dtype, not data.
            op = fn.rsplit(".", 1)[-1]
            if op in METADATA_LIKE and node.args:
                self._count(node.args[0], self.meta_loads)
            elif op in METADATA_NEW and isinstance(node.func, ast.Attribute):
                self._count(node.func.value, self.meta_loads)

        # Host reads are counted comprehensively in visit_Name (every Load-context
        # Name), so a subscript/slice read is seen and a launch arg is counted with the
        # same multiplicity in `loads` and `launch_loads` -- they cancel exactly.
        self.generic_visit(node)

    def _record_launch(self, kernel_name: str, node: ast.Call) -> None:
        kernel = self.model.kernels[kernel_name]
        index = len(self.model.launches)

        arg_map: dict[str, ast.expr] = {}
        for i, arg in enumerate(node.args):
            if i < len(kernel.param_order):
                arg_map[kernel.param_order[i]] = arg
        for kw in node.keywords:
            if kw.arg:
                arg_map[kw.arg] = kw.value

        launch = LaunchSite(
            index=index,
            kernel_name=kernel_name,
            enclosing=self.qual,
            lineno=node.lineno,
            loop_depth=len(self.loop_stack),
            loop_vars=list(self.loop_stack),
            arg_map=arg_map,
        )
        self.model.launches.append(launch)
        self.referenced.add(kernel_name)

        # Wire host buffers to the kernel's parameter roles.
        for param, expr in arg_map.items():
            role = kernel.params.get(param)
            if role is None or not role.is_pointer:
                continue
            var = _base_name(expr)
            if var is None:
                continue
            buf = self.buf(var)
            if role.stored and index not in buf.stored_by:
                buf.stored_by.append(index)
            if role.loaded and index not in buf.loaded_by:
                buf.loaded_by.append(index)

        # Names inside launch args are not "host reads" of the tensor.
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            self._count(arg, self.launch_loads)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.referenced.add(node.id)
            self.loads[node.id] = self.loads.get(node.id, 0) + 1
        self.generic_visit(node)


def _returned_names(node: ast.expr) -> set[str]:
    """Variables whose VALUE flows out of a ``return``.

    Not simply "every Name in the return expression": in ``return do_scale(tmp)``,
    ``tmp`` is an *argument* -- it is consumed by the helper, not returned. Treating it
    as returned would hide the very intermediate F2.1 exists to find. But in
    ``return out.view(-1)`` or ``return out * 2``, ``out`` really does flow out.
    """
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_returned_names(e) for e in node.elts)) if node.elts else set()
    if isinstance(node, ast.BinOp):
        return _returned_names(node.left) | _returned_names(node.right)
    # An output returned through an operator node still flows out and is consumed on the
    # host -- a ternary loss (`out.mean() if size_average else out.sum()`), a negated loss
    # (`-out.sum()`), a boolean (`out.sum() > 0`). Recurse every operator, not just BinOp,
    # or the buffer carries no use flag and F1.3 calls it discarded (BUG-34).
    if isinstance(node, ast.IfExp):
        return _returned_names(node.body) | _returned_names(node.orelse)
    if isinstance(node, ast.UnaryOp):
        return _returned_names(node.operand)
    if isinstance(node, ast.Compare):
        return set().union(_returned_names(node.left), *(_returned_names(c) for c in node.comparators))
    if isinstance(node, ast.BoolOp):
        return set().union(*(_returned_names(v) for v in node.values)) if node.values else set()
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _returned_names(node.value)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            # A free function in a module namespace (`torch.cat([a, b])`) combines its
            # arguments into the result, so they flow out; a tensor method
            # (`out.view(...)`) returns a view of its receiver.
            if _base_name(node.func.value) in MODULE_NAMESPACES:
                out: set[str] = set()
                for arg in node.args:
                    out |= _returned_names(arg)
                for kw in node.keywords:
                    out |= _returned_names(kw.value)
                return out
            return _returned_names(node.func.value)
        # A plain function call (`helper(tmp)`) returns something new.
        return set()
    return set()


def _base_name(node: ast.expr) -> str | None:
    """Leftmost ``Name`` of an expression: ``x.permute(0,1).contiguous()`` -> ``x``."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            node = node.value
        else:
            return None


def analyze_host(model: ModuleModel) -> None:
    kernels = set(model.kernels)
    visitors: dict[str, _FuncVisitor] = {}

    for qual, node in model.functions.items():
        pdc = _param_default_classes(node, model.local_classes)
        v = _FuncVisitor(model, qual, kernels, pdc)
        for stmt in node.body:
            v.visit(stmt)
        # Default argument values are host code too: `def __init__(self,
        # act_layer=GELU_Triton)` references GELU_Triton, whose kernel really runs.
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d]:
            v.visit(default)
        visitors[qual] = v

        # Parameters are buffers too (an in-place kernel writes into one).
        is_entry_like = qual.endswith(".forward")
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            buf = v.buf(arg.arg)
            buf.is_forward_input = is_entry_like

    _record_module_level(model)
    # Interprocedural propagation creates the buffers for `y = helper(x)` results, so
    # it must run before _resolve_reads records their host-read flags -- otherwise a
    # later host read of a helper-produced tensor is checked against a not-yet-existing
    # buffer and silently dropped.
    _propagate_interprocedural(model)
    _resolve_reads(model, visitors)
    # Push the caller's use flags *down* to helper parameters -- runs after _resolve_reads
    # because read_by_host is only known then.
    _propagate_uses_to_params(model)
    _resolve_reachability(model, visitors)


def _resolve_target_fn(model: ModuleModel, call: HostCall) -> str | None:
    """The in-file function a host call targets (resolving ``self.m`` against the entry
    class), or ``None`` if it is not a local function or has no AST node."""
    target_fn = call.qualname
    if target_fn.startswith("self.") and model.model_class:
        target_fn = f"{model.model_class}.{target_fn.split('.', 1)[1]}"
    if target_fn not in model.functions or call.node is None:
        return None
    return target_fn


def _propagate_uses_to_params(model: ModuleModel) -> None:
    """Push a caller's ``returned`` / ``read_by_host`` flags onto the helper *parameter*
    buffers it passes them to.

    F1.3 resolves a launch's store target in the launch's *enclosing* scope. When the
    launch sits in a helper writing one of the helper's own parameters, that parameter
    buffer never inherits the caller's use flags -- ``_propagate_interprocedural`` pushes
    the store role *up* to the caller but nothing pushes the caller's use *down* -- so the
    output reads as discarded though the caller returns or reads it (BUG-28). Iterated to a
    fixpoint so the flags reach through ``forward -> outer -> inner`` chains.
    """
    for _ in range(4):
        changed = False
        for call in model.host_calls:
            target_fn = _resolve_target_fn(model, call)
            if target_fn is None:
                continue
            names = [a.arg for a in model.functions[target_fn].args.args if a.arg != "self"]
            for i, arg in enumerate(call.node.args):
                if i >= len(names):
                    break
                var = _base_name(arg)
                if var is None:
                    continue
                caller = model.buffers.get(model.canonical(scoped(call.enclosing, var)))
                param = model.buffers.get(model.canonical(scoped(target_fn, names[i])))
                if caller is None or param is None:
                    continue
                if caller.returned and not param.returned:
                    param.returned = True
                    changed = True
                if caller.read_by_host and not param.read_by_host:
                    param.read_by_host = True
                    changed = True
        if not changed:
            break


def _param_default_classes(
    node: ast.FunctionDef, local_classes: dict[str, list[str]]
) -> dict[str, list[str]]:
    """param name -> local classes named in its default value.

    ``def __init__(self, act_layer=GELU_Triton)`` -> ``{"act_layer": ["GELU_Triton"]}``,
    so a later ``self.act = act_layer()`` can be wired to the class it constructs.
    """
    result: dict[str, list[str]] = {}
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)

    def classes_in(default: ast.expr) -> list[str]:
        return [
            n.id
            for n in ast.walk(default)
            if isinstance(n, ast.Name) and n.id in local_classes
        ]

    for arg, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
        found = classes_in(default)
        if found:
            result[arg.arg] = found
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None and (found := classes_in(default)):
            result[arg.arg] = found
    return result


def _record_module_level(model: ModuleModel) -> None:
    """Record calls that live outside any function body -- module-level statements
    (``scripted = torch.jit.script(plain)``) and decorators (``@torch.compile``).

    These run at import/definition time and route work away from Triton, so F1.7 must
    see them. They are scoped to a synthetic ``"<module>"`` enclosing name, which is in
    no check's timed/reachable scope set, so only F1.7 (which scans every host call)
    picks them up.
    """
    tree = model.tree
    if tree is None:
        return

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for n in ast.walk(stmt):
            if isinstance(n, ast.Call):
                fn = _dotted(n.func)
                if fn:
                    model.host_calls.append(
                        HostCall(qualname=fn, enclosing="<module>", lineno=n.lineno, node=n)
                    )

    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in n.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                fn = _dotted(target)
                if fn:
                    model.host_calls.append(
                        HostCall(
                            qualname=fn,
                            enclosing="<module>",
                            lineno=getattr(dec, "lineno", n.lineno),
                            node=dec if isinstance(dec, ast.Call) else None,
                        )
                    )


def _resolve_reads(model: ModuleModel, visitors: dict[str, _FuncVisitor]) -> None:
    """A buffer is "read by host" if it is loaded somewhere other than a launch
    argument, a helper-call argument, or the return statement."""
    for qual, v in visitors.items():
        for name, total in v.loads.items():
            elsewhere = (
                total
                - v.launch_loads.get(name, 0)
                - v.return_loads.get(name, 0)
                - v.helper_loads.get(name, 0)
                - v.meta_loads.get(name, 0)
            )
            if elsewhere > 0:
                key = model.canonical(scoped(qual, name))
                if key in model.buffers:
                    model.buffers[key].read_by_host = True


def _propagate_interprocedural(model: ModuleModel) -> None:
    """Push kernel store/load roles across function boundaries.

    The dominant real-world shape is one helper per kernel::

        def triton_norm(x):            out = torch.empty(...)
                                       norm_kernel[grid](x, out, ...)
                                       return out

        def forward(self, x):          norm = triton_norm(x)          # <- producer
                                       return triton_scale_bias(x, norm, ...)  # consumer

    Purely intra-procedural analysis sees two unrelated helpers and never notices
    that ``norm`` is an intermediate round-tripping through HBM. So: a variable bound
    to a helper's return value inherits the launches that wrote what the helper
    returned, and a tensor passed into a helper inherits the launches that read the
    corresponding parameter. Iterated to a fixpoint for helpers calling helpers.
    """
    for _ in range(4):
        changed = False

        # What each function does with its params, and what its return value carries.
        returns: dict[str, list[int]] = {}
        param_stored: dict[str, dict[str, list[int]]] = {}
        param_loaded: dict[str, dict[str, list[int]]] = {}

        for qual, node in model.functions.items():
            prefix = f"{qual}::"
            ret: list[int] = []
            for buf in model.buffers.values():
                if buf.canonical.startswith(prefix) and buf.returned:
                    ret.extend(buf.stored_by)
            returns[qual] = ret

            names = [a.arg for a in node.args.args if a.arg != "self"]
            param_stored[qual] = {}
            param_loaded[qual] = {}
            for name in names:
                buf = model.buffers.get(model.canonical(scoped(qual, name)))
                if buf is not None:
                    param_stored[qual][name] = list(buf.stored_by)
                    param_loaded[qual][name] = list(buf.loaded_by)

        for call in model.host_calls:
            target_fn = call.qualname
            if target_fn.startswith("self.") and model.model_class:
                target_fn = f"{model.model_class}.{target_fn.split('.', 1)[1]}"
            if target_fn not in model.functions or call.node is None:
                continue

            # y = helper(...): y inherits the launches that produced helper's result.
            if call.assigned_to:
                buf = model.buffers.setdefault(
                    model.canonical(scoped(call.enclosing, call.assigned_to)),
                    Buffer(canonical=model.canonical(scoped(call.enclosing, call.assigned_to))),
                )
                for index in returns.get(target_fn, []):
                    if index not in buf.stored_by:
                        buf.stored_by.append(index)
                        # The helper allocated it on our behalf; record that so the
                        # intermediate is not mistaken for an unowned tensor.
                        buf.alloc_fn = buf.alloc_fn or f"{target_fn}()"
                        buf.alloc_lineno = buf.alloc_lineno or call.lineno
                        changed = True

            # helper(a, b): a and b inherit how helper's params are used.
            names = [a.arg for a in model.functions[target_fn].args.args if a.arg != "self"]
            for i, arg in enumerate(call.node.args):
                if i >= len(names):
                    break
                var = _base_name(arg)
                if var is None:
                    continue
                key = model.canonical(scoped(call.enclosing, var))
                buf = model.buffers.get(key)
                if buf is None:
                    continue
                for index in param_stored.get(target_fn, {}).get(names[i], []):
                    if index not in buf.stored_by:
                        buf.stored_by.append(index)
                        changed = True
                for index in param_loaded.get(target_fn, {}).get(names[i], []):
                    if index not in buf.loaded_by:
                        buf.loaded_by.append(index)
                        changed = True

        if not changed:
            break


#: What calling an instance or an autograd Function actually runs: ``obj(x)`` is
#: ``__call__`` -> ``forward``, and ``Cls.apply(x)`` additionally runs
#: ``setup_context`` (torch >= 2.0). ``backward``/``jvp``/``vmap`` run only under
#: autograd, which the timed forward never triggers.
FORWARD_PROTOCOL = ("forward", "setup_context", "__call__")


def _resolve_reachability(model: ModuleModel, visitors: dict[str, _FuncVisitor]) -> None:
    """Worklist from the entry point to a fixpoint over referenced names.

    Beyond plain helper calls we must follow two idioms that are extremely common in
    generated kernels, and whose absence produces a flood of bogus "dead kernel"
    findings:

    * a **submodule defined in the same file** -- ``self.norm = LayerNormTriton(...)``
      in ``__init__`` and ``self.norm(x)`` in forward; the launch lives in
      ``LayerNormTriton.forward``;
    * a **torch.autograd.Function** -- ``MishAutoFn.apply(x)``, where the launch lives
      in ``MishAutoFn.forward``.

    Two sets come out of the same walk, because the check families need opposite
    approximations:

    * ``reachable`` expands a referenced class to **all** of its methods. Checks
      asking "is this kernel ever used?" (F1.2, F1.6) must over-approximate: a
      missed edge would invent a dead kernel, so a Triton kernel launched only
      from ``backward()`` still counts as launched.
    * ``timed_scopes`` expands a referenced class only to its forward-call
      protocol (:data:`FORWARD_PROTOCOL`). Checks asking "what does the timed
      forward compute?" (F1.4, F1.5, F2.2, F2.3) must under-approximate: autograd
      ``backward``/``jvp``/``vmap`` never run under the benchmark's forward call,
      and torch ops there would otherwise be reported as cheating.
    """
    if model.forward_entry is None:
        return

    def all_methods(name: str) -> list[str]:
        return model.local_classes.get(name, [])

    def forward_protocol(name: str) -> list[str]:
        return [
            m
            for m in model.local_classes.get(name, [])
            if m.rsplit(".", 1)[-1] in FORWARD_PROTOCOL
        ]

    model.reachable = _walk(model, visitors, all_methods)
    model.timed_scopes = _walk(model, visitors, forward_protocol)


def _walk(
    model: ModuleModel,
    visitors: dict[str, _FuncVisitor],
    expand_class: Callable[[str], list[str]],
) -> set[str]:
    worklist = [model.forward_entry]
    seen: set[str] = set()

    while worklist:
        qual = worklist.pop()
        if qual in seen or qual not in visitors:
            continue
        seen.add(qual)

        # `self.helper()` resolves against the class the current method lives in;
        # module-level functions fall back to the entry class.
        own_cls = qual.rsplit(".", 1)[0] if "." in qual else model.model_class

        for ref in visitors[qual].referenced:
            if ref in model.functions:
                worklist.append(ref)

            # A reference to a local class (MyFn.apply, MyMod()(x), or just MyFn)
            # reaches the methods `expand_class` says calling it would run.
            head = ref.split(".", 1)[0]
            worklist.extend(expand_class(head))

            if ref.startswith("self."):
                attr = ref.split(".", 1)[1].split(".", 1)[0]
                # self.helper(...) -> "<OwnClass>.helper"
                if own_cls and f"{own_cls}.{attr}" in model.functions:
                    worklist.append(f"{own_cls}.{attr}")
                # self.norm(...) where self.norm = LayerNormTriton(...)
                for owner in model.attr_classes.get(attr, []):
                    worklist.extend(expand_class(owner))

    return seen
