"""Host-side (non-kernel) analysis.

Recovers, for every function in the module:

* launch sites -- ``kernel[grid](...)`` is a ``Call`` whose ``func`` is a
  ``Subscript`` -- along with the Python loop nesting they sit in and a map from
  kernel parameter name to the host expression passed for it;
* reachability from the entry point, following calls through helper functions to
  a fixpoint (a kernel launched from a helper two hops away is still reachable);
* buffers and their aliases (``view``/``permute``/... produce a second name for
  the same memory, so they must be unioned before asking "is this intermediate
  read by anyone else?");
* which buffers flow into the return value, and which are read by host code.

Buffer names are scoped as ``"<function qualname>::<var>"``: an intermediate
allocated and consumed inside a helper is a local fact about that helper.
"""

from __future__ import annotations

import ast

from .model import Buffer, HostCall, LaunchSite, ModuleModel
from .parsing import _dotted

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


def scoped(func: str, name: str) -> str:
    return f"{func}::{name}"


class _FuncVisitor(ast.NodeVisitor):
    """Walks one host function, recording launches, calls, aliases and buffers."""

    def __init__(self, model: ModuleModel, qualname: str, kernels: set[str]):
        self.model = model
        self.qual = qualname
        self.kernels = kernels
        self.loop_stack: list[str] = []
        # Name-load bookkeeping used to decide `read_by_host`.
        self.loads: dict[str, int] = {}
        self.launch_loads: dict[str, int] = {}
        self.return_loads: dict[str, int] = {}
        # Passing a tensor into a helper that launches a kernel is plumbing, not a
        # host read -- otherwise every interprocedural intermediate looks "used".
        self.helper_loads: dict[str, int] = {}
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
        self.generic_visit(node)
        self.loop_stack.pop()

    def visit_While(self, node: ast.While) -> None:
        self.loop_stack.append("while")
        self.generic_visit(node)
        self.loop_stack.pop()

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
                    if cls.startswith("nn.") or cls.startswith("torch.nn."):
                        self.model.nn_modules_in_init[t.attr] = cls

                # self.layernorm = LayerNormTriton(...) -- a submodule defined in this
                # file. Its forward() is where the kernel is actually launched, so
                # reachability must follow it. Also catches nn.Sequential(Mish(), ...).
                local = [
                    n.id
                    for n in ast.walk(node.value)
                    if isinstance(n, ast.Name) and n.id in self.model.local_classes
                ]
                if local:
                    self.model.attr_classes.setdefault(t.attr, []).extend(local)

        if len(targets) == 1:
            name = targets[0].id
            self._record_binding(name, node.value)
            if isinstance(node.value, ast.Call):
                self.assign_target[id(node.value)] = name

        self.generic_visit(node)

    def _record_binding(self, name: str, value: ast.expr) -> None:
        key = scoped(self.qual, name)

        if isinstance(value, ast.Call):
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

        self._count(node, self.loads)
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
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _returned_names(node.value)
    if isinstance(node, ast.Call):
        # A method call (`out.view(...)`) returns a view of its receiver; a plain
        # function call (`helper(tmp)`) returns something new.
        if isinstance(node.func, ast.Attribute):
            return _returned_names(node.func.value)
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
        v = _FuncVisitor(model, qual, kernels)
        for stmt in node.body:
            v.visit(stmt)
        visitors[qual] = v

        # Parameters are buffers too (an in-place kernel writes into one).
        is_entry_like = qual.endswith(".forward")
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            buf = v.buf(arg.arg)
            buf.is_forward_input = is_entry_like

    _resolve_reads(model, visitors)
    _resolve_reachability(model, visitors)
    _propagate_interprocedural(model)


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

    Both are handled by the same rule: referencing a locally defined class from
    reachable code makes all of that class's methods reachable. That is deliberately
    conservative -- we would rather miss a genuinely dead kernel than invent one.
    """
    if model.entry is None:
        return

    cls = model.model_class
    worklist = [model.entry]
    seen: set[str] = set()

    def expand_class(name: str) -> list[str]:
        return model.local_classes.get(name, [])

    while worklist:
        qual = worklist.pop()
        if qual in seen or qual not in visitors:
            continue
        seen.add(qual)

        for ref in visitors[qual].referenced:
            if ref in model.functions:
                worklist.append(ref)

            # A bare reference to a local class (MyFn.apply, or just MyFn) reaches
            # every method of it.
            head = ref.split(".", 1)[0]
            worklist.extend(expand_class(head))

            if ref.startswith("self."):
                attr = ref.split(".", 1)[1].split(".", 1)[0]
                # self.helper(...) -> "ModelNew.helper"
                if cls and f"{cls}.{attr}" in model.functions:
                    worklist.append(f"{cls}.{attr}")
                # self.norm(...) where self.norm = LayerNormTriton(...)
                for owner in model.attr_classes.get(attr, []):
                    worklist.extend(expand_class(owner))

    model.reachable = seen
