"""Source -> skeleton :class:`ModuleModel`: parse, split kernels from host code,
build the function table, resolve the entry point.
"""

from __future__ import annotations

import ast
import warnings

from .model import KernelDef, ModuleModel

#: Classes we treat as the benchmark entry point, in priority order.
ENTRY_CLASSES = ("ModelNew", "Model")


def _decorator_names(node: ast.FunctionDef) -> list[str]:
    """Dotted names of every decorator, including those written as calls.

    ``@triton.autotune(...)`` stacked above ``@triton.jit`` means we must look at
    all decorators, not just the first.
    """
    names: list[str] = []
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = _dotted(target)
        if name:
            names.append(name)
    return names


def _dotted(node: ast.expr | None) -> str | None:
    """``a.b.c`` for Attribute/Name chains, else ``None``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def is_triton_kernel(node: ast.FunctionDef) -> bool:
    """True if any decorator is Triton's ``jit``.

    Handles ``@triton.jit``, ``@triton.jit()``, and the bare ``@jit`` produced by
    ``from triton import jit``.
    """
    for name in _decorator_names(node):
        if name == "triton.jit" or name.endswith(".jit") or name == "jit":
            return True
    return False


def has_autotune(node: ast.FunctionDef) -> bool:
    return any(
        n.endswith("autotune") or n.endswith("heuristics") for n in _decorator_names(node)
    )


def _base_names(cls: ast.ClassDef) -> list[str]:
    return [n for n in (_dotted(b) for b in cls.bases) if n]


def _is_nn_module(cls: ast.ClassDef) -> bool:
    return any(b.endswith("Module") for b in _base_names(cls))


def build_skeleton(source: str, path: str = "") -> ModuleModel:
    """Parse *source* and recover kernels, the function table and the entry point."""
    model = ModuleModel(path=path, source=source)

    if not source.strip():
        model.parse_status = "empty"
        return model

    # Generated sources routinely put LaTeX in a non-raw docstring ("\sum_i x_i"), and
    # Python warns about the invalid escape while compiling them. That is a complaint
    # about the generation, not about us, so keep it out of the batch progress output --
    # but record it per file rather than dropping it. Passing `filename` means anything
    # that does escape names the offending file instead of "<unknown>".
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        try:
            tree = ast.parse(source, filename=path or "<unknown>")
        except (SyntaxError, ValueError) as exc:  # ValueError: null bytes
            model.parse_status = "syntax_error"
            model.notes.append(f"parse failed: {type(exc).__name__}: {exc}")
            return model

    for warning in caught:
        if issubclass(warning.category, SyntaxWarning):
            model.notes.append(f"line {warning.lineno}: {warning.message}")

    model.tree = tree

    # Kernel discovery: every @triton.jit function at *any* nesting depth. Generated
    # code routinely defines a kernel inside its launcher (`def launch(x): @triton.jit
    # def k(...): ...; k[grid](...)`), and a @triton.jit device function called from
    # inside another kernel is a kernel too. Walking only module/class level misses
    # both -- the file then looks kernel-free and the nested body leaks into the host
    # scan. So collect kernels by a full walk, and build the function table separately,
    # skipping anything that is a kernel.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and is_triton_kernel(node):
            if node.name in model.kernels:
                model.notes.append(
                    f"duplicate @triton.jit name `{node.name}` (line {node.lineno})"
                )
            model.kernels[node.name] = KernelDef(
                name=node.name,
                node=node,
                has_autotune=has_autotune(node),
                lineno=node.lineno,
            )

    # Function table: module-level functions plus "Class.method" qualnames. Kernels
    # are excluded (handled above); nested non-kernel helpers keep their existing
    # out-of-table status.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if not is_triton_kernel(node):
                model.functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            methods: list[str] = []
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and not is_triton_kernel(sub):
                    qual = f"{node.name}.{sub.name}"
                    model.functions[qual] = sub
                    methods.append(qual)
            model.local_classes[node.name] = methods

    model.model_class = _resolve_model_class(tree)
    if model.model_class:
        entry = f"{model.model_class}.forward"
        if entry in model.functions:
            model.entry = entry
        else:
            model.notes.append(f"{model.model_class} has no forward()")
    else:
        model.notes.append("no entry-point class found")

    # The forward the benchmark actually enters -- the reachability root. If the entry
    # class spells out its own forward, that is it; otherwise resolve the one it inherits
    # through in-file base classes (`class ModelNew(Base): pass`). `entry` stays None in the
    # inherited case (checks that need "is a forward spelled out" still see None); analyses
    # that need "what does the timed forward run" use forward_entry.
    if model.entry:
        model.forward_entry = model.entry
    elif model.model_class:
        model.forward_entry = _inherited_forward(tree, model.model_class, model.functions)

    return model


def _inherited_forward(
    tree: ast.Module, cls_name: str, functions: dict[str, ast.FunctionDef]
) -> str | None:
    """``"{Base}.forward"`` inherited by *cls_name* from an in-file base, or ``None``.

    BFS the class's bases (a base itself may inherit), matching only classes defined in
    this file -- an external base's forward is not something we can scan.
    """
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    seen: set[str] = set()
    queue = [cls_name]
    while queue:
        name = queue.pop(0)
        if name in seen or name not in classes:
            continue
        seen.add(name)
        for base in _base_names(classes[name]):
            base_name = base.rsplit(".", 1)[-1]
            if f"{base_name}.forward" in functions:
                return f"{base_name}.forward"
            queue.append(base_name)
    return None


def _resolve_model_class(tree: ast.Module) -> str | None:
    """ModelNew -> Model -> ``Model = X`` alias -> the sole nn.Module subclass."""
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    for name in ENTRY_CLASSES:
        if name in classes:
            return name

    # `Model = SomeClass` / `ModelNew = SomeClass` adapter aliases.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & set(ENTRY_CLASSES):
                src = _dotted(node.value)
                if src in classes:
                    return src

    modules = [name for name, cls in classes.items() if _is_nn_module(cls)]
    if len(modules) == 1:
        return modules[0]
    return None
