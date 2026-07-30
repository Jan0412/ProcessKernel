"""Data model shared by the parser, the host-flow analysis and every check.

A check only ever sees a :class:`ModuleModel` and returns a list of
:class:`Finding`.  Everything a check could need must therefore be recovered
during analysis and stored here.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["info", "warn", "fail"]
ParseStatus = Literal["ok", "partial", "syntax_error", "empty", "read_error"]
KernelKind = Literal["matmul", "reduction", "elementwise", "copy", "unknown"]


# --------------------------------------------------------------------------
# Analysis model
# --------------------------------------------------------------------------


@dataclass
class ParamRole:
    """How a ``@triton.jit`` kernel uses one of its parameters."""

    name: str
    index: int
    is_constexpr: bool = False
    stored: bool = False  # pointer reaches the first arg of tl.store
    loaded: bool = False  # pointer reaches the first arg of tl.load
    atomic: bool = False  # target of tl.atomic_*
    #: at least one store to this pointer writes a strided/diagonal address (does not
    #: cover every element), so a zero-init of the bound buffer may be required
    partial_store: bool = False

    @property
    def is_pointer(self) -> bool:
        return self.stored or self.loaded or self.atomic


@dataclass
class KernelDef:
    name: str
    node: ast.FunctionDef
    params: dict[str, ParamRole] = field(default_factory=dict)
    param_order: list[str] = field(default_factory=list)
    kind: KernelKind = "unknown"
    has_autotune: bool = False
    lineno: int = 0
    #: other kernels this kernel calls in its body (a @triton.jit device function is
    #: inlined by Triton, never [grid]-launched) -- see F1.2
    calls: set[str] = field(default_factory=set)

    def outputs(self) -> list[str]:
        return [p.name for p in self.params.values() if p.stored]

    def inputs(self) -> list[str]:
        return [p.name for p in self.params.values() if p.loaded and not p.stored]


@dataclass
class LaunchSite:
    """A ``kernel[grid](...)`` call site."""

    index: int
    kernel_name: str
    enclosing: str  # qualname of the enclosing function, e.g. "ModelNew.forward"
    lineno: int
    loop_depth: int = 0
    loop_vars: list[str] = field(default_factory=list)
    # kernel parameter name -> host argument expression
    arg_map: dict[str, ast.expr] = field(default_factory=dict)
    #: this launch sits in a loop that carries a data dependency through the kernel
    #: (an input arg is reassigned from an output arg) -- a sequential recurrence that
    #: must not be moved into the launch grid (see F2.2)
    recurrence: bool = False


@dataclass
class Buffer:
    """A host-side tensor, after alias resolution."""

    canonical: str
    alloc_fn: str | None = None  # "empty" | "zeros" | "empty_like" | ... | None
    alloc_lineno: int | None = None
    alloc_call: ast.Call | None = None  # kept for shape inference
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    nbytes: int | None = None
    stored_by: list[int] = field(default_factory=list)  # LaunchSite indices
    loaded_by: list[int] = field(default_factory=list)
    read_by_host: bool = False
    returned: bool = False
    is_forward_input: bool = False


@dataclass
class HostCall:
    """A call in host code (i.e. outside any ``@triton.jit`` body)."""

    qualname: str  # "torch.matmul", "F.softmax", "x.contiguous", "self.linear"
    enclosing: str
    lineno: int
    is_method: bool = False
    receiver: str | None = None  # canonical buffer name for method calls
    node: ast.Call | None = None
    assigned_to: str | None = None  # `y = helper(x)` -> "y"


@dataclass
class ModuleModel:
    path: str = ""
    source: str = ""
    tree: ast.Module | None = None
    parse_status: ParseStatus = "ok"

    kernels: dict[str, KernelDef] = field(default_factory=dict)
    launches: list[LaunchSite] = field(default_factory=list)
    functions: dict[str, ast.FunctionDef] = field(default_factory=dict)

    model_class: str | None = None
    entry: str | None = None  # e.g. "ModelNew.forward"; None when forward is inherited
    #: the forward the benchmark actually enters: ``entry`` if spelled out, else the
    #: inherited ``"{Base}.forward"`` resolved through in-file bases (see parsing). This
    #: is the reachability root; ``entry`` stays None for an inherited forward.
    forward_entry: str | None = None
    #: conservative over-approximation: every function that could possibly run
    reachable: set[str] = field(default_factory=set)
    #: precise under-approximation: what the benchmark's forward() call executes
    #: (never autograd ``backward``/``jvp``/``vmap``) -- see hostflow
    timed_scopes: set[str] = field(default_factory=set)

    aliases: dict[str, str] = field(default_factory=dict)
    noncontiguous: set[str] = field(default_factory=set)
    buffers: dict[str, Buffer] = field(default_factory=dict)
    host_calls: list[HostCall] = field(default_factory=list)
    #: `nn.*` submodules bound in __init__, keyed by owning class then attribute:
    #: {"ModelNew": {"conv": "nn.Conv2d"}}. Per-class so an `nn.*` binding in one class
    #: cannot decide how an identically named attribute is judged in another (BUG-24).
    nn_modules_in_init: dict[str, dict[str, str]] = field(default_factory=dict)
    #: classes defined in this file, and their methods ("Cls" -> ["Cls.forward", ...])
    local_classes: dict[str, list[str]] = field(default_factory=dict)
    #: `self.attr = SomeLocalClass(...)` -> {"attr": ["SomeLocalClass"]}
    attr_classes: dict[str, list[str]] = field(default_factory=dict)
    #: attrs whose container construction (`nn.Sequential(...)`) also builds a heavy
    #: `nn.*` module, so it is a genuine fallback even though it wraps a local class too.
    containers_with_torch: set[str] = field(default_factory=set)

    input_shapes: list[tuple[tuple[int, ...], str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    #: Why ``compile(source, path, "exec")`` refused this file, and on which line.
    #: Only the submission front end fills it in -- ``ast.parse`` succeeding does not mean
    #: CPython can load the module, and the linter has no opinion on the difference.
    compile_error: str | None = None
    compile_error_lineno: int | None = None

    # -- convenience ------------------------------------------------------

    @property
    def reachable_launches(self) -> list[LaunchSite]:
        """Launches in code reachable from the entry point.

        With no resolvable forward entry we cannot prove anything is unreachable,
        so every launch counts. Checks that would fire on emptiness must
        therefore also consult :attr:`forward_entry`.
        """
        if self.forward_entry is None:
            return list(self.launches)
        return [ls for ls in self.launches if ls.enclosing in self.reachable]

    @property
    def timed_launches(self) -> list[LaunchSite]:
        """Launches the timed forward() actually executes (see :attr:`timed_scopes`)."""
        if self.forward_entry is None:
            return list(self.launches)
        return [ls for ls in self.launches if ls.enclosing in self.timed_scopes]

    def canonical(self, name: str) -> str:
        seen: set[str] = set()
        while name in self.aliases and name not in seen:
            seen.add(name)
            name = self.aliases[name]
        return name

    def buffer(self, name: str) -> Buffer | None:
        return self.buffers.get(self.canonical(name))


@dataclass
class Finding:
    check_id: str
    severity: Severity
    message: str  # actionable, human readable; fed back to the LLM later
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "message": self.message,
            "data": self.data,
        }


@dataclass
class FileReport:
    path: str
    parse_status: ParseStatus
    findings: list[Finding] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    run_name: str | None = None
    level: int | None = None
    problem_id: int | None = None
    sample_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "level": self.level,
            "problem_id": self.problem_id,
            "sample_id": self.sample_id,
            "path": self.path,
            "parse_status": self.parse_status,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
