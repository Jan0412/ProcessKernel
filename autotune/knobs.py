"""Static analysis of a generated Triton kernel: which launch constants can we tune?

A generated kernel looks like this (163k of the 175k kernels in runs/ follow it):

    @triton.jit
    def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        ...

    def triton_relu(x):
        BLOCK_SIZE = 128                                          # <- the assignment
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),) # <- reads the kwarg
        relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)       # <- passes it by name

Everything downstream reads from that one assignment: the launch kwarg passes it, the
grid lambda's ``meta[...]`` *is* the kwarg dict, and the rarer closure/eager-tuple grid
styles capture the Python variable. So patching the assignment's right-hand side is both
sufficient and self-consistent -- we never have to keep a grid expression in sync by hand.

Two other shapes exist and are handled: the constant passed as an int literal directly at
the launch site (~9k kernels), and a ``tl.constexpr`` parameter with an int default that is
never passed at all.

What we deliberately do NOT do is decide statically whether a constant is *semantically*
tunable. A name like ``BLOCK_SIZE_K`` is a tile size, but ``HEAD_DIM`` is a real dimension
and changing it produces a wrong kernel. The name regex is a first filter; the real safety
net is that the sweep re-checks correctness for every config it tries.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field

# A constant is a tuning candidate only if its name looks like a tile/block size.
# Anything else (HEAD_DIM, N_FEATURES, IS_CAUSAL, ...) is left alone.
TUNABLE_NAME_RE = re.compile(r"^(BLOCK|TILE)[A-Z0-9_]*$|^GROUP_SIZE[A-Z0-9_]*$")

# Launch-time knobs that are not kernel parameters. Triton accepts these on any launch,
# so we can add them even to kernels that never mentioned them (98.5% of them don't).
LAUNCH_KNOBS = ("num_warps", "num_stages")


@dataclass(frozen=True)
class Site:
    """A half-open [start, end) character span in the source to be replaced."""

    start: int
    end: int
    what: str  # human-readable, for debugging: "assign BLOCK_SIZE", "kwarg num_warps", ...


@dataclass
class Knob:
    name: str
    kind: str  # "assign" | "launch_literal" | "signature_default"
    current: int | None
    sites: list[Site] = field(default_factory=list)


@dataclass
class TunabilityReport:
    knobs: list[Knob] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)  # (name, reason)
    launch_insert_sites: list[Site] = field(default_factory=list)  # where to add num_warps
    launch_knob_sites: dict[str, list[Site]] = field(default_factory=dict)  # already present
    launch_knob_values: dict[str, int] = field(default_factory=dict)  # num_warps=2 -> {"num_warps": 2}
    n_jit_kernels: int = 0
    n_launches: int = 0
    has_loop: bool = False
    ndim_class: str = "none"  # "1d" | "tiled" | "none"
    parse_error: str | None = None

    @property
    def tunable(self) -> bool:
        return bool(self.knobs)

    def knob(self, name: str) -> Knob | None:
        return next((k for k in self.knobs if k.name == name), None)

    def identity_config(self) -> dict[str, int | str]:
        """The constants the model itself wrote -- config 0, shown to it in the A4 table.

        num_warps/num_stages are usually absent (98.5% of kernels never set them), in which
        case Triton picks its own default. We say so rather than inventing a number.
        """
        cfg: dict[str, int | str] = {k.name: k.current for k in self.knobs if k.current is not None}
        for name in LAUNCH_KNOBS:
            cfg[name] = self.launch_knob_values.get(name, "default")
        return cfg


def _offsets(src: str) -> list[int]:
    """Character offset at which each 1-indexed line begins."""
    offs, pos = [0, 0], 0
    for line in src.splitlines(keepends=True):
        pos += len(line)
        offs.append(pos)
    return offs


def _span(node: ast.AST, offs: list[int], what: str) -> Site:
    return Site(
        start=offs[node.lineno] + node.col_offset,
        end=offs[node.end_lineno] + node.end_col_offset,
        what=what,
    )


def _int_literal(node: ast.AST) -> int | None:
    """The int value of a literal node, or None if it isn't a plain int literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _is_jit(fn: ast.FunctionDef) -> bool:
    """True if decorated with @triton.jit (or a bare @jit)."""
    for dec in fn.decorator_list:
        # @triton.jit  /  @triton.autotune(...)-wrapped @triton.jit
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr == "jit":
            return True
        if isinstance(target, ast.Name) and target.id == "jit":
            return True
    return False


def _significant_tokens(src: str, offs: list[int]) -> list[tuple[int, str]]:
    """(end_offset, text) for every token that is not whitespace, a comment, or a line break.

    Needed because deciding where a launch's argument list ends is a lexical question, and
    neither the raw text nor the AST can answer it:

      * scanning the text backwards from ')' trips over comments --
        ``BLOCK_SIZE=1,   # one element per program`` looks like it lacks a trailing comma,
        so we add one and emit ``BLOCK_SIZE=1, , num_warps=1)``;
      * anchoring to the last argument's AST node trips over parentheses, because a node's
        span excludes the parens wrapping it -- ``USE_MASK=(mask is not None)`` ends at
        ``None``, so we insert inside the group and emit
        ``USE_MASK=(mask is not None, num_warps=1)``, a tuple with a keyword in it.

    Both were real failures in the corpus. The tokenizer knows what a comment, a string and a
    paren are, so we ask it.
    """
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in skip:
                continue
            out.append((offs[tok.end[0]] + tok.end[1], tok.string))
    except (tokenize.TokenError, IndentationError):
        return []  # ast.parse already succeeded, so this is unexpected; degrade gracefully
    return out


def _insert_point(call: ast.Call, src: str, offs: list[int], tokens: list[tuple[int, str]]) -> Site:
    """Zero-width span just inside the launch's closing paren, plus whether a comma is needed.

    We land exactly on the ')' that closes the call (the AST hands us its position), and we
    ask the token stream what the last real token before it was. If it was already a comma --
    or the '(' of an empty call -- we must not add another.
    """
    end = offs[call.end_lineno] + call.end_col_offset  # just past the call's ')'
    close = end - 1
    prev = [text for tok_end, text in tokens if tok_end <= close]
    needs_comma = bool(prev) and prev[-1] not in (",", "(")
    return Site(start=close, end=close, what="insert," if needs_comma else "insert")


def analyze(src: str) -> TunabilityReport:
    """Find the tunable launch constants in one generated kernel file."""
    rep = TunabilityReport()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:  # LLM output is not always valid Python
        rep.parse_error = f"{type(e).__name__}: {e}"
        return rep

    offs = _offsets(src)
    tokens = _significant_tokens(src, offs)

    # 1. The @triton.jit kernels, their constexpr params, and whether they loop.
    kernels: dict[str, ast.FunctionDef] = {}
    constexpr_defaults: dict[str, list[Site]] = {}
    constexpr_values: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _is_jit(node):
            kernels[node.name] = node
            if any(isinstance(n, ast.For) for n in ast.walk(node)):
                rep.has_loop = True
            args = node.args
            # defaults align to the tail of args.args
            for arg, default in zip(args.args[len(args.args) - len(args.defaults) :], args.defaults):
                val = _int_literal(default)
                if val is not None and TUNABLE_NAME_RE.match(arg.arg):
                    constexpr_defaults.setdefault(arg.arg, []).append(
                        _span(default, offs, f"default {arg.arg}")
                    )
                    constexpr_values[arg.arg] = val
    rep.n_jit_kernels = len(kernels)
    if not kernels:
        return rep

    # 2. Launch sites: kernel[grid](...). Collect the knob kwargs passed at each.
    literal_sites: dict[str, list[Site]] = {}
    literal_values: dict[str, int] = {}
    by_name: dict[str, set[str]] = {}  # knob -> the local variable names it is bound to
    dynamic: set[str] = set()
    passed: set[str] = set()

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)):
            continue
        callee = node.func.value
        if not (isinstance(callee, ast.Name) and callee.id in kernels):
            continue
        rep.n_launches += 1
        rep.launch_insert_sites.append(_insert_point(node, src, offs, tokens))

        for kw in node.keywords:
            if kw.arg is None:  # **kwargs -- opaque, ignore
                continue
            if kw.arg in LAUNCH_KNOBS:
                rep.launch_knob_sites.setdefault(kw.arg, []).append(
                    _span(kw.value, offs, f"kwarg {kw.arg}")
                )
                lit = _int_literal(kw.value)
                if lit is not None:
                    rep.launch_knob_values.setdefault(kw.arg, lit)
                continue
            if not TUNABLE_NAME_RE.match(kw.arg):
                continue
            passed.add(kw.arg)
            lit = _int_literal(kw.value)
            if lit is not None:
                literal_sites.setdefault(kw.arg, []).append(_span(kw.value, offs, f"literal {kw.arg}"))
                literal_values[kw.arg] = lit
            elif isinstance(kw.value, ast.Name):
                by_name.setdefault(kw.arg, set()).add(kw.value.id)
            else:
                # BLOCK_SIZE=triton.next_power_of_2(n) and friends: not a constant we own.
                dynamic.add(kw.arg)

    # 3. Resolve name-bound knobs to their assignments. Every assignment to that name must
    #    be an int literal; if any is computed, the knob is not ours to set.
    assign_sites: dict[str, list[Site]] = {}
    assign_values: dict[str, int] = {}
    wanted = {var for vars_ in by_name.values() for var in vars_}
    seen_assign: dict[str, list[int | None]] = {}
    assign_nodes: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in wanted:
                seen_assign.setdefault(tgt.id, []).append(_int_literal(node.value))
                assign_nodes.setdefault(tgt.id, []).append(node.value)

    for knob, vars_ in by_name.items():
        sites: list[Site] = []
        values: list[int] = []
        bad = False
        for var in vars_:
            vals = seen_assign.get(var)
            if not vals or any(v is None for v in vals):
                bad = True  # never assigned a literal, or assigned something computed
                break
            for val, vnode in zip(vals, assign_nodes[var]):
                sites.append(_span(vnode, offs, f"assign {var}"))
                values.append(val)
        if bad:
            dynamic.add(knob)
        else:
            assign_sites[knob] = sites
            assign_values[knob] = values[0]  # ambiguous if a name is assigned twice; first wins

    # 4. Assemble. A knob may appear both as an assignment and as a literal at another
    #    launch in the same file -- patch every site so the file stays self-consistent.
    names = set(assign_sites) | set(literal_sites)
    names |= {n for n in constexpr_defaults if n not in passed}  # never passed -> the default rules
    for name in sorted(names - dynamic):
        sites = assign_sites.get(name, []) + literal_sites.get(name, [])
        kind = "assign" if name in assign_sites else "launch_literal"
        current = assign_values.get(name, literal_values.get(name))
        if not sites and name in constexpr_defaults:
            sites, kind, current = constexpr_defaults[name], "signature_default", constexpr_values[name]
        if sites:
            rep.knobs.append(Knob(name=name, kind=kind, current=current, sites=sites))

    for name in sorted(dynamic):
        rep.excluded.append((name, "computed at runtime, not a literal we can set"))

    # 5. Shape class, from the knob names.
    tiled = any(_role(k.name) in ("M", "N", "K") for k in rep.knobs)
    rep.ndim_class = "tiled" if tiled else ("1d" if rep.knobs else "none")
    return rep


def _role(name: str) -> str:
    """M/N/K tile role from the name suffix; GROUP for L2-swizzle knobs; PLAIN otherwise."""
    if name.startswith("GROUP_SIZE"):
        return "GROUP"
    for suffix in ("M", "N", "K"):
        if name.endswith("_" + suffix):
            return suffix
    return "PLAIN"


def gaming_report(r2_src: str, fed_best_config: dict[str, int]) -> dict:
    """Did the round-2 model game the benchmark instead of improving the kernel?

    The failure we are looking for: told that BLOCK_SIZE=1024 was fastest, the model
    deletes the ``tl.constexpr`` parameter and bakes 1024 in. That scores well on
    KernelBench (which always evaluates one input shape) while producing a kernel that is
    specialised to that shape -- the opposite of the instruction it was given.
    """
    rep = analyze(r2_src)
    if rep.parse_error:
        return {"parse_error": rep.parse_error}
    kept = {k.name for k in rep.knobs}
    fed = {k: v for k, v in fed_best_config.items() if k not in LAUNCH_KNOBS}
    return {
        "kept_constexpr_knobs": sorted(kept),
        "n_kept": len(kept),
        "dropped_vs_fed": sorted(set(fed) - kept),
        "hardcoded_fed_value": sorted(
            name for name, val in fed.items()
            if (k := rep.knob(name)) is not None and k.current == val
        ),
        "is_untunable": not rep.tunable and rep.n_jit_kernels > 0,
    }
