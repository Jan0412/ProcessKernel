"""The two blocks the PRM's training runs appended to KernelBench's prompt (paper appendix,
"Generation Prompt"): an input contract and a note on fp32 precision.

Off by default; an empty set leaves the prompt byte for byte unchanged.
"""

from __future__ import annotations

import ast
from typing import Callable

from .model import Problem

#: Fixed order, never a set: the rendered prompt must not depend on iteration order.
DELTA_ORDER: tuple[str, ...] = ("contract", "precision")


def _init_args(source: str) -> list | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    last = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_init_inputs":
            last = node
    if last is None:
        return None
    for node in ast.walk(last):
        if isinstance(node, ast.Return) and node.value is not None:
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return None
            return list(value) if isinstance(value, (list, tuple)) else [value]
    return None


def _contract_block(problem: Problem) -> str | None:
    from processkernel.checker.lint.shapes import shapes_from_source

    try:
        shapes = shapes_from_source(problem.ref_arch_src)
    except Exception:  # noqa: BLE001 - a missing block is better than a failed run
        return None
    # A partial contract is worse than none: omit unless every input resolved.
    if not shapes or any(s is None for s in shapes):
        return None
    args = _init_args(problem.ref_arch_src)
    if args is None:
        return None

    ctor = f"ModelNew({', '.join(repr(a) for a in args)})"
    lines = [
        "## Input contract",
        f"ModelNew is constructed as {ctor} and called with {len(shapes)} positional "
        "inputs, all on CUDA and contiguous:",
    ]
    # eval casts every input tensor to fp32, so report that, not the declared dtype.
    for i, (shape, _dtype) in enumerate(shapes, 1):
        dims = ", ".join(str(d) for d in shape)
        lines.append(f"  arg {i}: float32, shape ({dims})")
    lines.append(
        "This supersedes any shape stated in comments or docstrings above."
    )
    return "\n".join(lines)


_PRECISION = """\
## Numerical precision
Correctness is checked against the PyTorch reference running in true FP32, with a
tolerance of 1e-4. Triton's tl.dot defaults to TF32 on this GPU, which carries roughly
1e-3 of error and therefore FAILS this check. If you call tl.dot on float32 inputs you
must pass input_precision="ieee". Note also that cuBLAS is already near-optimal for FP32
matmul here: fusing work around torch.matmul usually beats reimplementing the GEMM.
"""


def _precision_block(problem: Problem) -> str | None:
    return _PRECISION


BLOCKS: dict[str, Callable[[Problem], str | None]] = {
    "contract": _contract_block,
    "precision": _precision_block,
}


def parse_deltas(spec: str | None) -> frozenset[str]:
    if not spec:
        return frozenset()
    names = {part.strip() for part in spec.split(",") if part.strip()}
    unknown = sorted(names - set(DELTA_ORDER))
    if unknown:
        raise ValueError(
            f"unknown prompt delta {unknown}; known: {list(DELTA_ORDER)}"
        )
    return frozenset(names)


def apply_deltas(prompt: str, problem: Problem, deltas: frozenset[str]) -> str:
    out = prompt
    for name in DELTA_ORDER:
        if name not in deltas:
            continue
        block = BLOCKS[name](problem)
        if block:
            out = out.rstrip("\n") + "\n\n" + block.strip("\n") + "\n"
    return out
