"""Cross-module data-flow invariants -- the properties the audit's bug classes violate.

Each existing test pins one module. These pin the guarantees that span the seams, in the
three shapes the ``audit-kernel-gen`` skill hunts: **nothing dropped between the model
and disk, the right element chosen, arrays that stay aligned.** They are written so they
would FAIL on the buggy code, not merely characterize the current code -- and one
(``no untested public surface``) is a meta-guard that keeps this whole suite honest as
the package grows.
"""

from __future__ import annotations

import ast
import json
import pathlib

import numpy as np
import pytest

from processkernel.generation.core import artifacts
from processkernel.generation.core.backend import Backend, FakeBackend
from processkernel.generation.core.engine import generate
from processkernel.generation.core.model import Attempt, Problem
from processkernel.generation.core.sampling import (
    CODE_FENCE,
    PLAN_PREFIX,
    SamplingSpec,
    TracedCompletion,
    generate_batch_traced,
)
from processkernel.generation.core.text import extract_code_block
from processkernel.generation.core.trace import SEG_CODE, SEG_PLAN, TokenTrace

PLAN = "launch the kernel, do not fall back to torch\n"

_CORPUS = pathlib.Path(__file__).parents[1] / "fixtures" / "completions" / "corpus.jsonl"


def _load_corpus() -> list[dict]:
    return [json.loads(line) for line in _CORPUS.read_text().splitlines()]


# ---- class A: nothing dropped between the model and disk ------------------


def test_complete_traced_never_silently_nulls_the_internals():
    # The seam bug B-was-born-from: complete() returned only .text. A traced call must
    # carry ids, top-K and the finish reason through for every completion.
    backend = FakeBackend(default="```python\nimport torch\n```")
    outs = backend.complete_traced(["a", "b"], temperature=0.6, max_tokens=64, logprobs=8)

    for c in outs:
        assert c.token_ids, "token_ids dropped at the backend seam"
        assert c.topk is not None, "top-K dropped at the backend seam"
        assert c.finish_reason is not None, "finish_reason dropped at the backend seam"


def test_the_whole_chain_preserves_the_trace_and_the_plan(tmp_path, dead_kernel_file):
    # One traced generation through backend -> sampling -> engine -> artifacts -> disk,
    # then read back. The plan prose, the token trace and the exact code must all survive.
    fenced = "```python\n" + dead_kernel_file + "\n```"
    backend = FakeBackend(default=PLAN + CODE_FENCE + fenced)
    problem = Problem(level=1, problem_id=7, name="7_Add.py", ref_arch_src=dead_kernel_file)
    spec = SamplingSpec(think_temperature=1.0, temperature=0.6, trace_topk=8)

    trajs = generate(backend, [(problem, 0)], lambda p: "solve", spec)
    out = str(tmp_path)
    artifacts.write_traces(out, trajs, vocab_size=backend.vocab_size)

    record = artifacts.read_jsonl(
        str(pathlib.Path(artifacts.trace_dir(out)) / "attempts.jsonl")
    )[0]
    assert record["raw"].startswith(PLAN_PREFIX)  # the plan prose reached disk
    assert record["trace"]["n_plan_tokens"] > 0  # the trace reached disk
    assert record["code"] == trajs[0].last.code  # the kernel, verbatim
    assert record["prompt"] == "solve"


def test_attempt_still_hides_heavy_fields_from_the_journal():
    # The other half of the no-drop contract: raw/trace/prompt must NOT leak into
    # to_dict, or generation.jsonl grows and --skip-existing slows on every resume.
    attempt = Attempt(raw="R", code="C", prompt="P",
                      trace=TokenTrace(*(np.empty(0) for _ in range(6)), meta={}))
    assert set(attempt.to_dict()) == {"n_chars"}


# ---- class B: the right element chosen (on real data) --------------------


def test_extraction_is_the_last_valid_modelnew_on_the_whole_corpus():
    # Over every well-formed real completion, the extracted kernel equals the oracle
    # (last valid ModelNew) and parses. The property KGEN-1 violated, checked on the
    # real corpus rather than one repro.
    for case in _load_corpus():
        if case["category"] not in ("single_block", "revision", "revision_fragment_first"):
            continue
        out = extract_code_block(case["raw"])
        assert out.strip() == case["oracle"].strip(), case["id"]
        ast.parse(out)


# ---- class C: arrays stay aligned ----------------------------------------


def test_the_two_pass_trace_is_internally_aligned():
    backend = FakeBackend(default=PLAN + CODE_FENCE + "\nimport torch\n```\n")
    completion: TracedCompletion = generate_batch_traced(
        backend, ["solve"], SamplingSpec(think_temperature=1.0, temperature=0.3, trace_topk=8)
    )[0]
    trace = completion.trace

    lengths = {a.shape[0] for a in (trace.token_ids, trace.topk_ids, trace.topk_lp,
                                    trace.sampled_lp, trace.sampled_rank, trace.seg)}
    assert lengths == {len(trace)}  # every array one row per token
    n_plan = trace.meta["n_plan_tokens"]
    assert np.all(trace.seg[:n_plan] == SEG_PLAN) and np.all(trace.seg[n_plan:] == SEG_CODE)
    # the char offsets re-slice the assembled text back into its two halves
    text = completion.text
    assert text[trace.meta["plan_char_start"]:trace.meta["plan_char_end"]] == PLAN


def test_tracing_off_is_a_pure_addition():
    # The invariant that makes --trace safe to leave on: the text is identical whether
    # or not internals are captured.
    backend_a = FakeBackend(default=PLAN + CODE_FENCE + "\nimport torch\n```\n")
    backend_b = FakeBackend(default=PLAN + CODE_FENCE + "\nimport torch\n```\n")
    off = generate_batch_traced(backend_a, ["x"], SamplingSpec(think_temperature=1.0))
    on = generate_batch_traced(backend_b, ["x"], SamplingSpec(think_temperature=1.0, trace_topk=8))
    assert off[0].text == on[0].text


# ---- the meta-guard: no untested public surface --------------------------

_CORE = pathlib.Path(__file__).parents[3] / "src" / "processkernel" / "generation" / "core"
_TESTS = pathlib.Path(__file__).parents[1]


def _public_names():
    for pyfile in sorted(_CORE.glob("*.py")):
        if pyfile.name == "__init__.py":
            continue
        for node in ast.parse(pyfile.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    yield f"{pyfile.stem}.{node.name}", node.name


def test_every_public_core_symbol_is_referenced_by_a_test():
    # An alibi-suite grows by adding code without adding tests. This fails the moment a
    # public function or class in core/ has no test referencing it by name -- forcing a
    # real test, or a deliberate rename to _private.
    corpus = "\n".join(p.read_text() for p in _TESTS.rglob("test_*.py"))
    missing = sorted({qual for qual, name in _public_names() if name not in corpus})
    assert not missing, f"public core symbols with no test reference: {missing}"
