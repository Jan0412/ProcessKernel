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

from kernel_gen.core import artifacts
from kernel_gen.core.backend import Backend, FakeBackend
from kernel_gen.core.critics import lint_critic
from kernel_gen.core.engine import _previous_check_ids, run_rounds
from kernel_gen.core.model import Attempt, Problem, Trajectory
from kernel_gen.core.sampling import (
    CODE_FENCE,
    PLAN_PREFIX,
    SamplingSpec,
    TracedCompletion,
    generate_batch_traced,
)
from kernel_gen.core.text import extract_code_block
from kernel_gen.core.trace import SEG_CODE, SEG_PLAN, TokenTrace

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


def test_the_whole_chain_preserves_the_trace_and_the_findings(tmp_path, dead_kernel_file):
    # One traced generation through backend -> sampling -> engine -> critics ->
    # artifacts -> disk, then read back. The plan prose, the token trace and the
    # findings-with-lineno must all survive -- the three things the pipeline used to drop.
    fenced = "```python\n" + dead_kernel_file + "\n```"
    backend = FakeBackend(default=PLAN + CODE_FENCE + fenced)
    problem = Problem(level=1, problem_id=7, name="7_Add.py", ref_arch_src=dead_kernel_file)
    spec = SamplingSpec(think_temperature=1.0, temperature=0.6, trace_topk=8)

    trajs = run_rounds(
        backend, [(problem, 0)],
        lambda p: "solve", lambda p, a: "repair",
        spec, critic=lint_critic(), rounds=1,
    )
    out = str(tmp_path)
    artifacts.write_traces(out, trajs, 0, vocab_size=backend.vocab_size)

    record = artifacts.read_jsonl(
        str(pathlib.Path(artifacts.trace_dir(out, 0)) / "attempts.jsonl")
    )[0]
    assert record["raw"].startswith(PLAN_PREFIX)  # the plan prose reached disk
    assert record["trace"]["n_plan_tokens"] > 0  # the trace reached disk
    assert record["findings"], "the linter's findings were dropped"
    assert any(f["data"].get("lineno") for f in record["findings"]), "linenos were dropped"


def test_attempt_and_review_still_hide_heavy_fields_from_the_journal():
    # The other half of the no-drop contract: raw/trace/findings must NOT leak into
    # to_dict, or lint_loop.jsonl grows and --skip-existing slows on every resume.
    from kernel_gen.core.model import Review

    attempt = Attempt(round=0, raw="R", code="C",
                      review=Review(text="", clean=True, findings=[{"x": 1}]),
                      trace=TokenTrace(*(np.empty(0) for _ in range(6)), meta={}))
    assert set(attempt.to_dict()) == {"round", "n_chars", "clean"}


# ---- class B: the right element chosen (on real data) --------------------


# The round -> "what was the model told" rule, which is a seam by construction: the critic
# writes it and both the repair prompt and the readout read it. KGEN-17/18 were one field
# meaning two different things at the two ends.


def _traj_with(data: dict) -> Trajectory:
    from kernel_gen.core.model import Review

    traj = Trajectory(problem=Problem(level=1, problem_id=1, name="x", ref_arch_src=""),
                      sample_id=0)
    traj.attempts.append(
        Attempt(round=0, raw="", code="", review=Review(text="", clean=False, data=data))
    )
    return traj


@pytest.mark.parametrize(
    "data, expected, why",
    [
        ({"shown_check_ids": ["S1.0"], "check_ids": []}, {"S1.0"},
         "gate-blocked: the prompt named S1.0 and no lint check ran"),
        ({"shown_check_ids": ["F1.2"], "check_ids": ["F1.2", "F1.4"]}, {"F1.2"},
         "severity staging hid the warn, so it was never shown"),
        ({"shown_check_ids": ["S1.3"], "check_ids": ["F1.2"]}, {"S1.3"},
         "the gate replaced the lint text entirely"),
        ({"shown_check_ids": [], "check_ids": ["F1.2"]}, set(),
         "nothing was shown: must NOT fall back to what fired"),
        ({"check_ids": ["F1.2", "F1.4"]}, {"F1.2", "F1.4"},
         "pre-gate journal: no shown_check_ids, so reproduce the old behaviour exactly"),
        ({"check_ids": []}, set(), "pre-gate journal with nothing to say"),
        ({}, set(), "a record with neither key"),
    ],
)
def test_the_previous_round_is_read_as_what_was_shown(data, expected, why):
    assert _previous_check_ids(_traj_with(data)) == expected, why


def test_a_slot_with_no_previous_round_has_nothing_to_repeat():
    traj = Trajectory(problem=Problem(level=1, problem_id=1, name="x", ref_arch_src=""),
                      sample_id=0)
    assert _previous_check_ids(traj) == set()

    traj.attempts.append(Attempt(round=0, raw="", code="", review=None))
    assert _previous_check_ids(traj) == set()  # a critic that crashed claims nothing


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

_CORE = pathlib.Path(__file__).parents[2] / "kernel_gen" / "core"
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
