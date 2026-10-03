"""Generation's two non-negotiable properties: it batches, and it degrades.

One batch across every problem is invisible in the output -- a per-problem loop produces
byte-identical kernels, just far too slowly to run. So it is pinned here: a regression to
"loop over the problems" must FAIL a test, not merely be slow.

Degrading is the other half. A run is hours of GPU time; one unparseable completion must
cost that slot, not the run.
"""

from __future__ import annotations

import pytest

from processkernel.generation.core.backend import FakeBackend
from processkernel.generation.core.engine import generate
from processkernel.generation.core.model import Problem
from processkernel.generation.core.sampling import SamplingSpec

SPEC = SamplingSpec(think_temperature=None)  # single pass: one backend call per batch

GOOD = "```python\ngood\n```"


def problems(n: int) -> list[Problem]:
    return [
        Problem(level=1, problem_id=i, name=f"{i}_P.py", ref_arch_src="ref") for i in range(n)
    ]


def slots(probs: list[Problem], num_samples: int) -> list[tuple[Problem, int]]:
    return [(p, s) for p in probs for s in range(num_samples)]


def build_prompt(problem: Problem) -> str:
    return f"solve problem {problem.problem_id}"


def run(backend, *, n_problems=3, num_samples=4, spec=SPEC):
    return generate(backend, slots(problems(n_problems), num_samples), build_prompt, spec)


# -- batching --------------------------------------------------------------


def test_every_slot_of_every_problem_is_one_batch():
    # THE test. 3 problems x 4 samples is ONE call of 12 prompts, not 3 calls of 4 and
    # certainly not 12 calls of 1.
    backend = FakeBackend(default=GOOD)
    trajs = run(backend, n_problems=3, num_samples=4)

    assert len(backend.batches) == 1
    assert len(backend.batches[0]) == 12
    assert len(trajs) == 12
    assert all(len(t.attempts) == 1 for t in trajs)


def test_no_slots_means_no_backend_call():
    backend = FakeBackend(default=GOOD)
    assert generate(backend, [], build_prompt, SPEC) == []
    assert backend.batches == []


def test_each_attempt_captures_the_exact_prompt_it_was_given():
    # The user turn the model saw is carried onto the attempt so a trace can reconstruct
    # the conversation without replaying the prompt builders.
    trajs = run(FakeBackend(default=GOOD), n_problems=2, num_samples=1)
    assert [t.attempts[0].prompt for t in trajs] == ["solve problem 0", "solve problem 1"]


def test_slots_keep_their_sample_ids_and_problems():
    trajs = run(FakeBackend(default=GOOD), n_problems=2, num_samples=3)
    assert [(t.problem.problem_id, t.sample_id) for t in trajs] == [
        (p, s) for p in range(2) for s in range(3)
    ]


# -- degrading -------------------------------------------------------------


def test_an_unparseable_completion_degrades_that_slot_only():
    backend = FakeBackend(rules=[("solve problem 0", "")], default=GOOD)
    trajs = run(backend, n_problems=2, num_samples=1)

    assert trajs[0].last.code == ""  # empty, recorded, still written
    assert trajs[1].last.code == "good"  # its sibling is untouched


def test_an_extraction_that_raises_degrades_that_slot_only(monkeypatch):
    # extract_code_block is defensive by design: a completion it cannot handle costs
    # that attempt, not the run.
    import processkernel.generation.core.engine as engine_mod

    def boom(_raw):
        raise ValueError("unparseable")

    monkeypatch.setattr(engine_mod, "extract_code_block", boom)
    trajs = run(FakeBackend(default=GOOD), n_problems=1, num_samples=2)

    assert all(t.attempts[0].code == "" for t in trajs)  # degraded, not crashed


def test_a_backend_that_drops_a_completion_is_a_hard_error():
    # Silent misalignment between prompts and completions would attribute one slot's
    # kernel to another. That is never recoverable, so it raises rather than degrades.
    class DroppingBackend(FakeBackend):
        def complete_traced(self, prompts, **kwargs):
            out = super().complete_traced(prompts, **kwargs)
            return out[:-1]  # drop one

    with pytest.raises(RuntimeError, match="returned 3 completions for 4 prompts"):
        run(DroppingBackend(default=GOOD), n_problems=2, num_samples=2)


# -- tracing ---------------------------------------------------------------


def test_a_trace_reaches_the_attempt_it_belongs_to():
    trajs = run(
        FakeBackend(default=GOOD),
        n_problems=2,
        num_samples=2,
        spec=SamplingSpec(think_temperature=None, trace_topk=4),
    )

    for traj in trajs:
        trace = traj.attempts[0].trace
        assert trace is not None and len(trace) > 0
        assert trace.k == 4


def test_tracing_off_leaves_the_attempt_text_identical():
    # The non-regression that matters: --trace must be a pure addition, so an untraced
    # run and a traced run must produce the same kernels from the same fake.
    off = run(FakeBackend(default=GOOD), n_problems=2, num_samples=2)
    on = run(
        FakeBackend(default=GOOD),
        n_problems=2,
        num_samples=2,
        spec=SamplingSpec(think_temperature=None, trace_topk=8),
    )

    assert [t.last.code for t in off] == [t.last.code for t in on]
    assert [t.last.raw for t in off] == [t.last.raw for t in on]
