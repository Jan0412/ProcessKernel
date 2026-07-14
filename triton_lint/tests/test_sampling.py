"""The two-pass think/code sampler.

This logic has been in production in ``generate_kernels_samples.py`` since the first
run and has never been tested anywhere. The core re-implements it, so it gets pinned
here: the prefill, the stop string, and above all the reassembly -- if the halves are
stitched back wrong, ``extract_code_block`` silently returns prose and every kernel in
the run is garbage.
"""

from __future__ import annotations

from kernel_gen.core.backend import FakeBackend
from kernel_gen.core.sampling import CODE_FENCE, PLAN_PREFIX, SamplingSpec, generate_batch
from kernel_gen.core.text import extract_code_block

PLAN = "fuse the elementwise ops\n"
CODE = "\nimport torch\n```\n"


def _model() -> FakeBackend:
    """A fake that answers like a real one: the reply it gives depends on the prompt.

    Left to itself (pass 1) it writes a plan, then a fenced kernel -- and the sampler's
    ``stop`` truncates it at the fence, which is the fake's job to honor. Handed a
    prompt that already contains its plan and ends with the fence (pass 2), it writes
    only the code. Getting this wrong in the *fixture* is the same mistake the sampler
    itself could make, so the fixture has to model it properly.
    """
    return FakeBackend(rules=[(PLAN, CODE)], default=PLAN + CODE_FENCE + CODE)


def test_single_pass_sends_one_prompt_per_slot():
    backend = _model()
    out = generate_batch(backend, ["a", "b", "c"], SamplingSpec(think_temperature=None))

    assert len(out) == 3
    assert len(backend.batches) == 1  # one call, not one per prompt
    assert len(backend.batches[0]) == 3
    assert extract_code_block(out[0]) == "import torch"


def test_think_pass_prefills_the_plan_heading_and_stops_at_the_fence():
    backend = _model()
    generate_batch(backend, ["solve this"], SamplingSpec(think_temperature=1.0))

    plan_batch, code_batch = backend.batches
    # Instruct models ignore "plan first" and emit the fence immediately; the prefill
    # is what forces them to start in prose.
    assert plan_batch[0].endswith(PLAN_PREFIX)
    # Pass 1 stopped at the fence, so pass 2 resumes from a prompt ending in it --
    # which is why the model continues with code and not with more prose.
    assert code_batch[0] == plan_batch[0] + PLAN + CODE_FENCE


def test_think_reassembly_round_trips_through_extract_code_block():
    backend = _model()
    completion = generate_batch(backend, ["solve this"], SamplingSpec(think_temperature=1.0))[0]

    assert completion == PLAN_PREFIX + PLAN + CODE_FENCE + CODE
    # The whole point: the stitched halves still look like one normal fenced answer.
    assert extract_code_block(completion) == "import torch"


def test_think_path_batches_every_slot_in_one_call_per_pass():
    backend = _model()
    generate_batch(backend, [f"p{i}" for i in range(7)], SamplingSpec(think_temperature=1.0))

    assert len(backend.batches) == 2  # exactly two passes, not two per prompt
    assert [len(b) for b in backend.batches] == [7, 7]


def test_the_system_prompt_is_rendered_into_every_prompt():
    backend = _model()
    generate_batch(backend, ["solve this"], SamplingSpec(system="BE TERSE"))

    assert "BE TERSE" in backend.batches[0][0]


def test_empty_prompt_list_never_reaches_the_backend():
    backend = _model()
    assert generate_batch(backend, [], SamplingSpec()) == []
    assert backend.batches == []
