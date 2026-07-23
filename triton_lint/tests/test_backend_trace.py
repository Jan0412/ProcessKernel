"""The traced backend seam: ``complete_traced`` and what the fake has to reproduce.

``FakeBackend`` is the only backend the test suite can run, so anything it does not
model is untested in this repo. These tests pin the two vLLM behaviours the fake exists
to imitate -- token ids that outrun the truncated text, and logprob rows that are not
sorted -- because a fake that quietly gets either one right would hide a real bug.
"""

from __future__ import annotations

import pytest

from kernel_gen.core.backend import Backend, Completion, FakeBackend
from kernel_gen.core.trace import pack

PLAN = "fuse the elementwise ops\n"
FENCE = "```python"
CODE = "\nimport torch\n```\n"


def _model() -> FakeBackend:
    return FakeBackend(rules=[(PLAN, CODE)], default=PLAN + FENCE + CODE)


def test_complete_still_returns_plain_strings():
    # The old interface is the one every existing caller uses; it must not have moved.
    assert _model().complete(["x"], temperature=0.3, max_tokens=64) == [PLAN + FENCE + CODE]


def test_complete_and_complete_traced_agree_on_the_text():
    backend = _model()
    plain = backend.complete(["x"], temperature=0.3, max_tokens=64, stop=[FENCE])
    traced = backend.complete_traced(["x"], temperature=0.3, max_tokens=64, stop=[FENCE])

    assert plain == [traced[0].text] == [PLAN]


def test_the_base_class_degrades_to_text_only_rather_than_raising():
    # A backend with no internals to offer implements nothing and still works.
    class TextOnly(Backend):
        def complete(self, prompts, *, temperature, max_tokens, stop=None):
            return ["hello"] * len(prompts)

    out = TextOnly().complete_traced(["a", "b"], temperature=0.0, max_tokens=8)

    assert [c.text for c in out] == ["hello", "hello"]
    assert out[0].token_ids is None and out[0].topk is None


def test_no_logprobs_requested_means_no_logprobs_returned():
    out = _model().complete_traced(["x"], temperature=0.3, max_tokens=64)

    assert out[0].token_ids  # ids are free
    assert out[0].topk is None  # the expensive part is opt-in


def test_token_ids_cover_the_stop_string_that_the_text_does_not():
    # vLLM's detokenizer excludes the stop token from `text` and then re-appends it to
    # `token_ids`. So the plan pass returns MORE tokens than its text accounts for --
    # the ones that spelled the fence. Concatenating passes as if this were not true is
    # the misalignment the whole seam design is about.
    out = _model().complete_traced(["x"], temperature=1.0, max_tokens=64, stop=[FENCE])[0]

    assert out.text == PLAN
    assert out.stop_reason == FENCE

    # The count is the whole point: tokenizing the text we were GIVEN would produce
    # fewer tokens than we were HANDED, and the difference is the fence.
    text_only = FakeBackend(default=PLAN).complete_traced(["x"], temperature=1.0, max_tokens=64)[0]
    assert len(out.token_ids) > len(text_only.token_ids)


def test_logprob_rows_are_sampled_token_first_not_best_first():
    # The vLLM quirk that pack() has to undo. If the fake sorted its rows, the sort in
    # pack() would be dead code that no test could ever have exercised.
    out = _model().complete_traced(["x"], temperature=1.0, max_tokens=64, logprobs=8)[0]

    assert [row[0][0] for row in out.topk] == out.token_ids
    assert any(row[0][1] < row[1][1] for row in out.topk), "no row was out of order"


def test_some_rows_are_wider_than_k_because_the_sample_fell_outside():
    out = _model().complete_traced(["x"], temperature=1.0, max_tokens=64, logprobs=4)[0]

    widths = {len(row) for row in out.topk}
    assert widths <= {4, 5}
    assert 5 in widths, "the sampled-outside-top-K case never occurred"


def test_a_ragged_row_still_packs_rectangularly_with_its_sample_recorded():
    out = _model().complete_traced(["x"], temperature=1.0, max_tokens=64, logprobs=4)[0]
    trace = pack(out.token_ids, out.topk, k=4)

    assert trace.topk_ids.shape == (len(out.token_ids), 4)
    # Every token's own logprob survived, including the ones truncation dropped.
    assert (trace.sampled_rank > 0).all()
    assert (trace.sampled_lp < 0).all()


def test_the_fake_is_deterministic_so_fixtures_do_not_drift():
    first = _model().complete_traced(["x"], temperature=1.0, max_tokens=64, logprobs=4)[0]
    second = _model().complete_traced(["x"], temperature=1.0, max_tokens=64, logprobs=4)[0]

    assert first.token_ids == second.token_ids
    assert first.topk == second.topk


def test_one_completion_per_prompt_in_order():
    backend = FakeBackend(rules=[("alpha", "A"), ("beta", "B")], default="?")
    out = backend.complete_traced(
        ["alpha", "beta", "gamma"], temperature=0.3, max_tokens=8, logprobs=2
    )

    assert [c.text for c in out] == ["A", "B", "?"]
    assert len(backend.batches) == 1


def test_completion_defaults_leave_every_internal_absent():
    bare = Completion(text="hi")
    assert (bare.token_ids, bare.topk, bare.finish_reason, bare.stop_reason) == (
        None,
        None,
        None,
        None,
    )


# -- vLLM's two logprob container shapes -----------------------------------
#
# These use vLLM's real classes rather than stand-ins. They are the only tests in the
# suite that import vllm, and they import it lazily and skip when it is absent, because
# _topk's whole job is to read a data structure this repo does not own.


def _vllm_logprobs(flat: bool):
    """Build a real vLLM logprob container the way its engine does."""
    logprobs = pytest.importorskip("vllm.logprobs")
    container = logprobs.create_sample_logprobs(flat)
    # Two positions. vLLM inserts the SAMPLED token first, then the top-K in rank order,
    # so position 0's sampled token (id 3) ranked 2nd and position 1's (id 99) ranked
    # outside the top-2 entirely -- the ragged case.
    logprobs.append_logprobs_for_next_position(
        container, [3, 7, 3], [-1.2, -0.2, -1.2], [None] * 3, 2, 2
    )
    logprobs.append_logprobs_for_next_position(
        container, [99, 1, 2], [-9.0, -0.1, -2.0], [None] * 3, 5, 2
    )
    return container


@pytest.mark.parametrize("flat", [False, True])
def test_both_vllm_logprob_containers_read_the_same(flat):
    from kernel_gen.core.backend import _topk

    rows = _topk(_vllm_logprobs(flat))

    assert len(rows) == 2
    # Position 0: the dict shape dedupes the repeated sampled token, so it is 2 wide.
    assert dict(rows[0]) == {3: pytest.approx(-1.2), 7: pytest.approx(-0.2)}
    # Position 1: the sampled token ranked outside the top-2, so the row is 3 wide.
    assert dict(rows[1]) == {
        99: pytest.approx(-9.0),
        1: pytest.approx(-0.1),
        2: pytest.approx(-2.0),
    }


@pytest.mark.parametrize("flat", [False, True])
def test_packing_a_real_vllm_container_recovers_the_sampled_logprobs(flat):
    from kernel_gen.core.backend import _topk

    trace = pack([3, 99], _topk(_vllm_logprobs(flat)), k=2)

    assert trace.topk_ids[0].tolist() == [7, 3]  # sorted best-first, not as returned
    assert trace.sampled_lp.tolist() == [pytest.approx(-1.2), pytest.approx(-9.0)]
    assert trace.sampled_rank.tolist() == [2, 3]
    assert 99 not in trace.topk_ids  # truncated away, its logprob kept


def test_the_flat_container_is_not_iterated_into_dicts():
    # Iterating a FlatLogprobs materializes exactly the per-position dicts that asking
    # for it was meant to avoid. _topk must slice the parallel lists instead.
    container = _vllm_logprobs(flat=True)

    class Tripwire(type(container)):
        def __iter__(self):
            raise AssertionError("_topk iterated the flat container")

        def __getitem__(self, index):
            raise AssertionError("_topk indexed the flat container")

    from kernel_gen.core.backend import _topk

    tripwire = Tripwire(
        start_indices=container.start_indices,
        end_indices=container.end_indices,
        token_ids=container.token_ids,
        logprobs=container.logprobs,
        ranks=container.ranks,
        decoded_tokens=container.decoded_tokens,
    )
    assert len(_topk(tripwire)) == 2


def test_no_logprobs_container_is_none_not_an_empty_list():
    from kernel_gen.core.backend import _topk

    assert _topk(None) is None
