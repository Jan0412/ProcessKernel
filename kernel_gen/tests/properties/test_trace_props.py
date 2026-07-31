"""Property-based tests for ``kernel_gen.core.trace`` (Hypothesis).

Example tests pin the values I thought to check. These pin the invariants that must hold
for *every* input, exploring the space Hypothesis generates -- which is where the trace
math is most likely to be wrong (a NaN from an all-tail row, an entropy above its bound,
an array that comes out a different length).
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from kernel_gen.core.trace import concat_passes, derive_scalars, pack, read_trace, write_trace

# Top-K logprob matrices: T tokens, K alternatives, all finite and <= 0 (they are log
# probabilities). derive_scalars renormalizes, so the exact scale does not matter.
topk_matrices = hnp.arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(1, 12), st.integers(2, 20)),
    elements=st.floats(min_value=-40.0, max_value=-1e-4, allow_nan=False, allow_infinity=False),
)


@given(lp=topk_matrices)
def test_entropy_is_between_zero_and_log_k(lp):
    k = lp.shape[1]
    entropy = derive_scalars(lp)["entropy"]
    assert np.all(entropy >= -1e-6)
    assert np.all(entropy <= math.log(k) + 1e-4)


@given(lp=topk_matrices)
def test_every_scalar_is_finite_and_per_token(lp):
    scalars = derive_scalars(lp, vocab_size=151936)
    for name, values in scalars.items():
        assert values.shape == (lp.shape[0],), name
        assert np.all(np.isfinite(values)), name


@given(lp=topk_matrices)
def test_tail_mass_is_a_probability(lp):
    tail = derive_scalars(lp)["tail_mass"]
    assert np.all(tail >= -1e-9)
    assert np.all(tail <= 1.0 + 1e-9)


# Token-id / logprob rows for pack(): a list of rows, each row a list of (id, logprob).
def _rows(draw, n_tokens, k):
    rows = []
    for _ in range(n_tokens):
        ids = draw(st.lists(st.integers(0, 150000), min_size=k, max_size=k, unique=True))
        lps = draw(st.lists(st.floats(-30, -1e-4, allow_nan=False), min_size=k, max_size=k))
        rows.append(list(zip(ids, lps)))
    return rows


@st.composite
def token_traces(draw):
    n = draw(st.integers(1, 10))
    k = draw(st.integers(2, 12))
    rows = _rows(draw, n, k)
    token_ids = [row[draw(st.integers(0, k - 1))][0] for row in rows]  # sample from the row
    return token_ids, rows, k


@given(data=token_traces())
def test_pack_makes_every_array_the_same_length(data):
    token_ids, rows, k = data
    trace = pack(token_ids, rows, k=k)
    n = len(token_ids)
    for arr in (trace.token_ids, trace.topk_ids, trace.topk_lp, trace.sampled_lp,
                trace.sampled_rank, trace.seg):
        assert arr.shape[0] == n
    assert trace.topk_ids.shape == (n, k)


@given(data=token_traces())
def test_the_sampled_token_is_always_found_and_ranked(data):
    # The sampled token is drawn from its own row, so pack must always locate it: a rank
    # >= 1 and a real logprob. A rank of 0 would mean "not found" -- a silent drop.
    token_ids, rows, k = data
    trace = pack(token_ids, rows, k=k)
    assert np.all(trace.sampled_rank >= 1)
    assert np.all(trace.sampled_lp <= 0.0)


@given(data=token_traces())
def test_write_then_read_is_the_identity(data):
    # tempfile, not the tmp_path fixture: a function-scoped fixture is not reset between
    # Hypothesis examples, which the health check rightly flags. Each example gets a
    # fresh file here.
    import os
    import tempfile

    token_ids, rows, k = data
    original = pack(token_ids, rows, k=k)
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        write_trace(path, original)
        restored = read_trace(path)
    finally:
        os.unlink(path)

    assert restored.token_ids.tolist() == original.token_ids.tolist()
    assert restored.sampled_rank.tolist() == original.sampled_rank.tolist()
    np.testing.assert_array_equal(restored.topk_ids, original.topk_ids)


@given(a=token_traces(), b=token_traces())
@settings(max_examples=50)
def test_concat_length_is_the_sum_and_seg_flips_at_the_boundary(a, b):
    from kernel_gen.core.trace import SEG_CODE, SEG_PLAN

    ka = a[2]
    # concat requires equal K; force b to a's K by repacking with ka
    plan = pack(a[0], a[1], k=ka)
    code = pack(b[0], [row[:ka] for row in b[1]], k=ka)
    trace = concat_passes(plan, code)

    assert len(trace) == len(plan) + len(code)
    assert trace.meta["n_plan_tokens"] == len(plan)
    assert np.all(trace.seg[: len(plan)] == SEG_PLAN)
    assert np.all(trace.seg[len(plan):] == SEG_CODE)
