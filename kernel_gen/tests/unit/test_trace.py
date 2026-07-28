"""The per-token trace: packing, the two-pass seam, and the confidence arithmetic.

Every number here is checked against a hand-computed value, because there is no
downstream signal that would catch these being wrong. A misaligned seam or an inverted
confidence measure produces arrays of exactly the right shape full of exactly the wrong
values, and the first symptom would be a PRM that trains and does not work.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from kernel_gen.core.trace import (
    PAD_ID,
    SEG_CODE,
    SEG_PLAN,
    concat_passes,
    derive_scalars,
    pack,
    rank1_calibration,
    read_trace,
    summarize,
    write_trace,
)

VOCAB = 151936  # Qwen3's, so the self-certainty scale is the one we will really see


def _rows(*distributions: list[float]) -> list[list[tuple[int, float]]]:
    """Probability rows -> vLLM-shaped ``[(token_id, logprob), ...]``, best first."""
    return [
        [(i, math.log(p)) for i, p in enumerate(sorted(probs, reverse=True))]
        for probs in distributions
    ]


def _peaked(k: int = 20) -> list[float]:
    rest = (1.0 - 0.999) / (k - 1)
    return [0.999] + [rest] * (k - 1)


def _flat(k: int = 20) -> list[float]:
    return [1.0 / k] * k


# --------------------------------------------------------------------------- packing


def test_pack_keeps_the_sampled_token_even_when_it_is_not_the_argmax():
    # Sampling at temperature 1.0 regularly takes a runner-up; which one it took is the
    # signal, so surprisal must come from the sampled token and not from the top-1.
    topk = [[(7, math.log(0.6)), (3, math.log(0.3)), (9, math.log(0.1))]]
    trace = pack([3], topk, k=3)

    assert trace.token_ids.tolist() == [3]
    assert trace.topk_ids[0, 0] == 7  # argmax, unchanged
    assert trace.sampled_lp[0] == pytest.approx(math.log(0.3), abs=1e-6)
    assert trace.sampled_rank[0] == 2


def test_pack_sorts_rows_because_vllm_puts_the_sampled_token_first():
    # vLLM builds each position as {token_id: Logprob} with the SAMPLED token inserted
    # before the top-K, and a dict keeps first-insertion order even when the duplicate
    # key is overwritten. So a row genuinely arrives with the argmax in position 1, not
    # 0, whenever sampling took a runner-up. Trusting that order would file the sampled
    # token's logprob under top1_lp for exactly the tokens where the difference matters.
    as_vllm_returns_it = [[(3, math.log(0.3)), (7, math.log(0.6)), (9, math.log(0.1))]]
    trace = pack([3], as_vllm_returns_it, k=3)

    assert trace.topk_ids[0].tolist() == [7, 3, 9]
    assert derive_scalars(trace.topk_lp)["top1_lp"][0] == pytest.approx(math.log(0.6), abs=1e-3)


def test_pack_truncates_to_k_but_reads_the_sampled_logprob_first():
    # vLLM returns the top-K *plus* the sampled token when it ranked outside; the extra
    # entry is dropped from the rectangular array, so its logprob must be taken before.
    topk = [[(1, math.log(0.7)), (2, math.log(0.29)), (99, math.log(0.01))]]
    trace = pack([99], topk, k=2)

    assert trace.topk_ids.shape == (1, 2)
    assert 99 not in trace.topk_ids
    assert trace.sampled_lp[0] == pytest.approx(math.log(0.01), abs=1e-5)
    assert trace.sampled_rank[0] == 3  # rank survives even though the entry is dropped


def test_pack_pads_short_rows_rather_than_going_ragged():
    trace = pack([1, 2], [[(5, -0.1)], [(6, -0.2), (7, -3.0)]], k=4)

    assert trace.topk_ids.shape == (2, 4)
    assert trace.topk_ids[0].tolist() == [5, PAD_ID, PAD_ID, PAD_ID]
    assert np.isneginf(np.asarray(trace.topk_lp[0, 1:], dtype=np.float64)).all()


def test_pack_without_logprobs_still_produces_a_valid_trace():
    # A backend with no internals to give must not force every caller to branch.
    trace = pack([1, 2, 3], None, k=20)

    assert len(trace) == 3
    assert trace.topk_ids.shape == (3, 20)
    assert (trace.topk_ids == PAD_ID).all()


# ------------------------------------------------------------------------ the seam


def test_concat_passes_flips_seg_exactly_at_the_plan_code_boundary():
    plan = pack([1, 2, 3], None, k=4)
    code = pack([4, 5], None, k=4)
    trace = concat_passes(plan, code)

    assert len(trace) == 5
    assert trace.token_ids.tolist() == [1, 2, 3, 4, 5]
    assert trace.seg.tolist() == [SEG_PLAN] * 3 + [SEG_CODE] * 2
    assert trace.meta["n_plan_tokens"] == 3
    assert trace.meta["n_code_tokens"] == 2


def test_every_array_stays_the_same_length_after_concat():
    # The one invariant the whole file format rests on: token position t means the same
    # thing in all five arrays.
    trace = concat_passes(pack([1, 2], _rows(_flat(4), _flat(4)), k=4), pack([3], None, k=4))

    n = len(trace)
    for array in (
        trace.token_ids,
        trace.topk_ids,
        trace.topk_lp,
        trace.sampled_lp,
        trace.sampled_rank,
        trace.seg,
    ):
        assert array.shape[0] == n


def test_concat_passes_rejects_passes_that_disagree_on_k():
    with pytest.raises(ValueError, match="disagree on K"):
        concat_passes(pack([1], None, k=4), pack([2], None, k=8))


def test_caller_supplied_meta_survives_and_wins():
    trace = concat_passes(
        pack([1], None, k=2, meta={"pass": "plan"}),
        pack([2], None, k=2),
        meta={"plan_char_end": 42},
    )
    assert trace.meta["plan_char_end"] == 42
    assert trace.meta["pass"] == "plan"


# ------------------------------------------------------------------- the arithmetic


def test_entropy_of_a_uniform_top_k_is_log_k():
    scalars = derive_scalars(pack([0], _rows(_flat(8)), k=8).topk_lp)
    assert scalars["entropy"][0] == pytest.approx(math.log(8), abs=1e-3)


def test_entropy_of_a_one_hot_distribution_is_zero():
    row = [[(0, 0.0)] + [(i, -30.0) for i in range(1, 8)]]
    scalars = derive_scalars(pack([0], row, k=8).topk_lp)
    assert scalars["entropy"][0] == pytest.approx(0.0, abs=1e-3)


def test_deepconf_scores_a_peaked_distribution_higher_than_a_flat_one():
    # THE sign test. DeepConf's C is a *mean negative logprob over the top-K*, so it
    # rises when the runners-up collapse -- the opposite direction from entropy. An
    # inverted deepconf_c would flip every downstream conclusion while looking fine.
    peaked = derive_scalars(pack([0], _rows(_peaked()), k=20).topk_lp)["deepconf_c"][0]
    flat = derive_scalars(pack([0], _rows(_flat()), k=20).topk_lp)["deepconf_c"][0]

    assert peaked > flat
    assert flat == pytest.approx(math.log(20), abs=1e-2)  # -(1/K) sum log(1/K) = log K


def test_deepconf_alone_cannot_tell_confident_from_diffuse():
    # The trap, pinned so nobody "simplifies" the scalar set down to deepconf_c. A
    # distribution smeared over the WHOLE vocabulary has top-K logprobs near log(1/V),
    # so its deepconf_c (11.9) beats a flat top-K's (3.0) despite being maximally
    # uncertain. tail_mass and self_cert are what disambiguate, so all three are stored.
    diffuse = _rows([1.0 / VOCAB] * 20)
    flat = _rows(_flat())
    d = derive_scalars(pack([0], diffuse, k=20).topk_lp, vocab_size=VOCAB)
    f = derive_scalars(pack([0], flat, k=20).topk_lp, vocab_size=VOCAB)

    assert d["deepconf_c"][0] > f["deepconf_c"][0]  # misleading on its own ...
    assert d["tail_mass"][0] > 0.99 and f["tail_mass"][0] < 0.01  # ... caught here ...
    assert d["self_cert"][0] < f["self_cert"][0]  # ... and ranked correctly here


def test_self_certainty_is_monotone_across_the_whole_confidence_range():
    # The one measure of the set that orders every case correctly, which is why it is
    # worth the vocab_size argument the others do not need.
    ladder = [
        [1.0 - 1e-9] + [1e-9 / 19] * 19,
        [0.99] + [0.01 / 19] * 19,
        [0.5, 0.2, 0.1, 0.05] + [0.15 / 16] * 16,
        _flat(),
        [1.0 / VOCAB] * 20,
    ]
    values = [
        derive_scalars(pack([0], _rows(p), k=20).topk_lp, vocab_size=VOCAB)["self_cert"][0]
        for p in ladder
    ]
    assert values == sorted(values, reverse=True)


def test_deepconf_averages_over_present_entries_not_over_k():
    # A row padded to K must not be dragged toward zero by entries it never had.
    short = derive_scalars(pack([0], [[(0, -2.0), (1, -4.0)]], k=20).topk_lp)
    assert short["deepconf_c"][0] == pytest.approx(3.0, abs=1e-3)


def test_tail_mass_measures_what_the_top_k_did_not_see():
    row = [[(0, math.log(0.5)), (1, math.log(0.2))]]  # 0.3 unaccounted for
    scalars = derive_scalars(pack([0], row, k=2).topk_lp)
    assert scalars["tail_mass"][0] == pytest.approx(0.3, abs=1e-3)


def test_entropy_renormalizes_so_truncation_does_not_read_as_confidence():
    # Two tokens at 0.5/0.2 have the same *shape* over the top-2 as 0.5/0.2 renormalized
    # to 0.714/0.286; entropy reports the latter, which is why tail_mass is stored too.
    scalars = derive_scalars(pack([0], [[(0, math.log(0.5)), (1, math.log(0.2))]], k=2).topk_lp)
    q = 0.5 / 0.7
    expected = -(q * math.log(q) + (1 - q) * math.log(1 - q))
    assert scalars["entropy"][0] == pytest.approx(expected, abs=1e-3)


def test_margin_is_p1_minus_p2():
    scalars = derive_scalars(pack([0], [[(0, math.log(0.6)), (1, math.log(0.25))]], k=2).topk_lp)
    assert scalars["margin"][0] == pytest.approx(0.35, abs=1e-3)


def test_surprisal_follows_the_sampled_token_not_the_argmax():
    topk = [[(7, math.log(0.9)), (3, math.log(0.1))]]
    trace = pack([3], topk, k=2)
    scalars = derive_scalars(trace.topk_lp, trace.sampled_lp)

    assert scalars["surprisal"][0] == pytest.approx(-math.log(0.1), abs=1e-3)
    assert scalars["top1_lp"][0] == pytest.approx(math.log(0.9), abs=1e-3)


def test_self_certainty_is_zero_for_a_uniform_full_vocabulary():
    # KL(P || U) = 0 when P *is* U. The tail is charged at maximum entropy, which for a
    # genuinely uniform distribution is exactly right -- so this case is tight, not a bound.
    k = 20
    row = [[(i, -math.log(VOCAB)) for i in range(k)]]
    scalars = derive_scalars(pack([0], row, k=k).topk_lp, vocab_size=VOCAB)
    assert scalars["self_cert"][0] == pytest.approx(0.0, abs=1e-2)


def test_self_certainty_is_near_log_vocab_for_a_one_hot_distribution():
    row = [[(0, 0.0)] + [(i, -40.0) for i in range(1, 20)]]
    scalars = derive_scalars(pack([0], row, k=20).topk_lp, vocab_size=VOCAB)
    assert scalars["self_cert"][0] == pytest.approx(math.log(VOCAB), abs=1e-2)


def test_self_certainty_is_omitted_rather_than_guessed_without_a_vocab_size():
    assert "self_cert" not in derive_scalars(pack([0], _rows(_flat()), k=20).topk_lp)


def test_scalars_are_finite_even_for_a_trace_with_no_logprobs_at_all():
    # An untraced backend must yield zeros, never NaNs that poison every later mean.
    trace = pack([1, 2, 3], None, k=20)
    scalars = derive_scalars(trace.topk_lp, trace.sampled_lp, vocab_size=VOCAB)

    for name, values in scalars.items():
        assert np.isfinite(values).all() or name == "top1_lp", name


def test_every_scalar_has_one_value_per_token():
    trace = pack([1, 2, 3], _rows(_flat(), _peaked(), _flat()), k=20)
    scalars = derive_scalars(trace.topk_lp, trace.sampled_lp, vocab_size=VOCAB)

    assert {v.shape for v in scalars.values()} == {(3,)}


def test_derive_scalars_rejects_a_one_dimensional_array():
    with pytest.raises(ValueError, match=r"2-D"):
        derive_scalars(np.array([-1.0, -2.0]))


# ---------------------------------------------------------------------- summarizing


def test_c_least_finds_the_worst_window_not_the_worst_token():
    # DeepConf's claim is that a trace fails on a confidently-wrong *stretch*; a single
    # low token inside an otherwise sure span must not dominate the trace's score.
    conf = np.array([10.0] * 20 + [1.0] * 4 + [10.0] * 20, dtype=np.float32)
    stats = summarize({"deepconf_c": conf}, window=4)

    assert stats["c_least"] == pytest.approx(1.0)
    assert stats["mean_deepconf_c"] == pytest.approx(float(conf.mean()), abs=1e-4)
    assert stats["n_tokens"] == 44


def test_summarize_handles_a_trace_shorter_than_the_window():
    stats = summarize({"deepconf_c": np.array([2.0, 4.0], dtype=np.float32)}, window=512)
    assert stats["c_least"] == pytest.approx(3.0)  # one window over the whole trace


def test_summarize_of_an_empty_trace_is_empty_not_an_exception():
    assert summarize({"deepconf_c": np.array([], dtype=np.float32)}) == {}


def test_one_token_with_no_alternatives_does_not_poison_the_whole_summary():
    # top1_lp is -inf for a row that held no logprobs at all (_topk yields an empty row
    # for a position vLLM returned as None). The per-token array says so honestly, but a
    # plain mean does not: one such token in a 100-token trace used to drag mean_top1_lp
    # to -inf, destroying the record's summary and writing -Infinity into attempts.jsonl.
    rows = [[(i * 3 + j, -0.1 - 0.5 * j) for j in range(5)] for i in range(100)]
    rows[42] = []
    ids = [row[0][0] if row else 999 for row in rows]

    scalars = derive_scalars(pack(ids, rows, k=5).topk_lp, vocab_size=VOCAB)
    stats = summarize(scalars, window=32)

    assert not np.isfinite(scalars["top1_lp"]).all()  # the array stays honest
    for name, value in stats.items():
        assert math.isfinite(value), f"{name} is {value}"
    # …and the record it lands in is plain JSON, with no -Infinity/NaN token in it.
    assert json.loads(json.dumps(stats)) == stats
    assert "Infinity" not in json.dumps(stats) and "NaN" not in json.dumps(stats)


def test_summarize_without_deepconf_still_reports_the_means_it_has():
    # The window statistics are all derived from deepconf_c; the per-name means are not.
    stats = summarize({"entropy": np.array([1.0, 3.0], dtype=np.float32)})
    assert stats == {"mean_entropy": 2.0}


def test_a_confidence_array_with_nothing_finite_yields_no_window_statistics():
    # derive_scalars cannot produce this -- deepconf_c is finite by construction -- but
    # summarize is public and takes the dict it is given. The window statistics are means
    # and a min over `conf`; with nothing finite left there is no honest number to report,
    # and _sliding_mean over an empty array would raise on groups.min().
    assert summarize({"deepconf_c": np.array([np.inf, -np.inf], dtype=np.float32)}) == {}


def test_a_scalar_that_is_never_finite_is_omitted_rather_than_faked():
    # No alternatives anywhere -- an untraced backend. A zero would read as "logprob 0",
    # i.e. probability 1, which is the most confident claim there is.
    scalars = derive_scalars(pack([1, 2, 3], None, k=20).topk_lp, vocab_size=VOCAB)
    stats = summarize(scalars)

    assert "mean_top1_lp" not in stats
    assert all(math.isfinite(v) for v in stats.values())


# ------------------------------------------------------------------------ round trip


def test_trace_round_trips_through_disk_without_allow_pickle(tmp_path):
    # np.load defaults to allow_pickle=False; these files outlive the code that wrote
    # them, so reading one must never be a trust decision.
    original = concat_passes(
        pack([1, 2], _rows(_peaked(), _flat()), k=20, meta={"model": "qwen"}),
        pack([3], _rows(_peaked()), k=20),
        meta={"plan_char_end": 17},
    )
    path = str(tmp_path / "trace.npz")
    write_trace(path, original)
    restored = read_trace(path)

    assert restored.token_ids.tolist() == original.token_ids.tolist()
    assert restored.seg.tolist() == original.seg.tolist()
    assert restored.sampled_rank.tolist() == original.sampled_rank.tolist()
    assert restored.meta == original.meta
    np.testing.assert_allclose(
        np.asarray(restored.topk_lp, dtype=np.float32),
        np.asarray(original.topk_lp, dtype=np.float32),
    )


# ------------------------------------------------------------------- calibration


#: Deliberately skewed. Temperature is a no-op on a uniform distribution -- sharpening
#: it just gives it back -- so a uniform fixture would make the T<1 arm below pass
#: whatever the code did.
SKEWED = [0.4, 0.3, 0.2, 0.1]


def _sampled_at(temperature: float, n: int, seed: int) -> tuple:
    """n draws from SKEWED sharpened by 1/T, packed against the UNSHARPENED record."""
    rng = np.random.default_rng(seed)
    sharpened = np.array(SKEWED) ** (1 / temperature)
    sharpened /= sharpened.sum()
    rows = [[(i, math.log(p)) for i, p in enumerate(SKEWED)] for _ in range(n)]
    trace = pack(list(rng.choice(4, size=n, p=sharpened)), rows, k=4)
    return rank1_calibration(trace.topk_lp, trace.sampled_rank)


def test_calibration_agrees_when_the_record_is_what_was_sampled_from():
    # The T=1.0 arm: the sampler drew from exactly the recorded distribution, so the
    # rank-1 rate must match the recorded p1. This is the reference.
    recorded, observed = _sampled_at(1.0, 20000, seed=0)

    assert recorded == pytest.approx(0.4, abs=1e-3)
    assert observed == pytest.approx(0.4, abs=0.02)


def test_calibration_detects_sampling_sharper_than_the_record():
    # The T<1 arm, and the whole point: the record still says p1=0.4 but the sampler
    # took rank 1 far more often, which can only happen if the record is PRE-temperature.
    # Were it processed, this ratio would read 1.00 and the tempering would be invisible
    # in every trace ever written.
    recorded, observed = _sampled_at(0.5, 20000, seed=1)

    assert recorded == pytest.approx(0.4, abs=1e-3)
    assert observed / recorded > 1.2


def test_calibration_of_an_empty_trace_is_nan_not_a_crash():
    trace = pack([], None, k=20)
    recorded, observed = rank1_calibration(trace.topk_lp, trace.sampled_rank)
    assert math.isnan(recorded) and math.isnan(observed)


def test_more_logprob_rows_than_tokens_are_ignored():
    # A backend that returns more rows than it sampled tokens must not overrun the
    # arrays -- pack stops at the token count.
    rows = _rows(_flat(4), _flat(4), _flat(4))
    trace = pack([1, 2], rows, k=4)  # 3 rows, 2 tokens
    assert len(trace) == 2


def test_a_single_token_trace_summarizes_without_a_sliding_window():
    # window degenerates to <= 1: the values are their own group.
    stats = summarize({"deepconf_c": np.array([4.0], dtype=np.float32)}, window=512)
    assert stats["c_least"] == pytest.approx(4.0)
    assert stats["n_tokens"] == 1
