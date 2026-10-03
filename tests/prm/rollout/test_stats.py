"""``prm.rollout.stats``: the campaign diagnostics report (PLAN_v2 §6, §12).

The load-bearing number here is the within-list V̂ spread *against its own noise floor*.
§12 recorded that the fixed 0.15 threshold passes on pure noise -- the range of `n` noisy
draws grows with `n` even when every true value is identical -- so this module reports the
observed spread and a null bootstrapped from the campaign's own `scores`, and the margin
between them is what the gate reads.

CPU only, and deliberately torch-free: a diagnostics pass that needs a CUDA venv to run is
one nobody runs.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import random

import pytest

from processkernel.config import PRMRolloutConfig, RerankerConfig
from processkernel.prm.data import build
from processkernel.prm.rollout import calibrate, lists, prefixes, rollout, stage, stats, values


def cfg(**over) -> PRMRolloutConfig:
    base = PRMRolloutConfig(
        run_tags={"a_run": "ar"}, baseline_timing_json=__file__, K=4, min_rollouts=1
    )
    for k, v in over.items():
        setattr(base, k, v)
    base.validate()
    return base


def row(*rels: float, ids: tuple[str, ...] | None = None, **over) -> lists.ListRow:
    """One list, named by the relevances of its items -- `rel` is 2*V̂ (lists.py)."""
    names = ids or tuple(f"p{i}" for i in range(len(rels)))
    fields = dict(
        list_key="ar:2:37:20",
        run_tag="ar",
        level=2,
        problem_id=37,
        cut_index=20,
        rel_depth_mean=0.5,
        split="train",
        source="cut",
        items=[
            lists.Item(prefix_id=name, rel=rel, n_rollouts=4, se=0.1)
            for name, rel in zip(names, rels)
        ],
    )
    fields.update(over)
    return lists.ListRow(**fields)


# --- within-list V̂ spread ----------------------------------------------------------------


def test_spread_is_the_gap_between_the_best_and_worst_item_on_v_hats_scale():
    # rel = 2*V̂, so a list running rel 0.5 -> 1.5 is V̂ 0.25 -> 0.75: a spread of 0.5.
    assert stats.spread(row(0.5, 1.0, 1.5)) == 0.5


def test_a_list_whose_items_all_agree_has_no_spread():
    assert stats.spread(row(1.0, 1.0)) == 0.0


# --- the bootstrapped null: §12's noise floor ---------------------------------------------


def scored(*per_item: list[float]) -> dict[str, list[float]]:
    """``prefix_id -> scores``, matching the ids `row` gives its items."""
    return {f"p{i}": s for i, s in enumerate(per_item)}


def test_a_list_whose_every_rollout_scored_the_same_has_a_null_spread_of_zero():
    # Nothing to resample: the pool is one value, so every bootstrap V̂ is that value.
    got = stats.null_spreads(
        [row(1.0, 1.0)], scored([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        random.Random(0), replicates=8,
    )
    assert got == [0.0] * 8


def test_the_null_pools_within_a_list_and_never_across_lists():
    # Two lists, each internally flat, at opposite ends of the scale. Pooled campaign-wide
    # the null would draw from {0.0, 1.0} and report a large floor for both; pooled per list
    # -- the null §12 asks for, "every item of THIS list shares one true value" -- each list
    # sees only its own constant and every floor is zero.
    rows = [row(0.0, 0.0), row(2.0, 2.0, ids=("p2", "p3"))]
    by_id = {"p0": [0.0, 0.0], "p1": [0.0, 0.0], "p2": [1.0, 1.0], "p3": [1.0, 1.0]}
    got = stats.null_spreads(rows, by_id, random.Random(0), replicates=4)
    assert got == [0.0] * 8


def test_every_list_is_bootstrapped_every_replicate():
    rows = [row(0.0, 1.0), row(0.5, 1.5), row(1.0, 2.0)]
    got = stats.null_spreads(rows, scored([0.1, 0.9], [0.2, 0.8]), random.Random(0), replicates=5)
    assert len(got) == 15


class RecordingRandom:
    """A seam over `random.Random`, so "how many draws per item" is asserted, not inferred."""

    def __init__(self) -> None:
        self.ks: list[int] = []

    def choices(self, pool, k):
        self.ks.append(k)
        return [pool[0]] * k


def test_each_item_is_resampled_at_its_own_rollout_count():
    # The floor has to be the floor for *this* campaign's K. An item measured from 2
    # rollouts is noisier than one measured from 5, and drawing both at one count would
    # report a null for a campaign that was never run.
    rng = RecordingRandom()
    stats.null_spreads([row(0.0, 1.0)], scored([0.1, 0.2], [0.3] * 5), rng, replicates=2)
    assert rng.ks == [2, 5, 2, 5]


def test_the_same_seed_bootstraps_the_same_null_twice():
    rows, by_id = [row(0.0, 2.0)], scored([0.0, 0.5, 1.0], [0.2, 0.4, 0.9])
    assert stats.null_spreads(rows, by_id, random.Random(7), 20) == stats.null_spreads(
        rows, by_id, random.Random(7), 20
    )


def test_an_item_with_no_stored_scores_raises_rather_than_bootstrapping_a_thinner_null():
    with pytest.raises(KeyError, match="no per-rollout scores"):
        stats.null_spreads([row(0.0, 1.0)], scored([0.5, 0.5]), random.Random(0), 1)


# --- histograms ---------------------------------------------------------------------------


def test_the_v_hat_histogram_bins_every_item_over_the_unit_interval():
    # rel 0.0, 0.5, 1.0, 2.0 -> V̂ 0.0, 0.25, 0.5, 1.0.
    got = stats.value_histogram([row(0.0, 0.5, 1.0, 2.0)])
    assert got["0.0-0.1"] == 1
    assert got["0.2-0.3"] == 1
    assert got["0.5-0.6"] == 1
    assert sum(got.values()) == 4


def test_a_perfect_v_hat_lands_in_the_top_bin_rather_than_off_the_end():
    # The top bin is closed: V̂ = 1.0 is every rollout correct at full speed, not an overflow.
    assert stats.value_histogram([row(2.0, 2.0)])["0.9-1.0"] == 2


def test_every_bin_is_reported_including_the_ones_nothing_landed_in():
    # An absent bin and an empty one read the same in a report and mean opposite things.
    got = stats.value_histogram([row(0.0, 2.0)])
    assert len(got) == 10
    assert got["0.4-0.5"] == 0


def test_the_size_histogram_counts_lists_by_how_many_items_they_hold():
    rows = [row(0.0, 1.0), row(0.0, 1.0), row(0.0, 1.0, 2.0)]
    assert stats.size_histogram(rows) == {"2": 2, "3": 1}


def test_two_item_lists_are_called_out_separately_from_the_histogram():
    # §2: a corpus that is mostly pairs is a pairwise dataset wearing a listwise schema, and
    # a lone median list size would not show it.
    rows = [row(0.0, 1.0), row(0.0, 1.0), row(0.0, 1.0, 2.0), row(0.0, 0.5, 1.0, 2.0)]
    assert stats.two_item(rows) == 2


# --- depth slicing: §6 wants every headline number sliced, never averaged over depth -------


def test_each_band_holds_the_lists_whose_mean_depth_falls_in_it():
    rows = [row(0.0, 1.0, rel_depth_mean=d) for d in (0.15, 0.35, 0.55, 0.75)]
    bands = stats.depth_bands(rows, cfg(depth_buckets=4))
    assert [(b.lo, b.hi) for b in bands] == [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
    assert [len(b.rows) for b in bands] == [1, 1, 1, 1]


def test_a_band_no_list_landed_in_is_still_a_band():
    # §12 gates on "no empty bucket". A band dropped for being empty cannot fail that gate.
    bands = stats.depth_bands([row(0.0, 1.0, rel_depth_mean=0.15)], cfg(depth_buckets=4))
    assert len(bands) == 4
    assert [len(b.rows) for b in bands] == [1, 0, 0, 0]


def test_a_list_outside_the_window_is_counted_rather_than_absorbed_into_an_end_band():
    # Depth knobs that moved since job A enumerated. Clamping silently would report a metric
    # over lists the window says are not in the campaign.
    rows = [row(0.0, 1.0, rel_depth_mean=d) for d in (0.05, 0.5, 0.99)]
    assert stats.outside_window(rows, cfg()) == 2


def test_every_list_lands_in_exactly_one_band():
    rows = [row(0.0, 1.0, rel_depth_mean=d) for d in (0.1, 0.3, 0.5, 0.7, 0.9)]
    bands = stats.depth_bands(rows, cfg(depth_buckets=4))
    assert sum(len(b.rows) for b in bands) == len(rows)


# --- the funnel: attempted vs kept, per source or any other prefix attribute (§6) ----------


def pre(prefix_id="p0", **over) -> prefixes.Prefix:
    fields = dict(
        prefix_id=prefix_id,
        source="cut",
        run_name="a_run",
        run_tag="ar",
        shard="shard_00",
        level=2,
        problem_id=37,
        sample_id=0,
        stem="level_2_problem_37_sample_0_kernel",
        cut_char=10,
        cut_index=20,
        cut_kind="code",
        n_cuts_total=80,
        rel_depth=0.5,
        list_key="ar:2:37:20",
        split="train",
        selection="random",
        selection_score=None,
        K=4,
        min_rollouts=1,
    )
    fields.update(over)
    return prefixes.Prefix(**fields)


def test_a_slice_separates_the_prefixes_that_were_measured_from_the_ones_that_were_not():
    # Attempted vs kept is the whole point: a slice that enumerated 2 prefixes and measured
    # none is invisible in a total that only counts what survived.
    rows = [pre("p0"), pre("p1"), pre("p2", level=3), pre("p3", level=3)]
    got = stats.slice_counts(rows, {"p0", "p1", "p2"}, [], "level")
    assert got["2"]["prefixes"] == 2
    assert got["2"]["measured"] == 2
    assert got["3"] == {"prefixes": 2, "measured": 1, "lists": 0, "items": 0}


def test_a_slice_attributes_each_list_and_its_items_to_the_slice_it_came_from():
    rows = [pre("p0"), pre("p1", level=3)]
    listed = [row(0.0, 1.0), row(0.0, 1.0, 2.0, level=3)]
    got = stats.slice_counts(rows, {"p0", "p1"}, listed, "level")
    assert (got["2"]["lists"], got["2"]["items"]) == (1, 2)
    assert (got["3"]["lists"], got["3"]["items"]) == (1, 3)


def test_the_same_slice_works_over_source_because_stage_two_adds_a_second_one():
    rows = [pre("p0"), pre("p1", source="beam")]
    got = stats.slice_counts(rows, {"p0"}, [], "source")
    assert got["cut"]["measured"] == 1
    assert got["beam"]["measured"] == 0


# --- cost: dedup savings and the evals/GPU-hour the whole budget scales on (§6, §9) --------


def test_the_dedup_hit_rate_is_the_share_of_rollouts_that_reused_another_eval():
    assert stats.dedup_rate({"rollouts": 200, "deduped": 50}) == 25.0


def test_a_campaign_that_deduped_nothing_reports_zero_not_a_division_error():
    assert stats.dedup_rate({"rollouts": 0, "deduped": 0}) == 0.0


def test_evals_per_gpu_hour_divides_the_evals_by_the_gpu_time_that_bought_them():
    got = stats.evals_per_gpu_hour({"shards": [{"elapsed_s": 3600, "gpus": 1}]}, 318)
    assert got == 318.0


def test_a_shard_on_eight_gpus_spends_eight_gpu_hours_an_hour():
    # §12 leaves num_gpu_devices at 1 and flags the untaken 8x. Whoever takes it must not
    # have the throughput number silently multiply by 8 along with the allocation.
    got = stats.evals_per_gpu_hour({"shards": [{"elapsed_s": 3600, "gpus": 8}]}, 800)
    assert got == 100.0


def test_the_shards_of_one_campaign_are_summed_into_one_rate():
    acct = {"shards": [{"elapsed_s": 1800, "gpus": 1}, {"elapsed_s": 1800, "gpus": 1}]}
    assert stats.evals_per_gpu_hour(acct, 200) == 200.0


def test_no_accounting_file_reports_none_rather_than_a_made_up_rate():
    # §9's whole budget scales on this number; inventing one is worse than admitting it is
    # unrecorded, which is what the §12 check then fails on.
    assert stats.evals_per_gpu_hour(None, 318) is None


def test_accounting_that_records_no_gpu_time_raises_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="no GPU time"):
        stats.evals_per_gpu_hour({"shards": [{"elapsed_s": 0, "gpus": 1}]}, 318)


# --- V̂ vs continuation length: §6's guard against "longer is better" ----------------------


def roll(rollout_id="r0", prefix_id="p0", n_gen_tokens=100, truncation="ok"):
    return rollout.Rollout(
        rollout_id=rollout_id,
        prefix_id=prefix_id,
        j=0,
        continuation="x",
        code="y",
        code_sha1="z",
        n_prefix_tokens=10,
        n_gen_tokens=n_gen_tokens,
        finish_reason={"plan": None, "code": "stop"},
        truncation=truncation,
    )


def test_a_prefixs_length_is_the_mean_over_its_rollouts():
    rolls = [roll("r0", n_gen_tokens=100), roll("r1", n_gen_tokens=200)]
    assert stats.mean_lengths(rolls) == {"p0": 150.0}


def test_a_truncated_rollout_is_left_out_of_the_mean_length():
    # Every truncated rollout sits at max_new_tokens by construction, so counting them would
    # inflate the length of exactly the prefixes that had truncations -- biasing the one
    # correlation that exists to detect a length artifact.
    rolls = [roll("r0", n_gen_tokens=100), roll("r1", n_gen_tokens=16384, truncation="truncated")]
    assert stats.mean_lengths(rolls) == {"p0": 100.0}


def test_a_prefix_whose_every_rollout_was_truncated_has_no_length_at_all():
    assert stats.mean_lengths([roll("r0", truncation="truncated")]) == {}


def test_correlation_is_one_when_value_rises_with_length():
    assert stats.correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == 1.0


def test_correlation_is_minus_one_when_value_falls_with_length():
    assert stats.correlation([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == -1.0


def test_a_constant_series_has_no_correlation_rather_than_a_zero_one():
    # Undefined, not "uncorrelated": zero would read as "checked, and length does not matter".
    assert stats.correlation([1.0, 1.0, 1.0], [10.0, 20.0, 30.0]) is None


def test_a_single_prefix_cannot_be_correlated_with_anything():
    assert stats.correlation([1.0], [10.0]) is None


# --- V̂ vs the ORM's encoded sequence length: NOT code length (task 3/6's misnomer) --------


def test_mean_encoded_lengths_reads_n_code_tokens_not_generation_length():
    # n_gen_tokens (999) is deliberately way off n_code_tokens (10, 30) -- if this read the
    # wrong field the mean would come out near 999, not 20.
    rolls = [roll("r0", n_gen_tokens=999), roll("r1", n_gen_tokens=999)]
    orm_scores = {"r0": {"n_code_tokens": 10}, "r1": {"n_code_tokens": 30}}
    assert stats.mean_encoded_lengths(rolls, orm_scores) == {"p0": 20.0}


def test_mean_encoded_lengths_skips_rollouts_the_orm_never_scored():
    # A measured campaign never runs job B2 at all -- every rollout is unscored, and the
    # function must return {} rather than KeyError on the first lookup.
    rolls = [roll("r0"), roll("r1")]
    assert stats.mean_encoded_lengths(rolls, {"r0": {"n_code_tokens": 10}}) == {"p0": 10.0}
    assert stats.mean_encoded_lengths(rolls, {}) == {}


def test_mean_encoded_lengths_excludes_truncated_rollouts_like_mean_lengths_does():
    rolls = [roll("r0", n_gen_tokens=100), roll("r1", truncation="truncated")]
    orm_scores = {"r0": {"n_code_tokens": 10}, "r1": {"n_code_tokens": 9999}}
    assert stats.mean_encoded_lengths(rolls, orm_scores) == {"p0": 10.0}


# --- label_source: config and values.jsonl must agree (task 6 extra req.) -----------------


def test_check_label_source_passes_when_config_and_values_agree():
    assert stats.check_label_source("measured", [_value("p0", [0.0])]) == "measured"


def test_check_label_source_passes_on_an_empty_campaign():
    # Nothing to disagree with -- §12's own checks already gate on an empty campaign.
    assert stats.check_label_source("measured", []) == "measured"


def test_check_label_source_raises_on_a_mismatch():
    v = dataclasses.replace(_value("p0", [0.0]), label_source="imputed")
    with pytest.raises(ValueError, match="label_source"):
        stats.check_label_source("measured", [v])


# --- calibration diagnostics: read job D's files, never re-fit (N7) -----------------------


def test_calibration_summary_is_none_without_a_calibrate_manifest(tmp_path):
    assert stats.calibration_summary(str(tmp_path), [], "imputed") is None


def _write_calibration(out_dir, *, knots_x=(0.0, 1.0), resid_var=(0.02, 0.02),
                       offsets=None, orm_seen=4, orm_unseen=6, label_source="imputed",
                       fit=None):
    (out_dir / calibrate.CURVE).write_text(json.dumps({
        "knots_x": list(knots_x), "knots_y": [0.0, 1.0], "resid_var_by_band": list(resid_var),
        "n_by_band": [3, 3], "n_fit": 6,
        "target_dist": {"fit": {"n": 6, "mean": 0.4}, "all": {"n": 8, "mean": 0.5}},
    }))
    (out_dir / calibrate.OFFSETS).write_text(json.dumps({
        "meta": {"tau2": 0.1, "tau2_raw": 0.1, "tau2_estimable": True, "n_clamped": 0,
                 "n_problems": 1, "shrink_mean": 0.5},
        "offsets": offsets if offsets is not None else {"2:37": {"c": 0.1}},
    }))
    (out_dir / calibrate.CALIB_MANIFEST).write_text(json.dumps({
        "label_source": label_source, "anchors": {"orm_seen": orm_seen, "orm_unseen": orm_unseen},
        "fit": fit if fit is not None else {"converged": True, "n_iters": 3,
                                           "max_offset_delta": 0.001, "max_knot_delta": 0.002},
    }))


def test_calibration_summary_is_none_on_a_measured_campaign_that_also_ran_job_d(tmp_path):
    # The level-1 validation sweep, which is a real campaign that just ran: job D fits and
    # writes calibrate_manifest.json beside labels that came from evals, not from the curve.
    # Keying the block off the file existing described a curve that labelled nothing.
    _write_calibration(tmp_path, label_source="measured")
    assert stats.calibration_summary(str(tmp_path), [pre("p0")], "measured") is None


def test_calibration_summary_reports_the_curve_bands_offsets_and_orm_seen_split(tmp_path):
    _write_calibration(tmp_path)
    cal = stats.calibration_summary(str(tmp_path), [pre("p0"), pre("p1")], "imputed")
    assert cal["label_source"] == "imputed"
    assert cal["curve_flags"] == []
    assert cal["curve_bands"] == [
        {"band": 0, "score": 0.0, "target": 0.0, "n": 3},
        {"band": 1, "score": 1.0, "target": 1.0, "n": 3},
    ]
    assert cal["offsets"]["tau2"] == 0.1
    # Both p0 and p1 are level 2 problem 37 (this file's `pre()`), and offsets.json names
    # exactly that pkey -- no_anchor_rate is 0.0, not the campaign's raw problem count.
    assert cal["offsets"]["no_anchor_rate"] == 0.0
    assert cal["orm_seen_vs_unseen"] == {
        "n_seen": 4, "n_unseen": 6,
        "target_dist_fit_unseen_only": {"n": 6, "mean": 0.4},
        "target_dist_all": {"n": 8, "mean": 0.5},
    }


def test_calibration_summary_counts_a_campaign_problem_with_no_offset(tmp_path):
    _write_calibration(tmp_path, offsets={})   # no problem has an offset at all
    cal = stats.calibration_summary(str(tmp_path), [pre("p0")], "imputed")
    assert cal["offsets"]["no_anchor_rate"] == 100.0


def test_calibration_summary_flags_a_non_ascending_curve_rather_than_raising(tmp_path):
    _write_calibration(tmp_path, knots_x=(1.0, 0.0))
    cal = stats.calibration_summary(str(tmp_path), [], "imputed")
    assert any("ascending" in f for f in cal["curve_flags"])


def test_calibration_summary_flags_equal_adjacent_knots_too(tmp_path):
    # Same rule values.load_curve refuses on: a tie leaves one band unreachable by band().
    _write_calibration(tmp_path, knots_x=(1.0, 1.0))
    cal = stats.calibration_summary(str(tmp_path), [], "imputed")
    assert any("ascending" in f for f in cal["curve_flags"])


def test_calibration_summary_flags_an_all_zero_resid_var_rather_than_raising(tmp_path):
    _write_calibration(tmp_path, resid_var=(0.0, 0.0))
    cal = stats.calibration_summary(str(tmp_path), [], "imputed")
    assert any("resid_var_by_band" in f for f in cal["curve_flags"])


def test_calibration_summary_surfaces_whether_job_ds_fit_converged(tmp_path):
    # The real level-1 fit is converged=False at max_knot_delta 0.0296, stable over 40-80
    # iterations, and a campaign labelled from it must not look identical to a converged one.
    _write_calibration(tmp_path, fit={"converged": False, "n_iters": 80,
                                      "max_offset_delta": 0.02, "max_knot_delta": 0.0296})
    cal = stats.calibration_summary(str(tmp_path), [], "imputed")
    assert cal["fit"] == {"converged": False, "n_iters": 80,
                          "max_offset_delta": 0.02, "max_knot_delta": 0.0296}


def test_calibration_summary_reports_the_campaigns_clip_rate(tmp_path):
    # values.py counts these while labelling and used to discard them; the share of rollouts
    # landing outside the curve's fitted range is the distribution-shift signal.
    _write_calibration(tmp_path)
    _json(tmp_path / values.VALUES_MANIFEST,
          {"curve_lookups": 400, "clipped": 7, "clip_rate_pct": 1.75})
    cal = stats.calibration_summary(str(tmp_path), [], "imputed")
    assert cal["clipped"] == {"n": 7, "lookups": 400, "rate_pct": 1.75}


def test_report_surfaces_calibration_diagnostics_when_present(tmp_path):
    # A report has to stay legible even over a broken curve (values.load_curve REFUSES this
    # exact input before labelling anything -- see test_values.py); here it must flag, not
    # crash, and the rest of the report (spread, checks, ...) still comes back.
    conf = campaign(tmp_path, label_source="imputed")
    _write_calibration(tmp_path / "campaign", knots_x=(1.0, 0.0), resid_var=(0.0, 0.0))
    out = stats.report(conf)
    assert out["calibration"]["curve_flags"]
    assert out["checks"]   # the rest of the report was not aborted by the broken curve
    assert "encoded_seq_len_r" in out["length_correlation"]


def test_the_rendered_report_carries_the_calibration_block_when_present(tmp_path):
    conf = campaign(tmp_path, label_source="imputed")
    _write_calibration(tmp_path / "campaign")
    text = stats.render(stats.report(conf))
    assert "calibration" in text
    assert "encoded seq. length" in text


def test_the_rendered_report_shouts_when_the_fit_did_not_converge(tmp_path):
    # Not a field forty keys down a JSON blob: 1.1M labels come off this curve, so a reader
    # skimming the report has to trip over it.
    conf = campaign(tmp_path, label_source="imputed")
    _write_calibration(tmp_path / "campaign",
                       fit={"converged": False, "n_iters": 80, "max_offset_delta": 0.02,
                            "max_knot_delta": 0.0296})
    text = stats.render(stats.report(conf))
    assert "DID NOT CONVERGE" in text
    assert "!!" in text


def test_the_rendered_report_stays_quiet_when_the_fit_converged(tmp_path):
    conf = campaign(tmp_path, label_source="imputed")
    _write_calibration(tmp_path / "campaign")
    text = stats.render(stats.report(conf))
    assert "DID NOT CONVERGE" not in text
    assert "converged True" in text


def test_the_rendered_report_has_no_calibration_block_on_a_measured_campaign(tmp_path):
    # The manifest IS written here, which is the case that actually occurs (the level-1
    # validation sweep runs job D beside a measured campaign). Without it this test passed
    # against a fixture that could not have shown the bug.
    conf = campaign(tmp_path)
    _write_calibration(tmp_path / "campaign", label_source="measured")
    text = stats.render(stats.report(conf))
    assert "calibration (" not in text


# --- the pass: a whole campaign on disk -> the report and its §12 checks -------------------


def val_row(*rels, ids=None, **over) -> lists.ListRow:
    return row(*rels, ids=ids, split="val", **over)


def campaign(
    tmp_path,
    *,
    ps=None,
    train=None,
    val=None,
    scores=None,
    rollouts=(),
    accounting=None,
    values_dropped=None,
    lists_dropped=None,
    stage_over=None,
    label_source="measured",
    **over,
):
    """Every artifact jobs A-D write, laid out as they lay it out. Returns the config.

    ``label_source`` moves the config AND the ``values.jsonl`` rows together -- `report` cross-
    checks the two (`check_label_source`), so a fixture that moved only one would not build.
    """
    out = tmp_path / "campaign"
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    ps = [pre("p0"), pre("p1")] if ps is None else ps
    train = [row(0.0, 2.0)] if train is None else train
    val = [] if val is None else val
    scores = {"p0": [0.0, 0.0], "p1": [1.0, 1.0]} if scores is None else scores

    (out / prefixes.PREFIXES).write_text(
        "".join(json.dumps(dataclasses.asdict(p)) + "\n" for p in ps)
    )
    (out / values.VALUES).write_text(
        "".join(json.dumps(dataclasses.asdict(_value(pid, s, label_source))) + "\n"
                for pid, s in scores.items())
    )
    for split, rows in (("train", train), ("val", val)):
        (out / lists.LISTS.format(split=split)).write_text(
            "".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in rows)
        )
    with gzip.open(out / "rollouts" / "a_run__shard_00.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in rollouts))

    n_roll = len(rollouts) or 8
    _json(out / build.MANIFEST, {"created": "T", "git_sha": "abc", "git_dirty": False,
                                 "prefixes": len(ps)})
    _json(out / stage.STAGE_MANIFEST, {"rollouts": n_roll, "staged": n_roll, "deduped": 0,
                                       **(stage_over or {})})
    _json(out / values.VALUES_MANIFEST, {
        "prefixes": len(ps), "rollouts": n_roll, "evals": n_roll, "values": len(scores),
        "shared": 0,
        "dropped": {r: 0 for r in values.LEDGER} | (values_dropped or {}),
    })
    _json(out / lists.LISTS_MANIFEST, {
        "lists": {"train": len(train), "val": len(val)},
        "items": {"train": sum(len(r.items) for r in train),
                  "val": sum(len(r.items) for r in val)},
        "two_item_lists": {"train": stats.two_item(train), "val": stats.two_item(val)},
        "dropped": {r: 0 for r in lists.LEDGER} | (lists_dropped or {}),
    })
    if accounting is not None:
        _json(out / stats.ACCOUNTING, accounting)
    if label_source == "imputed":
        over.setdefault("orm_checkpoint", "ck")   # config.validate refuses imputed without one
    return RerankerConfig(prm_rollout=cfg(out_dir=str(out), label_source=label_source, **over))


def _value(prefix_id, scores, label_source="measured"):
    n, imputed = len(scores), label_source == "imputed"
    return values.Value(
        prefix_id=prefix_id, K_requested=4, n_rollouts=n,
        n_dropped={r: 0 for r in values.REASONS}, n_dedup_shared=0,
        # N1: the measured-only counters are null on an imputed row, never 0.
        n_compiled=None if imputed else n,
        n_correct=None if imputed else sum(1 for s in scores if s > 0),
        v_binary=None if imputed else 0.0,
        v_graded=sum(scores) / n,
        se_binary=None if imputed else 0.0,
        se_graded=None if imputed else 0.0,
        scores=scores, label_source=label_source,
        orm_offset=0.0 if imputed else None, se_imputed=0.1 if imputed else None,
    )


def _json(path, obj):
    path.write_text(json.dumps(obj))


def test_the_report_lands_beside_the_campaign_it_measured(tmp_path):
    conf = campaign(tmp_path)
    stats.report(conf)
    assert (tmp_path / "campaign" / stats.STATS).is_file()


def test_the_funnel_reads_prefixes_measured_and_listed_off_the_manifests(tmp_path):
    conf = campaign(tmp_path)
    out = stats.report(conf)
    assert out["pipeline"]["prefixes"] == 2
    assert out["pipeline"]["values"] == 2
    assert out["pipeline"]["lists"] == {"train": 1, "val": 0}


def test_the_spread_is_reported_against_a_null_bootstrapped_from_the_same_lists(tmp_path):
    # The whole point of §12's recalibration: a bare median clears 0.15 on noise alone, so
    # the number that decides stage 2 is the margin over the campaign's own floor.
    out = stats.report(campaign(tmp_path))
    assert out["spread"]["median"] == 1.0
    assert out["spread"]["null_median"] is not None
    assert out["spread"]["margin"] == out["spread"]["median"] - out["spread"]["null_q95"]


def only(out: dict, fragment: str) -> dict:
    """The one check whose name contains ``fragment``."""
    hits = [c for c in out["checks"] if fragment in c["check"]]
    assert len(hits) == 1, f"{fragment} matched {[c['check'] for c in hits]}"
    return hits[0]


def test_truncated_rollouts_over_two_percent_fail_their_gate(tmp_path):
    conf = campaign(tmp_path, values_dropped={"truncated": 3}, stage_over={"rollouts": 100})
    assert not only(stats.report(conf), "truncated")["ok"]


def test_truncated_rollouts_under_two_percent_pass(tmp_path):
    conf = campaign(tmp_path, values_dropped={"truncated": 1}, stage_over={"rollouts": 100})
    assert only(stats.report(conf), "truncated")["ok"]


def test_too_many_prefixes_dropped_for_too_few_rollouts_fails(tmp_path):
    # 2 prefixes, one of them dropped: 50%, ten times the gate.
    conf = campaign(tmp_path, values_dropped={"too_few_rollouts": 1})
    assert not only(stats.report(conf), "too_few_rollouts")["ok"]


def test_no_all_equal_lists_at_all_fails_the_band_from_below(tmp_path):
    # §12 expects 2-10%. Zero means the labels are not moving between the items of a list,
    # which is the same finding as "all of them are" and needs the same look.
    assert not only(stats.report(campaign(tmp_path)), "all-equal")["ok"]


def test_half_the_lists_dropped_all_equal_fails_the_band_from_above(tmp_path):
    conf = campaign(tmp_path, lists_dropped={"all_equal": 1})
    assert not only(stats.report(conf), "all-equal")["ok"]


def test_an_all_equal_rate_inside_the_band_passes(tmp_path):
    conf = campaign(tmp_path, train=[row(0.0, 2.0)] * 19, lists_dropped={"all_equal": 1})
    assert only(stats.report(conf), "all-equal")["ok"]


def test_a_campaign_short_of_three_hundred_lists_fails(tmp_path):
    assert not only(stats.report(campaign(tmp_path)), "usable lists")["ok"]


def test_a_train_split_prefix_on_a_val_list_fails_n3(tmp_path):
    # lists.py takes the split from splits.json, not from the prefix. If the two disagree a
    # train prefix lands on a val list and the v1-PRM ranking number is measured on data the
    # selector saw. Nothing in either file can notice on its own.
    conf = campaign(
        tmp_path,
        ps=[pre("p0", split="train"), pre("p1", split="val")],
        train=[],
        val=[val_row(0.0, 2.0)],
    )
    assert not only(stats.report(conf), "N3")["ok"]


def test_a_val_list_of_val_prefixes_passes_n3(tmp_path):
    conf = campaign(
        tmp_path,
        ps=[pre("p0", split="val"), pre("p1", split="val")],
        train=[],
        val=[val_row(0.0, 2.0)],
    )
    assert only(stats.report(conf), "N3")["ok"]


def test_a_list_whose_prefixes_were_cut_at_two_depths_fails_n4(tmp_path):
    conf = campaign(tmp_path, ps=[pre("p0", cut_index=20), pre("p1", cut_index=21)])
    assert not only(stats.report(conf), "N4")["ok"]


def test_an_empty_depth_bucket_fails_the_coverage_gate(tmp_path):
    conf = campaign(tmp_path, depth_buckets=4)
    assert not only(stats.report(conf), "depth bucket")["ok"]


def test_a_list_outside_the_depth_window_is_caught(tmp_path):
    conf = campaign(tmp_path, train=[row(0.0, 2.0, rel_depth_mean=0.99)], depth_buckets=1)
    assert not only(stats.report(conf), "depth window")["ok"]


def test_lists_that_only_spread_as_much_as_their_own_noise_fail_the_signal_gate(tmp_path):
    # Two items measured from the same pool of rollout scores. They differ -- so the list is
    # not all-equal and survives job D -- but by no more than resampling that pool produces.
    conf = campaign(
        tmp_path,
        train=[row(0.0, 1.0)],
        scores={"p0": [0.0, 0.0, 1.0], "p1": [0.0, 1.0, 1.0]},
        depth_buckets=1,
    )
    assert not only(stats.report(conf), "spread")["ok"]


def test_lists_that_spread_far_past_their_noise_pass_the_signal_gate(tmp_path):
    # Every rollout of p0 failed and every rollout of p1 succeeded: the pooled null can
    # reproduce that split only by drawing one item all-zero and the other all-one.
    conf = campaign(
        tmp_path,
        train=[row(0.0, 2.0)] * 5,
        scores={"p0": [0.0] * 8, "p1": [1.0] * 8},
        depth_buckets=1,
    )
    assert only(stats.report(conf), "spread")["ok"]


def test_a_campaign_that_did_not_record_its_gpu_time_fails(tmp_path):
    assert not only(stats.report(campaign(tmp_path)), "GPU-hour")["ok"]


def test_recorded_gpu_time_passes_and_lands_in_the_report(tmp_path):
    conf = campaign(tmp_path, accounting={"shards": [{"elapsed_s": 3600, "gpus": 1}]})
    out = stats.report(conf)
    assert only(out, "GPU-hour")["ok"]
    assert out["evals_per_gpu_hour"] == 8.0


def clean(tmp_path, **over):
    """A campaign that clears every §12 gate -- the shape the full run is aiming at."""
    ps = [pre("p0"), pre("p1"), pre("p2", split="val"), pre("p3", split="val")]
    return campaign(
        tmp_path,
        ps=ps,
        train=[row(0.0, 2.0)] * 300,
        val=[val_row(0.0, 2.0, ids=("p2", "p3"))] * 60,
        scores={"p0": [0.0] * 8, "p1": [1.0] * 8, "p2": [0.0] * 8, "p3": [1.0] * 8},
        lists_dropped={"all_equal": 10},
        accounting={"shards": [{"elapsed_s": 3600, "gpus": 1}]},
        depth_buckets=1,
        **over,
    )


def test_a_campaign_that_clears_every_gate_reports_no_failure(tmp_path):
    out = stats.report(clean(tmp_path))
    assert [c["check"] for c in out["checks"] if not c["ok"]] == []


def test_the_rendered_report_carries_the_spread_against_its_null(tmp_path):
    text = stats.render(stats.report(clean(tmp_path)))
    assert "null" in text
    assert "margin" in text


def test_the_rendered_report_marks_every_check_pass_or_fail(tmp_path):
    out = stats.report(campaign(tmp_path))
    text = stats.render(out)
    assert text.count("[PASS]") + text.count("[FAIL]") == len(out["checks"])
    assert "[FAIL]" in text


def test_main_exits_non_zero_when_a_gate_fails(tmp_path, monkeypatch, capsys):
    conf = campaign(tmp_path)
    monkeypatch.setattr(stats, "load_config", lambda argv: conf)
    with pytest.raises(SystemExit) as exc:
        stats.main([])
    assert exc.value.code == 1
    assert "[FAIL]" in capsys.readouterr().out


def test_main_returns_quietly_when_the_campaign_is_clean(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stats, "load_config", lambda argv: clean(tmp_path))
    stats.main([])
    assert "[FAIL]" not in capsys.readouterr().out


def test_a_directory_that_is_not_a_campaign_names_the_job_that_would_make_it_one(tmp_path):
    # Pointed at an empty out_dir the report would otherwise measure nothing and pass.
    conf = RerankerConfig(prm_rollout=cfg(out_dir=str(tmp_path)))
    with pytest.raises(FileNotFoundError, match="rollout.prefixes"):
        stats.report(conf)


# --- the signal gate: each list against its OWN null, not against a pooled one -------------


def test_a_list_that_spreads_past_its_own_noise_floor_counts_as_signal():
    # Every rollout of p0 failed, every rollout of p1 succeeded. Resampling the pooled
    # {0*8, 1*8} reproduces that split only by drawing 8 zeros and 8 ones -- p < 0.4%.
    rows = [row(0.0, 2.0)]
    scores = {"p0": [0.0] * 8, "p1": [1.0] * 8}
    assert stats.beats_own_null(rows, scores, random.Random(0), 200) == (1, 1)


def test_a_list_that_only_spreads_as_much_as_its_own_noise_does_not():
    rows = [row(0.0, 1.0)]
    scores = {"p0": [0.0, 0.0, 1.0], "p1": [0.0, 1.0, 1.0]}
    assert stats.beats_own_null(rows, scores, random.Random(0), 200) == (0, 1)


def test_the_share_is_counted_over_every_list_not_only_the_ones_that_beat_it():
    rows = [row(0.0, 2.0), row(0.0, 1.0, ids=("p2", "p3"))]
    scores = {"p0": [0.0] * 8, "p1": [1.0] * 8, "p2": [0.0, 1.0], "p3": [0.0, 1.0]}
    assert stats.beats_own_null(rows, scores, random.Random(0), 200) == (1, 2)


def test_each_list_is_held_against_its_own_width_not_against_a_pooled_null():
    # The defect this replaced: a null pooled across widths has its q95 set by the widest
    # lists while the observed median is set by the narrowest, so the two never compare.
    # A wide all-noise list must not lift the floor a narrow real one is judged against.
    narrow = row(0.0, 2.0)
    wide = row(*[1.0] * 8, ids=tuple(f"w{i}" for i in range(8)))
    scores = {"p0": [0.0] * 8, "p1": [1.0] * 8} | {f"w{i}": [0.0, 1.0] * 4 for i in range(8)}
    assert stats.beats_own_null([narrow, wide], scores, random.Random(0), 200)[0] == 1


def test_the_signal_gate_reads_the_share_and_not_the_pooled_margin(tmp_path):
    # A campaign with real per-item differences must pass, and the pooled-margin gate this
    # replaced failed it: measured -0.400 on corpora with and without signal alike.
    conf = campaign(
        tmp_path,
        train=[row(0.0, 2.0)] * 300,
        scores={"p0": [0.0] * 8, "p1": [1.0] * 8},
        depth_buckets=1,
    )
    out = stats.report(conf)
    assert out["spread"]["beats_null_pct"] == 100.0
    assert only(out, "spread")["ok"]


def test_a_campaign_whose_lists_only_carry_noise_fails_the_signal_gate(tmp_path):
    conf = campaign(
        tmp_path,
        train=[row(0.0, 1.0)] * 300,
        scores={"p0": [0.0, 0.0, 1.0], "p1": [0.0, 1.0, 1.0]},
        depth_buckets=1,
    )
    out = stats.report(conf)
    assert out["spread"]["beats_null_pct"] == 0.0
    assert not only(out, "spread")["ok"]


def test_a_lists_null_is_bootstrapped_from_that_list_alone_and_nothing_else():
    # Structural, because the statistical form of this test does not discriminate: a
    # campaign-wide q95 still exceeds a flat list's spread often enough to look right.
    # Counting the draws does discriminate. Per list it is replicates * items; bootstrapping
    # every list for every list is len(rows) times that -- quadratic, and it judges a list
    # against widths it shares nothing with, which is the defect this gate was rewritten to
    # remove.
    rng = RecordingRandom()
    rows = [row(0.0, 2.0), row(0.0, 2.0, ids=("p2", "p3"))]
    scores = {"p0": [0.0], "p1": [1.0], "p2": [0.0], "p3": [1.0]}
    stats.beats_own_null(rows, scores, rng, 3)
    assert len(rng.ks) == 2 * 3 * 2


def test_quantile_takes_the_nearest_rank_rather_than_interpolating():
    # Nearest rank keeps the reported floor a value the null actually produced, which
    # matters when V-hat is coarse: at K=5 it only takes values in {0, .2, .4, .6, .8, 1}.
    xs = [float(i) for i in range(100)]
    assert (stats.quantile(xs, 0.5), stats.quantile(xs, 0.95)) == (50.0, 95.0)
    assert stats.quantile([3.0], 0.95) == 3.0


def test_the_reported_null_floor_is_the_high_quantile_not_the_middle_of_the_null(tmp_path):
    # Two items of two rollouts each over a pooled {0,0,1,1}: the null spread is 0 in 37.5%
    # of draws, 0.5 in 50% and 1.0 in 12.5%. Median 0.5, q95 1.0. Reporting the median under
    # the "q95" name would understate the floor a reader judges the campaign against.
    sp = stats.report(campaign(tmp_path, depth_buckets=1))["spread"]
    assert sp["null_q95"] > sp["null_median"]
