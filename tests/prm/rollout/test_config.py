"""``PRMRolloutConfig``: the v2 campaign's knobs, and the ones that would spend the budget wrong."""

from __future__ import annotations

import dataclasses
import os

import pytest
import yaml

from processkernel.config import PROJECT_ROOT, PRMRolloutConfig, RerankerConfig, _resolve, load_config

CONFIGS = os.path.join(PROJECT_ROOT, "configs")
FULL = "prm_rollout.yaml"
SHIPPED = [FULL]


def rollout(**over) -> PRMRolloutConfig:
    """A valid section, so each test changes exactly the one thing it is about."""
    # validate() requires the baseline on disk; any existing file satisfies that.
    base = PRMRolloutConfig(run_tags={"a_run": "ra"}, baseline_timing_json=__file__)
    return dataclasses.replace(base, **over)


def shipped(name: str) -> PRMRolloutConfig:
    return load_config(["--config", os.path.join(CONFIGS, name)]).prm_rollout


def written_keys(name: str) -> set[str]:
    """The knobs the file spells out, as opposed to the ones it inherits from the default."""
    with open(os.path.join(CONFIGS, name)) as f:
        return set(yaml.safe_load(f)["prm_rollout"])


def test_the_default_corpus_is_one_job_b_can_actually_read():
    # data/prm predates PLAN_v2 §8: its rows carry no system_prompt_sha1 and there is no
    # system_prompts.json beside them. Job A reads neither and succeeds; job B needs both and
    # dies a job later. The two knobs also have to name ONE build -- prefixes.py raises when
    # a problem has no split, and mismatched defaults are exactly how that happens.
    default = PRMRolloutConfig()
    assert default.parts_glob == "data/prm_v2/parts/*.jsonl"
    assert default.splits_json == "data/prm_v2/splits.json"
    assert os.path.dirname(os.path.dirname(default.parts_glob)) == os.path.dirname(
        default.splits_json
    )


def test_the_section_hangs_off_the_root_config_and_the_defaults_validate():
    assert isinstance(RerankerConfig().prm_rollout, PRMRolloutConfig)
    rollout().validate()


# --- the files that actually get used -------------------------------------------------


@pytest.mark.parametrize("name", SHIPPED)
def test_the_shipped_configs_load_and_validate(name):
    # _from_dict raises on an unknown key, so this also catches a typo'd knob in the YAML
    # -- which would otherwise leave the default in place and run something else.
    cfg = shipped(name)
    # The baseline lives outside the repo, so whether it is there is a property of the
    # machine and not of the file under test; swapping it keeps every other knob checked.
    dataclasses.replace(cfg, baseline_timing_json=__file__).validate()
    assert cfg.run_tags


@pytest.mark.parametrize("name", SHIPPED)
def test_the_shipped_baseline_is_where_the_config_says_it_is(name):
    # The one check that needs the cluster mount, kept separate so the rest of the suite
    # runs anywhere -- and the one that catches a moved timing file before job D grades
    # every measured rollout as ungradeable.
    path = _resolve(shipped(name).baseline_timing_json)
    if not os.path.isfile(path):
        pytest.skip(f"baseline not mounted here: {path}")
    shipped(name).validate()


# Everything that decides *what* a prefix is and how its rollouts are graded.
SHARED = [
    "source", "depths_per_group", "min_rel_depth", "max_rel_depth",
    "min_list_size", "max_list_size", "train_selection", "select_seed",
    "K", "min_rollouts", "temperature", "think_temperature", "max_new_tokens", "gen_model",
    "label_mode", "speedup_stat", "speedup_lo", "speedup_hi", "speed_quant",
    "baseline_timing_json", "num_correct_trials", "num_perf_trials",
]


@pytest.mark.parametrize("name", SHIPPED)
def test_every_selection_and_grading_knob_is_written_out_rather_than_inherited(name):
    # A campaign must be readable off its own config; a knob left to the dataclass default
    # changes silently when the default does.
    assert set(SHARED) | {"run_tags", "parts_glob", "splits_json"} <= written_keys(name)


def test_the_campaign_reads_the_build_of_prm_build_yaml():
    # One joint build feeds every campaign; a campaign on another build has another split.
    build = load_config(["--config", os.path.join(CONFIGS, "prm_build.yaml")]).prm.out_dir
    cfg = shipped(FULL)
    assert os.path.dirname(os.path.dirname(cfg.parts_glob)) == build
    assert os.path.dirname(cfg.splits_json) == build


def test_the_first_campaign_ships_with_random_selection():
    # §6, decided 2026-08-12: entropy and prm_spread are deferred, and a scored default
    # would leave no random arm to answer "did selection help?" against.
    assert shipped(FULL).train_selection == "random"
    assert shipped(FULL).source == "cut"


def test_an_unknown_prm_rollout_key_raises(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("prm_rollout:\n  max_list_sze: 8\n")
    with pytest.raises(KeyError, match="max_list_sze"):
        load_config(["--config", str(path)])


# --- validate(): the shapes a CLI override or a hand-edit can produce -----------------


def test_two_runs_sharing_one_tag_raises_rather_than_merging_their_lists():
    # The tag is the first component of `list_key`, so two runs under one tag put samples
    # from different generations in the same list -- N2, and invisible downstream.
    with pytest.raises(ValueError, match="run_tags"):
        rollout(run_tags={"a_run": "r", "b_run": "r"}).validate()


@pytest.mark.parametrize("bad", [{}, {"a_run": ""}, {"a_run": 7}, ["a_run"]])
def test_run_tags_must_name_at_least_one_run_with_a_usable_tag(bad):
    # Empty selects no run and the campaign builds nothing; an empty or non-string tag
    # lands in `list_key` and in `prefix_id`.
    with pytest.raises(ValueError, match="run_tags"):
        rollout(run_tags=bad).validate()


def test_a_ceiling_below_the_floor_raises():
    # Every list would be built, then capped below the floor, then dropped -- a campaign
    # that reads the whole corpus and emits nothing.
    with pytest.raises(ValueError, match="max_list_size"):
        rollout(min_list_size=4, max_list_size=3).validate()


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"min_list_size": 1}, "min_list_size"),   # a list of one has no pair to rank
        ({"min_list_size": 0}, "min_list_size"),
        ({"depths_per_group": 0}, "depths_per_group"),
        ({"min_rel_depth": 0.9, "max_rel_depth": 0.1}, "rel_depth"),
        ({"min_rel_depth": -0.1}, "rel_depth"),
        ({"max_rel_depth": 1.0}, "rel_depth"),     # rel_depth is (n-1)/n at the last cut
        ({"K": 0}, "prm_rollout.K"),
        ({"min_rollouts": 0}, "min_rollouts"),
        ({"K": 3, "min_rollouts": 4}, "min_rollouts"),  # a floor no run can reach
        ({"eval_shards": 0}, "eval_shards"),
        ({"depth_buckets": 0}, "depth_buckets"),
        ({"max_length": 0}, "max_length"),
        ({"num_workers": 0}, "num_workers"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        # Job B's slice of a unit: at 0 the driver would hand generate() nothing, forever.
        ({"prefixes_per_batch": 0}, "prefixes_per_batch"),
    ],
)
def test_a_knob_that_would_build_nothing_or_crash_raises(over, match):
    with pytest.raises(ValueError, match=match):
        rollout(**over).validate()


@pytest.mark.parametrize("bad", ["cutt", "beem", ""])
def test_an_unknown_prefix_source_raises(bad):
    with pytest.raises(ValueError, match="prm_rollout.source"):
        rollout(source=bad).validate()


def test_an_unknown_selection_mode_raises():
    with pytest.raises(ValueError, match="train_selection"):
        rollout(train_selection="entropy_v2").validate()


def test_the_two_deferred_selection_modes_still_validate():
    # Deferred, not foreclosed (§6): the names keep parsing so a later campaign needs no
    # config change, and prefixes.py is where the NotImplementedError lives.
    rollout(train_selection="entropy").validate()
    rollout(train_selection="prm_spread", prm_v1_checkpoint=__file__).validate()


def test_prm_spread_without_a_checkpoint_to_score_with_raises():
    with pytest.raises(ValueError, match="prm_v1_checkpoint"):
        rollout(train_selection="prm_spread").validate()


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"label_mode": "binry"}, "label_mode"),
        ({"speedup_stat": "minn"}, "speedup_stat"),
        ({"speedup_lo": 5.0}, "speedup_lo"),
        ({"speed_quant": 2.0}, "speed_quant"),
    ],
)
def test_a_grading_knob_is_checked_here_not_in_job_d(over, match):
    # targets.check_knobs is the same gate target_for uses, so the two cannot disagree.
    # Unchecked, a typo'd knob first raises in job D -- after every eval has been paid for.
    with pytest.raises(ValueError, match=match):
        rollout(**over).validate()


def test_a_baseline_file_that_is_not_there_raises():
    # Absent, job D grades every correct kernel as ungradeable and every V̂ comes out 0.
    with pytest.raises(ValueError, match="baseline_timing_json"):
        rollout(baseline_timing_json="/no/such/timing.json").validate()


def test_the_permissive_ends_of_those_ranges_are_legal():
    rollout(min_rel_depth=0.0, max_rel_depth=0.999, min_list_size=2, max_list_size=2).validate()
    rollout(depths_per_group=1, K=1, min_rollouts=1, num_workers=1, eval_shards=1).validate()
    rollout(prefixes_per_batch=1).validate()


# --- v3: ORM-imputed labelling -------------------------------------------------------


def _cfg(**kw):
    c = PRMRolloutConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_v3_defaults_match_the_orm_checkpoint():
    c = PRMRolloutConfig()
    assert c.label_source == "measured"          # v2 behaviour is the default
    assert c.orm_max_length == 6144              # listwise_base.yaml, not PLAN_v3's 4096
    assert c.orm_reserve_ref_tokens == 1024
    assert c.calib_speed_quant == 0.0            # what the ORM trained under
    # A ceiling, not a schedule: the damped fit halves its residual per pass and breaks
    # out when it converges -- level 1 stops at 14. 3 was chosen when the loop cycled
    # and the count only picked a phase.
    assert c.curve_bins == 40 and c.curve_iters == 16
    assert c.offset_kappa == "auto" and c.offset_clamp == 4.0
    assert c.use_anchors is True


def test_imputed_requires_a_checkpoint():
    with pytest.raises(ValueError, match="orm_checkpoint"):
        _cfg(label_source="imputed", orm_checkpoint=None).validate()


def test_label_source_is_closed():
    with pytest.raises(ValueError, match="label_source"):
        _cfg(label_source="both").validate()


def test_orm_budget_is_not_the_prm_budget():
    # Two models, two budgets. Equal by accident is the silent trap PLAN_v3 §7 names.
    c = PRMRolloutConfig()
    assert c.orm_max_length != c.max_length


# --- the single-pass regime (v6 native-thinking runs) ------------------------------------


def test_think_temperature_zero_means_one_pass_not_a_cold_plan_pass():
    # processkernel.generation.generate maps `--think-temperature 0` to SamplingSpec.think_temperature=None, i.e.
    # no plan pass at all. Here it stays a float, so the ONLY correct reading is > 0.
    assert rollout(think_temperature=0.0).two_pass is False
    assert rollout(think_temperature=1.0).two_pass is True


def test_zero_think_temperature_is_not_none_which_is_the_trap_two_pass_exists_for():
    # `think_temperature is not None` is True at 0.0, so a caller using it reads every v6 run
    # as two-pass and prefills a "## Plan" the source policy never wrote.
    c = rollout(think_temperature=0.0)
    assert (c.think_temperature is not None) is True
    assert c.two_pass is False


def test_native_thinking_with_a_plan_pass_is_refused():
    # The pair processkernel.generation.generate rejects at startup: both knobs open the assistant turn, so the
    # plan and the code land inside a <think> block the model never closes. No source run can
    # have been generated this way.
    with pytest.raises(ValueError, match="think_temperature"):
        _cfg(enable_thinking=True, think_temperature=1.0).validate()


def test_native_thinking_single_pass_is_the_v6_regime_and_is_accepted():
    rollout(enable_thinking=True, think_temperature=0.0).validate()


def test_the_tail_cut_defaults_are_the_flags_own_off_positions():
    # A run generated before --top-p/--top-k existed carries no key for them, and
    # check_gen_model compares against these values when the key is absent.
    c = PRMRolloutConfig()
    assert (c.enable_thinking, c.top_p, c.top_k) == (False, 1.0, 0)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_top_p_outside_the_unit_interval_is_refused(bad):
    with pytest.raises(ValueError, match="top_p"):
        _cfg(top_p=bad).validate()


def test_a_negative_top_k_is_refused():
    with pytest.raises(ValueError, match="top_k"):
        _cfg(top_k=-1).validate()
