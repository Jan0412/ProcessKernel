"""``PRMRolloutConfig``: the v2 campaign's knobs, and the ones that would spend the budget wrong."""

from __future__ import annotations

import dataclasses
import os

import pytest
import yaml

from reranker.src.config import PROJECT_ROOT, PRMRolloutConfig, RerankerConfig, _resolve, load_config

CONFIGS = os.path.join(PROJECT_ROOT, "configs")
FULL, SMOKE = "prm_rollout.yaml", "prm_rollout_smoke.yaml"
SHIPPED = [FULL, SMOKE]


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
    assert cfg.run_tags and cfg.rounds


@pytest.mark.parametrize("name", SHIPPED)
def test_the_shipped_baseline_is_where_the_config_says_it_is(name):
    # The one check that needs the cluster mount, kept separate so the rest of the suite
    # runs anywhere -- and the one that catches a moved timing file before job D grades
    # every measured rollout as ungradeable.
    path = _resolve(shipped(name).baseline_timing_json)
    if not os.path.isfile(path):
        pytest.skip(f"baseline not mounted here: {path}")
    shipped(name).validate()


def test_the_smoke_config_is_one_part_one_round():
    cfg = shipped(SMOKE)
    assert cfg.rounds == [0]
    assert cfg.parts_glob.endswith(".jsonl") and "*" not in cfg.parts_glob
    # A smoke campaign never lands on the real campaign's artifacts, or on its eval run.
    full = shipped(FULL)
    assert cfg.out_dir != full.out_dir and cfg.eval_run_name != full.eval_run_name


# Everything that decides *what* a prefix is and how its rollouts are graded. The smoke
# campaign is a sample of the real one, not a different one: if these drift, the thing
# verified at small scale is not the thing that then runs at full scale.
SHARED = [
    "source", "depths_per_group", "min_rel_depth", "max_rel_depth",
    "min_list_size", "max_list_size", "train_selection", "select_seed",
    "K", "min_rollouts", "temperature", "think_temperature", "max_new_tokens", "gen_model",
    "label_mode", "speedup_stat", "speedup_lo", "speedup_hi", "speed_quant",
    "baseline_timing_json", "num_correct_trials", "num_perf_trials",
]


def test_the_two_shipped_configs_select_and_grade_identically():
    full, smoke = shipped(FULL), shipped(SMOKE)
    assert [getattr(full, k) for k in SHARED] == [getattr(smoke, k) for k in SHARED]


@pytest.mark.parametrize("name", SHIPPED)
def test_every_selection_and_grading_knob_is_written_out_rather_than_inherited(name):
    # A campaign must be readable off its own config; a knob left to the dataclass default
    # changes silently when the default does.
    assert set(SHARED) | {"rounds", "run_tags", "parts_glob", "splits_json"} <= written_keys(name)


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


def test_a_string_where_the_round_list_is_required_raises():
    # config._coerce has no list case: `prm_rollout.rounds=[0]` on the CLI arrives as the
    # string "[0]" and would iterate as five characters, selecting no round at all.
    with pytest.raises(ValueError, match="prm_rollout.rounds"):
        rollout(rounds="[0]").validate()


@pytest.mark.parametrize("bad", [[], ["0"], [True]])
def test_the_round_list_must_hold_at_least_one_integer(bad):
    # bool is an int; a round number it is not, and YAML reads `[yes]` as one.
    with pytest.raises(ValueError, match="prm_rollout.rounds"):
        rollout(rounds=bad).validate()


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
