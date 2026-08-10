"""``PRMConfig``: the shipped YAMLs parse, and a knob that would build the wrong thing raises."""

from __future__ import annotations

import dataclasses
import os

import pytest
import yaml

from reranker.src.config import PROJECT_ROOT, PRMConfig, _resolve, load_config

CONFIGS = os.path.join(PROJECT_ROOT, "configs")
FULL, SMOKE = "prm_config.yaml", "prm_smoke.yaml"
SHIPPED = [FULL, SMOKE]


def prm(**over) -> PRMConfig:
    """A valid section, so each test changes exactly the one thing it is about."""
    # validate() requires the baseline to be on disk; any existing file satisfies that.
    base = PRMConfig(run_dirs=["/runs/a"], baseline_timing_json=__file__)
    return dataclasses.replace(base, **over)


def shipped(name: str) -> PRMConfig:
    return load_config(["--config", os.path.join(CONFIGS, name)]).prm


def written_keys(name: str) -> set[str]:
    """The knobs the file spells out, as opposed to the ones it inherits from the default."""
    with open(os.path.join(CONFIGS, name)) as f:
        return set(yaml.safe_load(f)["prm"])


# --- the files that actually get used -------------------------------------------------


@pytest.mark.parametrize("name", SHIPPED)
def test_the_shipped_configs_load_and_validate(name):
    # _from_dict raises on an unknown key, so this also catches a typo'd knob in the YAML
    # -- which would otherwise leave the default in place and build something else.
    cfg = shipped(name)
    # The baseline lives outside the repo, so whether it is there is a property of the
    # machine and not of the file under test; swapping it keeps every other knob checked.
    dataclasses.replace(cfg, baseline_timing_json=__file__).validate()
    assert cfg.run_dirs and cfg.rounds


@pytest.mark.parametrize("name", SHIPPED)
def test_the_shipped_baseline_is_where_the_config_says_it_is(name):
    # The one check that needs the cluster mount, kept separate so the rest of the suite
    # runs anywhere. It is also the check that catches a moved timing file before a build
    # grades every correct kernel as ungradeable and writes an all-zero dataset.
    cfg = shipped(name)
    path = _resolve(cfg.baseline_timing_json)
    if not os.path.isfile(path):
        pytest.skip(f"baseline not mounted here: {path}")
    cfg.validate()


def test_the_smoke_config_is_one_run_one_round_one_shard():
    cfg = shipped(SMOKE)
    assert (len(cfg.run_dirs), cfg.rounds, cfg.max_shards) == (1, [0], 1)
    assert cfg.out_dir != shipped(FULL).out_dir  # a smoke build never lands on the real one


GRADING = [
    "label_mode", "speedup_stat", "speedup_lo", "speedup_hi", "speed_quant",
    "prose_lines_per_chunk", "code_steps_per_chunk", "min_frac",
    "tokenizer", "max_length", "baseline_timing_json",
]


def test_the_two_shipped_configs_grade_identically():
    # The smoke build is a sample of the real build, not a different one: if these drift, the
    # thing verified at small scale is not the thing that then runs at full scale.
    full, smoke = shipped(FULL), shipped(SMOKE)
    assert [getattr(full, k) for k in GRADING] == [getattr(smoke, k) for k in GRADING]


@pytest.mark.parametrize("name", SHIPPED)
def test_every_grading_knob_is_written_out_rather_than_inherited(name):
    # A build must be readable off its own config; a knob left to the dataclass default
    # changes silently when the default does.
    assert set(GRADING) <= written_keys(name)


def test_an_unknown_prm_key_raises(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("prm:\n  speedup_qant: 0.1\n")
    with pytest.raises(KeyError, match="speedup_qant"):
        load_config(["--config", str(path)])


# --- validate(): the shapes a CLI override or a hand-edit can produce -----------------


@pytest.mark.parametrize("name", ["run_dirs", "rounds"])
def test_a_string_where_a_list_is_required_raises(name):
    # config._coerce has no list case: `prm.rounds=[0]` on the CLI arrives as "[0]" and
    # would iterate as five characters, and `prm.run_dirs=/a` as a bare path string.
    with pytest.raises(ValueError, match=f"prm.{name}"):
        prm(**{name: "[0]"}).validate()


@pytest.mark.parametrize("name", ["run_dirs", "rounds"])
def test_an_empty_list_raises_rather_than_building_nothing(name):
    with pytest.raises(ValueError, match=f"prm.{name}"):
        prm(**{name: []}).validate()


def test_a_cli_override_of_a_list_is_caught_by_validate(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("prm:\n  run_dirs: [/runs/a]\n")
    cfg = load_config(["--config", str(path), "prm.rounds=[0]"])
    assert cfg.prm.rounds == "[0]"
    with pytest.raises(ValueError, match="prm.rounds"):
        cfg.prm.validate()


def test_a_scalar_cli_override_still_works(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("prm:\n  run_dirs: [/runs/a]\n")
    cfg = load_config(["--config", str(path), "prm.num_workers=1", "prm.max_shards=null"])
    assert (cfg.prm.num_workers, cfg.prm.max_shards) == (1, None)
    cfg.prm.validate()


def test_rounds_must_hold_integers():
    with pytest.raises(ValueError, match="prm.rounds"):
        prm(rounds=["0"]).validate()
    with pytest.raises(ValueError, match="prm.rounds"):
        prm(rounds=[True]).validate()  # bool is an int; a round number it is not


def test_run_dirs_must_hold_paths():
    with pytest.raises(ValueError, match="prm.run_dirs"):
        prm(run_dirs=[["/runs/a"]]).validate()


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"prose_lines_per_chunk": 0}, "chunk sizes"),
        ({"code_steps_per_chunk": 0}, "chunk sizes"),
        ({"min_frac": 1.0}, "min_frac"),
        ({"min_frac": -0.1}, "min_frac"),
        ({"max_length": 0}, "max_length"),
        ({"num_workers": 0}, "num_workers"),
        ({"max_shards": 0}, "max_shards"),
        ({"split_ratios": [0.85, 0.15]}, "split_ratios"),
        ({"split_ratios": [0.5, 0.3, 0.3]}, "split_ratios"),
        # Sums to 1 and has three entries, but n_train = round(1.5n) runs past the list
        # and _split_ids then puts every problem in train, with no val and no test.
        ({"split_ratios": [1.5, -0.5, 0.0]}, "split_ratios"),
        ({"split_ratios": "abc"}, "split_ratios"),  # len 3, and sum() would raise TypeError
        ({"split_ratios": [True, False, False]}, "split_ratios"),  # YAML [yes, no, no]
    ],
)
def test_a_knob_that_would_build_nothing_or_crash_raises(over, match):
    with pytest.raises(ValueError, match=match):
        prm(**over).validate()


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"label_mode": "binry"}, "label_mode"),
        ({"speedup_stat": "minn"}, "speedup_stat"),
        ({"speedup_lo": 5.0}, "speedup_lo"),
        ({"speedup_lo": 0.0}, "speedup_lo"),
        ({"speed_quant": -1.0}, "speed_quant"),
        ({"speed_quant": 2.0}, "speed_quant"),
    ],
)
def test_a_grading_knob_is_checked_here_not_inside_a_worker(over, match):
    # targets.check_knobs is the same gate target_for uses, so the two cannot disagree.
    # Unchecked, a typo'd knob first raises on the first graded row of each of 8 workers,
    # by which time part files are already on disk.
    with pytest.raises(ValueError, match=match):
        prm(**over).validate()


def test_a_baseline_file_that_is_not_there_raises(tmp_path):
    # Without it every correct kernel is ungradeable and the build still finishes, writing
    # a PRM training set in which every surviving row is 0.0.
    with pytest.raises(ValueError, match="baseline_timing_json"):
        prm(baseline_timing_json=str(tmp_path / "gone.json")).validate()


def test_binary_mode_does_not_need_a_speed_scale():
    # target_for never reads them in that mode, so validate must not either.
    prm(label_mode="binary", speedup_lo=5.0, speedup_hi=4.0, speed_quant=9.0).validate()


def test_a_float_max_shards_raises_rather_than_failing_on_the_slice():
    # `prm.max_shards=1e3` on the CLI coerces to a float, which passes `< 1` and then dies
    # in units() with "slice indices must be integers".
    with pytest.raises(ValueError, match="max_shards"):
        prm(max_shards=1e3).validate()


def test_the_permissive_ends_of_those_ranges_are_legal():
    prm(min_frac=0.0, max_shards=None, num_workers=1, max_length=1).validate()
    prm(min_frac=0.99, max_shards=1).validate()


def test_the_bare_defaults_are_not_buildable():
    # An empty run_dirs is the one default that cannot be right, so it must not be silent.
    with pytest.raises(ValueError, match="prm.run_dirs"):
        PRMConfig().validate()
