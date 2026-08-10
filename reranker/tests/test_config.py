"""``reranker.src.config``: the loader every pipeline shares, and its CLI override rules."""

from __future__ import annotations

import os

import pytest

from reranker.src.config import (
    BASELINE_TIMING_JSON,
    PROJECT_ROOT,
    DataConfig,
    MLflowConfig,
    PRMConfig,
    RerankerConfig,
    _resolve,
    load_config,
    to_flat_dict,
)


def written(tmp_path, text: str) -> str:
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return str(path)


def test_an_absolute_path_is_left_alone_and_a_relative_one_lands_under_the_project():
    assert _resolve("/tmp/x.json") == "/tmp/x.json"
    assert _resolve("data/x.json") == os.path.join(PROJECT_ROOT, "data/x.json")


def test_an_empty_config_file_gives_the_defaults(tmp_path):
    cfg = load_config(["--config", written(tmp_path, "")])
    assert cfg == RerankerConfig()


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("train.bf16=false", False),
        ("train.bf16=TRUE", True),
        ("train.pos_weight=null", None),
        ("train.pos_weight=None", None),
        ("train.epochs=7", 7),
        ("train.lr=1e-4", 1e-4),
        ("model.head_type=yes_no_lm", "yes_no_lm"),
    ],
)
def test_an_override_is_coerced_to_the_type_it_looks_like(tmp_path, override, expected):
    cfg = load_config(["--config", written(tmp_path, ""), override])
    section, _, leaf = override.partition("=")[0].partition(".")
    assert getattr(getattr(cfg, section), leaf) == expected


def test_an_override_of_a_key_that_does_not_exist_raises(tmp_path):
    with pytest.raises(KeyError, match="train.epocs"):
        load_config(["--config", written(tmp_path, ""), "train.epocs=1"])


def test_an_override_without_an_equals_sign_raises(tmp_path):
    with pytest.raises(ValueError, match="key=value"):
        load_config(["--config", written(tmp_path, ""), "train.epochs"])


def test_flattening_dots_the_nesting_and_joins_lists():
    flat = to_flat_dict(RerankerConfig())
    assert flat["train.epochs"] == 3
    assert flat["data.split_ratios"] == "0.7,0.15,0.15"
    assert flat["prm.rounds"] == "0,1,2"


def test_a_scalar_level_is_broadcast_over_the_run_dirs():
    cfg = DataConfig(run_dirs=["a", "b"], level=2)
    assert cfg.levels_for_run_dirs() == [2, 2]


def test_a_level_list_must_be_as_long_as_the_run_dirs():
    assert DataConfig(run_dirs=["a", "b"], level=[1, 3]).levels_for_run_dirs() == [1, 3]
    with pytest.raises(ValueError, match="2 entries"):
        DataConfig(run_dirs=["a"], level=[1, 3]).levels_for_run_dirs()


def test_every_pipeline_grades_against_the_one_baseline():
    # The ORM and the PRM divide by this number, so two paths means the same kernel can be
    # 1.2x in one dataset and 1.0x in the other. They diverged once already: data/ pointed at
    # A100 and prm/ at H100, and the two H100 files on this machine are different
    # measurements -- 15,823 of 16,311 shared problems disagree on `mean`.
    assert DataConfig().baseline_timing_json == BASELINE_TIMING_JSON
    assert PRMConfig().baseline_timing_json == BASELINE_TIMING_JSON


@pytest.mark.parametrize("name", ["listwise_config.yaml", "prm_config.yaml", "prm_smoke.yaml"])
def test_no_shipped_config_grades_against_a_different_baseline(name):
    cfg = load_config(["--config", os.path.join(PROJECT_ROOT, "configs", name)])
    section = cfg.prm if name.startswith("prm") else cfg.data
    assert _resolve(section.baseline_timing_json) == BASELINE_TIMING_JSON


def test_the_mlflow_uri_is_sqlite_under_the_project_root():
    assert MLflowConfig().tracking_uri() == "sqlite:///" + os.path.join(PROJECT_ROOT, "mlflow.db")
