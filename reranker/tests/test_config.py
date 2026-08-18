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


def based(tmp_path, base_text: str, text: str) -> str:
    (tmp_path / "base.yaml").write_text(base_text)
    return written(tmp_path, "_base: base.yaml\n" + text)


def test_a_base_supplies_what_the_config_leaves_out(tmp_path):
    cfg = load_config(["--config", based(tmp_path, "train:\n  epochs: 9\n", "train:\n  lr: 0.5\n")])
    assert (cfg.train.epochs, cfg.train.lr) == (9, 0.5)


def test_a_config_wins_over_its_base_key_by_key(tmp_path):
    text = "train:\n  epochs: 1\n"
    cfg = load_config(["--config", based(tmp_path, "train:\n  epochs: 9\n  seed: 7\n", text)])
    assert (cfg.train.epochs, cfg.train.seed) == (1, 7)


def test_a_list_replaces_rather_than_extends_the_base_one(tmp_path):
    # run_dirs is the reason: a single-run variant must not inherit the other run.
    base = "data:\n  run_dirs: [a, b]\n"
    cfg = load_config(["--config", based(tmp_path, base, "data:\n  run_dirs: [c]\n")])
    assert cfg.data.run_dirs == ["c"]


def test_a_base_is_resolved_next_to_the_file_that_names_it(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "base.yaml").write_text("train:\n  epochs: 9\n")
    leaf = tmp_path / "sub" / "leaf.yaml"
    leaf.write_text("_base: base.yaml\n")
    assert load_config(["--config", str(leaf)]).train.epochs == 9


def test_a_base_may_itself_have_a_base(tmp_path):
    (tmp_path / "a.yaml").write_text("train:\n  epochs: 9\n  seed: 7\n")
    (tmp_path / "b.yaml").write_text("_base: a.yaml\ntrain:\n  seed: 8\n")
    cfg = load_config(["--config", based(tmp_path, "_base: b.yaml\n", "train:\n  lr: 0.5\n")])
    assert (cfg.train.epochs, cfg.train.seed, cfg.train.lr) == (9, 8, 0.5)


def test_a_base_cycle_raises_instead_of_recursing_forever(tmp_path):
    (tmp_path / "a.yaml").write_text("_base: c.yaml\n")
    (tmp_path / "c.yaml").write_text("_base: a.yaml\n")
    with pytest.raises(ValueError, match="_base cycle"):
        load_config(["--config", str(tmp_path / "a.yaml")])


def test_an_override_beats_the_base_and_the_config(tmp_path):
    path = based(tmp_path, "train:\n  epochs: 9\n", "train:\n  epochs: 1\n")
    assert load_config(["--config", path, "train.epochs=3"]).train.epochs == 3


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


SHIPPED = sorted(f for f in os.listdir(os.path.join(PROJECT_ROOT, "configs")) if f.endswith(".yaml"))


@pytest.mark.parametrize("name", SHIPPED)
def test_no_shipped_config_grades_against_a_different_baseline(name):
    # Every config in the dir, found by listing it: a variant added later inherits the check
    # instead of being remembered into a hand-written list.
    cfg = load_config(["--config", os.path.join(PROJECT_ROOT, "configs", name)])
    section = cfg.prm if name.startswith("prm") else cfg.data
    assert _resolve(section.baseline_timing_json) == BASELINE_TIMING_JSON


def test_the_shipped_listwise_variants_write_to_disjoint_paths():
    # Six datasets built from one base: a copy-pasted output path would have one build
    # silently overwrite another's dataset, lists, or checkpoints.
    seen: dict[str, str] = {}
    for name in (n for n in SHIPPED if n.startswith("listwise_kb_")):
        cfg = load_config(["--config", os.path.join(PROJECT_ROOT, "configs", name)])
        for path in (
            cfg.data.dataset_jsonl,
            cfg.data.splits_json,
            cfg.train.output_dir,
            cfg.listwise.lists_train_jsonl,
            cfg.listwise.lists_val_jsonl,
            cfg.listwise.lists_splits_json,
        ):
            assert path not in seen, f"{name} and {seen[path]} both write {path}"
            seen[path] = name
    assert len(seen) == 6 * 6


def test_the_shipped_listwise_variants_share_every_training_knob():
    cfgs = [
        load_config(["--config", os.path.join(PROJECT_ROOT, "configs", n)])
        for n in SHIPPED
        if n.startswith("listwise_kb_")
    ]
    assert len(cfgs) == 6
    for field_name in ("epochs", "lr", "seed", "metric_for_best_model", "max_steps"):
        assert len({getattr(c.train, field_name) for c in cfgs}) == 1
    for field_name in ("sigma", "loss_alpha", "list_size", "speedup_lo", "speedup_hi", "list_seed"):
        assert len({getattr(c.listwise, field_name) for c in cfgs}) == 1
    assert len({c.model.base_model for c in cfgs}) == 1


def test_the_mlflow_uri_is_sqlite_under_the_project_root():
    assert MLflowConfig().tracking_uri() == "sqlite:///" + os.path.join(PROJECT_ROOT, "mlflow.db")
