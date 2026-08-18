"""``PRMTrainConfig`` and the cross-section budget guard (PLAN_TRAINER §6)."""

from __future__ import annotations

import os

import pytest

from reranker.src.config import PROJECT_ROOT, RerankerConfig, check_prm_budgets, load_config

CONFIGS = os.path.join(PROJECT_ROOT, "configs")

# The campaign knobs the trainer reads out of `prm_rollout`. The arms inherit them through
# `_base: prm_rollout.yaml` rather than copying them, so this is checked as inheritance
# instead of as a parity test over a duplicate.
CAMPAIGN_KNOBS = (
    "out_dir", "parts_glob", "max_length", "depth_buckets", "min_rel_depth", "max_rel_depth",
)

# The backbone arms. Adding one here is what puts it under every test below.
ARMS = ("prm_train_qwen3base06b.yaml", "prm_train_qwen25coder05b.yaml")


def paired(prm_model="Qwen/Qwen3-Reranker-4B", orm_model="Qwen/Qwen3-Reranker-4B"):
    """A root config whose two backbone names are set explicitly, so each test moves one."""
    cfg = RerankerConfig()
    cfg.prm_rollout.base_model = prm_model
    cfg.model.base_model = orm_model
    return cfg


def test_the_section_hangs_off_the_root_config_with_the_two_lambdarank_knobs():
    # Two knobs, because two is all that is new: everything else the trainer needs is read
    # from prm_rollout / model / train rather than copied into a third place.
    prm_train = RerankerConfig().prm_train
    assert prm_train.sigma == 1.0
    assert prm_train.loss_alpha == 0.5
    prm_train.validate()


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_a_non_positive_sigma_is_rejected(sigma):
    # sigma scales the score difference inside softplus(-sigma * (s_i - s_j)). At 0 every pair
    # contributes softplus(0) whatever the model does, so the loss is a constant and training
    # is a no-op that still reports a falling-nothing curve.
    cfg = RerankerConfig()
    cfg.prm_train.sigma = sigma
    with pytest.raises(ValueError, match="prm_train.sigma"):
        cfg.prm_train.validate()


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_a_loss_alpha_outside_the_unit_interval_is_rejected(alpha):
    # alpha and 1 - alpha weight the two pair groups. Outside [0, 1] one group gets a negative
    # weight, so the loss is minimized by ranking that group *wrong*.
    cfg = RerankerConfig()
    cfg.prm_train.loss_alpha = alpha
    with pytest.raises(ValueError, match="prm_train.loss_alpha"):
        cfg.prm_train.validate()


def test_the_endpoints_of_loss_alpha_are_allowed():
    # 0 and 1 are the ablations -- all weight on speed pairs, all weight on correctness pairs.
    for alpha in (0.0, 1.0):
        cfg = RerankerConfig()
        cfg.prm_train.loss_alpha = alpha
        cfg.prm_train.validate()


# --- the eval cap ------------------------------------------------------------------------


def test_the_eval_cap_defaults_to_measuring_every_list():
    # 0 = no cap. The default has to be "measure everything" so that adding the knob changes
    # no existing run's number; capping is opt-in per config.
    prm_train = RerankerConfig().prm_train
    assert prm_train.eval_max_lists == 0
    assert prm_train.eval_subsample_seed == 42
    prm_train.validate()


def test_a_negative_eval_cap_is_rejected():
    cfg = RerankerConfig()
    cfg.prm_train.eval_max_lists = -1
    with pytest.raises(ValueError, match="prm_train.eval_max_lists"):
        cfg.prm_train.validate()


def test_a_positive_eval_cap_validates():
    cfg = RerankerConfig()
    cfg.prm_train.eval_max_lists = 2000
    cfg.prm_train.validate()


# --- the S4 budget guard ---------------------------------------------------------------


def test_two_names_for_the_backbone_must_agree():
    # `prm_rollout.base_model` is rank_eval's tokenizer fallback and `model.base_model` is what
    # build_backbone loads. Crossed, the trainer trains one model and job E scores its lists
    # with another's tokenizer -- every number still computes.
    cfg = paired(prm_model="Qwen/Qwen3-Reranker-4B", orm_model="Qwen/Qwen3-Reranker-0.6B")
    with pytest.raises(ValueError) as e:
        check_prm_budgets(cfg)
    assert "prm_rollout.base_model" in str(e.value) and "model.base_model" in str(e.value)


def test_agreeing_backbones_pass():
    check_prm_budgets(paired())


def test_the_shipped_defaults_disagree_and_are_caught():
    # ModelConfig defaults to 0.6B and PRMRolloutConfig to 4B. A PRM training config that
    # forgets `model.base_model` therefore trains a different model than it says it does, and
    # this is the guard that refuses rather than the run that silently does it.
    with pytest.raises(ValueError):
        check_prm_budgets(RerankerConfig())


def test_the_summary_names_all_three_budgets_with_their_models():
    # S4's whole point: a crossed pair is visible in the first ten lines of a log, not never.
    lines = check_prm_budgets(paired())
    assert len(lines) == 3
    joined = "\n".join(lines)
    for token in ("max_new_tokens", "gpt-oss-120b", "Qwen/Qwen3-Reranker-4B"):
        assert token in joined
    # The ORM row is printed precisely because it is the one this job must NOT pick up.
    assert "orm" in joined and str(RerankerConfig().model.max_length) in joined


# --- the shipped file ------------------------------------------------------------------


def shipped(name: str) -> RerankerConfig:
    return load_config(["--config", os.path.join(CONFIGS, name)])


@pytest.mark.parametrize("arm", ARMS)
def test_each_shipped_arm_loads_and_passes_both_gates(arm):
    # _from_dict raises on an unknown key, so this also catches a typo'd knob -- which would
    # otherwise leave the default in place and train something other than the file says.
    cfg = shipped(arm)
    cfg.prm_train.validate()
    check_prm_budgets(cfg)


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_selects_on_the_prm_metric(arm):
    # build_training_args passes these straight to TrainingArguments, and the default
    # `eval_pr_auc` is a pointwise metric this trainer never computes -- load_best_model_at_end
    # would then never find its key.
    train = shipped(arm).train
    assert train.metric_for_best_model == "eval_prm_ndcg"
    assert train.greater_is_better is True


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("knob", CAMPAIGN_KNOBS)
def test_each_arm_inherits_the_campaign_it_trains_on(arm, knob):
    # `_base: prm_rollout.yaml` makes this structural: move the campaign's out_dir and every
    # arm follows, where a copied block would have to be found and edited in each file.
    campaign = shipped("prm_rollout.yaml").prm_rollout
    assert getattr(shipped(arm).prm_rollout, knob) == getattr(campaign, knob)


@pytest.mark.parametrize("arm", ARMS)
def test_an_arm_moves_the_backbone_in_both_places_the_guard_compares(arm):
    # base_model is the one campaign knob an arm may move -- it IS the thing being compared.
    # Moving only one of the two is what check_prm_budgets exists to refuse.
    cfg = shipped(arm)
    assert cfg.prm_rollout.base_model == cfg.model.base_model
    assert cfg.prm_rollout.base_model != shipped("prm_rollout.yaml").prm_rollout.base_model


def test_the_arms_train_different_backbones_and_write_to_different_places():
    a, b = (shipped(x) for x in ARMS)
    assert a.model.base_model != b.model.base_model
    assert a.train.output_dir != b.train.output_dir
    assert a.mlflow.run_name != b.mlflow.run_name


def test_every_arm_caps_the_eval_on_the_identical_seeded_subset():
    # Two backbones scored on different val subsets are not comparable, and nothing in
    # eval_prm_ndcg would say so. Same cap and same seed is what makes the arms an experiment.
    caps = {(shipped(x).prm_train.eval_max_lists, shipped(x).prm_train.eval_subsample_seed)
            for x in ARMS}
    assert len(caps) == 1
    assert caps.pop()[0] > 0


@pytest.mark.parametrize("arm", ARMS)
def test_the_micro_batch_stays_at_one_so_the_effective_batch_is_the_accumulation(arm):
    # Measured on the ORM: throughput FALLS as per_device grows (1.10 -> 0.79 -> 0.53 lists/s
    # at 1/2/4) because the collator pads to the longest candidate in the batch. The effective
    # batch is 1 x 32 x 4 GPUs = 128; prm_train.sh is what supplies the 4.
    t = shipped(arm).train
    assert t.per_device_train_batch_size == 1
    assert t.gradient_accumulation_steps == 32
