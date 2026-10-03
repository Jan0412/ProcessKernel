"""``PRMTrainConfig`` and the cross-section budget guard (PLAN_TRAINER §6)."""

from __future__ import annotations

import os
import re

import pytest

from processkernel.config import PROJECT_ROOT, RerankerConfig, check_prm_budgets, load_config

CONFIGS = os.path.join(PROJECT_ROOT, "configs")
SHIPPED = "prm_train.yaml"

#: The effective batch the lr was tuned at. THIS is the invariant, not any one factor of it.
EFFECTIVE_BATCH = 128


def usage_gpus() -> int:
    """The rank count the config's own usage line launches with."""
    with open(os.path.join(CONFIGS, SHIPPED)) as f:
        head = f.read().split("\n\n", 1)[0]
    n = re.search(r"--num_processes=(\d+)", head)
    assert n, f"{SHIPPED} has no accelerate --num_processes=<n> usage line"
    return int(n.group(1))


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


def shipped(name: str = SHIPPED) -> RerankerConfig:
    return load_config(["--config", os.path.join(CONFIGS, name)])


def test_the_shipped_config_loads_and_passes_both_gates():
    # _from_dict raises on an unknown key, so this also catches a typo'd knob -- which would
    # otherwise leave the default in place and train something other than the file says.
    cfg = shipped()
    cfg.prm_train.validate()
    check_prm_budgets(cfg)


def test_it_selects_on_the_prm_metric():
    # The default `eval_pr_auc` is a pointwise metric this trainer never computes, so
    # load_best_model_at_end would never find its key.
    train = shipped().train
    assert train.metric_for_best_model == "eval_prm_ndcg"
    assert train.greater_is_better is True


def test_it_trains_on_the_build_of_prm_build_yaml():
    build = shipped("prm_build.yaml").prm.out_dir
    rollout = shipped().prm_rollout
    assert os.path.dirname(os.path.dirname(rollout.parts_glob)) == build
    assert os.path.dirname(rollout.splits_json) == build


def test_every_config_imputes_and_picks_with_the_orm_of_orm_yaml():
    orm = shipped("orm.yaml").train.output_dir
    for ckpt in (shipped().prm_rollout.orm_checkpoint,
                 shipped("prm_rollout.yaml").prm_rollout.orm_checkpoint,
                 shipped("prm_search.yaml").prm_search.orm_checkpoint):
        assert ckpt == os.path.join(orm, "final")


def test_the_micro_batch_is_one_and_the_effective_batch_is_the_tuned_one():
    # Throughput FALLS as per_device grows (the collator pads to the longest candidate), so
    # the micro batch is pinned at 1; the effective batch is the product the lr was tuned at.
    t = shipped().train
    assert t.per_device_train_batch_size == 1
    effective = t.per_device_train_batch_size * t.gradient_accumulation_steps * usage_gpus()
    assert effective == EFFECTIVE_BATCH
