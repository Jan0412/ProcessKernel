"""``prm.rollout.train``: the LambdaRank trainer and its depth-sliced eval (PLAN_TRAINER §7).

The load-bearing test here is the parity one: the trainer's eval and job E's report must be one
statistic over one set of lists, not two implementations that agree until one is edited.

A tiny deterministic scorer stands in for the backbone. No checkpoint, no GPU, no download.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn
import yaml
from transformers import TrainingArguments

from reranker.src.config import PROJECT_ROOT, load_config
from reranker.src.listwise.dataset import ListwiseCollator
from reranker.src.listwise.trainer import lambdarank_loss
from reranker.src.model import HeadInfo
from reranker.src.prm.rollout import dataset, lists, rank_eval, train
from reranker.tests.prm.rollout import campaignfixture
from reranker.tests.prm.rollout.campaignfixture import (
    TAG,
    CharTokenizer,
    campaign,
    cfg,
    pre,
    two_prefix_campaign,
)


def row(*rels, rel_depth_mean=0.5, key=None, cut_index=20) -> lists.ListRow:
    key = key or f"{TAG}:2:37:0:{cut_index}"
    return lists.ListRow(
        list_key=key, run_tag=TAG, level=2, problem_id=37, round=0, cut_index=cut_index,
        rel_depth_mean=rel_depth_mean, split="val", source="cut",
        items=[lists.Item(f"{key}#{i}", rel=r, n_rollouts=4, se=0.2)
               for i, r in enumerate(rels)],
    )


def scores_of(r, *values) -> dict[str, float]:
    return {item.prefix_id: v for item, v in zip(r.items, values)}


# --- the flat metric dict --------------------------------------------------------------


def test_the_metrics_are_rank_evals_own_numbers_under_the_trainers_names():
    # Parity, overall. rank_eval.report is imported and called rather than reimplemented, so a
    # training curve and a job-E report are the same statistic by construction (§7).
    conf = cfg(depth_buckets=1)
    r = row(2.0, 1.0, 0.0)
    scores = scores_of(r, 3.0, 2.0, 1.0)

    got = train.metrics_from([r], scores, conf, "eval")
    want = rank_eval.report([r], scores, conf)

    assert got["eval_prm_ndcg"] == want["ndcg"]
    assert got["eval_prm_pairwise_acc"] == want["pairwise_acc"]
    assert got["eval_prm_pairs"] == want["pairs"]
    assert got["eval_prm_lists"] == want["lists"]


def test_every_band_that_measured_something_is_reported_under_its_index():
    conf = cfg(depth_buckets=4)
    shallow = row(2.0, 0.0, rel_depth_mean=0.15)
    deep = row(2.0, 0.0, rel_depth_mean=0.85, key=f"{TAG}:2:38:0:20")
    scores = {**scores_of(shallow, 9.0, 1.0), **scores_of(deep, 1.0, 9.0)}

    got = train.metrics_from([shallow, deep], scores, conf, "eval")
    want = rank_eval.report([shallow, deep], scores, conf)

    assert got["eval_prm_pairwise_acc_b0"] == want["buckets"][0]["pairwise_acc"] == 1.0
    assert got["eval_prm_pairwise_acc_b3"] == want["buckets"][3]["pairwise_acc"] == 0.0


def test_a_band_that_measured_nothing_is_absent_rather_than_zero():
    # 0.0 reads as "the model ranks nothing right here", which is a different claim from
    # "nothing was measured here" -- rank_eval._band already refuses to make it, and flattening
    # a None into HF's metric dict would make it anyway.
    conf = cfg(depth_buckets=4)
    shallow = row(2.0, 0.0, rel_depth_mean=0.15)

    got = train.metrics_from([shallow], scores_of(shallow, 9.0, 1.0), conf, "eval")

    assert "eval_prm_pairwise_acc_b0" in got
    for empty in ("b1", "b2", "b3"):
        assert f"eval_prm_pairwise_acc_{empty}" not in got
        assert f"eval_prm_ndcg_{empty}" not in got


def test_accuracy_pools_pairs_across_lists_rather_than_averaging_them():
    # A 3-item list carries 3 pairs and a 2-item list 1, so a mean of per-list accuracies
    # would weight the pair as heavily as the list.
    conf = cfg(depth_buckets=1)
    wide = row(2.0, 1.0, 0.0)
    narrow = row(1.0, 0.0, key=f"{TAG}:2:38:0:20")
    scores = {**scores_of(wide, 3.0, 2.0, 1.0), **scores_of(narrow, 1.0, 2.0)}

    got = train.metrics_from([wide, narrow], scores, conf, "eval")

    assert got["eval_prm_pairs"] == 4
    assert got["eval_prm_pairwise_acc"] == 0.75


def test_the_prefix_is_applied_so_hf_can_select_on_the_key_the_config_names():
    # train.metric_for_best_model is `eval_prm_ndcg`; HF passes metric_key_prefix="eval".
    conf = cfg(depth_buckets=1)
    r = row(2.0, 0.0)
    got = train.metrics_from([r], scores_of(r, 2.0, 1.0), conf, "eval")
    assert "eval_prm_ndcg" in got


def test_every_reported_metric_is_a_plain_float_hf_can_log():
    conf = cfg(depth_buckets=2)
    r = row(2.0, 1.0, 0.0, rel_depth_mean=0.15)
    got = train.metrics_from([r], scores_of(r, 3.0, 2.0, 1.0), conf, "eval")
    assert got and all(isinstance(v, float) for v in got.values())


# --- the trainer ------------------------------------------------------------------------


class LengthScorer(nn.Module):
    """One scalar per sequence: its real token count. Deterministic and input-dependent."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, **kw):
        n = attention_mask.sum(dim=1).float().unsqueeze(-1)
        return type("Out", (), {"logits": n + self.bias})()


def trainer_on(tmp_path, cfg_root, ds):
    collator = ListwiseCollator(CharTokenizer())
    return train.PRMListwiseTrainer(
        model=LengthScorer(),
        args=TrainingArguments(output_dir=str(tmp_path / "ckpt"), report_to=[], use_cpu=True),
        train_dataset=ds,
        data_collator=collator,
        head_info=HeadInfo("seq_cls"),
        sigma=cfg_root.prm_train.sigma,
        alpha=cfg_root.prm_train.loss_alpha,
        list_collator=collator,
        eval_lists_dataset=ds,
        rollout_cfg=cfg_root.prm_rollout,
    )


def test_the_eval_scores_the_val_lists_and_reports_rank_evals_numbers(tmp_path):
    # The integration half of the parity claim: the scores come off a real forward pass and
    # the report is computed over the lists the dataset actually holds.
    _, cfg_root = two_prefix_campaign(tmp_path)
    ds = dataset.PRMListDataset(cfg_root, "val", CharTokenizer())
    got = trainer_on(tmp_path, cfg_root, ds)._evaluate_lists("eval")

    # p0 is cut at 4 chars and p1 at 12, so LengthScorer ranks p1 above p0 -- and p0 carries
    # the higher rel, so the list's single pair is ranked wrong.
    assert got["eval_prm_lists"] == 1
    assert got["eval_prm_pairs"] == 1
    assert got["eval_prm_pairwise_acc"] == 0.0


def test_the_eval_pairs_each_list_with_its_own_scores_not_the_next_lists(tmp_path):
    # The loader is batch_size=1 and unshuffled precisely so `self.lists[i]` lines up with
    # batch i. A reordering would score every list against another's items with every number
    # still computing.
    ps = [pre("p0", sid=0, cut_char=4), pre("p1", sid=1, cut_char=12)]
    rows = [
        lists.ListRow(list_key=f"{TAG}:2:37:0:{k}", run_tag=TAG, level=2, problem_id=37,
                      round=0, cut_index=k, rel_depth_mean=d, split="val", source="cut",
                      items=[lists.Item("p0", rel=2.0, n_rollouts=4, se=0.2),
                             lists.Item("p1", rel=0.0, n_rollouts=4, se=0.2)])
        for k, d in ((5, 0.15), (9, 0.85))
    ]
    cfg_root = campaign(tmp_path, ps, rows, depth_buckets=4)
    ds = dataset.PRMListDataset(cfg_root, "val", CharTokenizer())

    got = trainer_on(tmp_path, cfg_root, ds)._evaluate_lists("eval")

    # Both lists hold the same two prefixes, so both bands must show the same wrong ordering.
    assert got["eval_prm_pairwise_acc_b0"] == 0.0
    assert got["eval_prm_pairwise_acc_b3"] == 0.0
    assert got["eval_prm_lists"] == 2


def test_the_loss_is_the_inherited_lambdarank_averaged_over_the_batchs_lists(tmp_path):
    # Nothing overrides compute_loss: ListwiseTrainer already splits by group_sizes and calls
    # lambdarank_loss per list. This pins that the inherited path is the one running.
    _, cfg_root = two_prefix_campaign(tmp_path)
    ds = dataset.PRMListDataset(cfg_root, "val", CharTokenizer())
    trainer = trainer_on(tmp_path, cfg_root, ds)

    collator = ListwiseCollator(CharTokenizer())
    batch = collator([ds[0]])
    got = trainer.compute_loss(trainer.model, batch)

    scores = trainer._forward_list_logits(trainer.model, batch)
    want = lambdarank_loss(scores, batch["rels"], sigma=cfg_root.prm_train.sigma,
                           alpha=cfg_root.prm_train.loss_alpha)
    assert torch.allclose(got, want)


# --- the entrypoint ----------------------------------------------------------------------


def write_cfg(tmp_path, **over) -> str:
    """A training config on disk, valid unless a test breaks exactly one thing."""
    body = {
        "prm_rollout": {"out_dir": str(tmp_path / "campaign"),
                        "parts_glob": str(tmp_path / "parts" / "*.jsonl"),
                        "base_model": "Qwen/Qwen3-Reranker-4B", "max_length": 128,
                        "depth_buckets": 1},
        "model": {"base_model": "Qwen/Qwen3-Reranker-4B"},
        "train": {"output_dir": str(tmp_path / "ckpt"),
                  "metric_for_best_model": "eval_prm_ndcg"},
        "prm_train": {"sigma": 1.0, "loss_alpha": 0.5},
    }
    for section, knobs in over.items():
        body.setdefault(section, {}).update(knobs)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(body))
    return str(path)


def test_the_module_imports_without_mlflow_installed():
    # mlflow is imported inside main(), not at module scope: `PRMListwiseTrainer` and
    # `metrics_from` are imported by tests and by anything that scores a checkpoint, and the
    # cluster venv has no mlflow today (ARCHITECTURE §9). A module-scope import would make the
    # whole suite unrunnable there.
    # Asserted on sys.modules rather than by blocking the import: transformers *probes* for
    # mlflow with importlib.util.find_spec, which a genuinely-absent package answers with None
    # rather than by raising. A finder that raises breaks the probe and tests the wrong thing.
    # find_spec never populates sys.modules, so this holds whether or not mlflow is installed.
    code = (
        "import sys\n"
        "import reranker.src.prm.rollout.train as t\n"
        "assert 'mlflow' not in sys.modules, 'train.py imported mlflow at module scope'\n"
        "assert hasattr(t, 'PRMListwiseTrainer') and hasattr(t, 'metrics_from')\n"
        "print('ok')\n"
    )
    # The repo root, not PROJECT_ROOT: `reranker.src...` resolves from the directory that
    # *contains* the reranker package, which is PROJECT_ROOT's parent.
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=os.path.dirname(PROJECT_ROOT))
    assert got.returncode == 0, got.stderr
    assert "ok" in got.stdout


def test_a_crossed_backbone_stops_the_run_before_a_model_is_loaded(tmp_path):
    # The S4 guard has to fire first or it is worthless: a 4B backbone load would come before
    # it and the run would die on memory instead of on the config error that caused it.
    path = write_cfg(tmp_path, model={"base_model": "Qwen/Qwen3-Reranker-0.6B"})
    with pytest.raises(ValueError) as e:
        train.main(["--config", path])
    assert "prm_rollout.base_model" in str(e.value) and "model.base_model" in str(e.value)


def test_a_bad_lambdarank_knob_stops_the_run_too(tmp_path):
    path = write_cfg(tmp_path, prm_train={"sigma": 0.0})
    with pytest.raises(ValueError, match="prm_train.sigma"):
        train.main(["--config", path])


def test_the_three_budgets_are_logged_before_anything_that_can_fail(tmp_path, capsys):
    # S4: "visible in the first ten lines of every job's log instead of never". The campaign
    # does not exist here, so the run dies right after -- and the lines must already be out.
    path = write_cfg(tmp_path)
    with pytest.raises(Exception):
        train.main(["--config", path])

    printed = capsys.readouterr().out
    assert printed.count("model=") == 3
    assert "gpt-oss-120b" in printed and "unused here" in printed


def one_list(split, cut_index):
    return lists.ListRow(
        list_key=f"{TAG}:2:37:0:{cut_index}", run_tag=TAG, level=2, problem_id=37, round=0,
        cut_index=cut_index, rel_depth_mean=0.5, split=split, source="cut",
        items=[lists.Item("p0", rel=2.0, n_rollouts=4, se=0.2),
               lists.Item("p1", rel=0.0, n_rollouts=4, se=0.2)],
    )


@pytest.mark.skipif(
    "warmup_ratio" not in inspect.signature(TrainingArguments.__init__).parameters,
    reason=(
        "transformers >= 5 dropped TrainingArguments.warmup_ratio, which "
        "reranker/src/trainer.py::build_training_args passes unconditionally -- so NO trainer "
        "in this repo constructs on 5.x, the three ORM ones included. pyproject pins no "
        "transformers version. Fixing it means either pinning <5 or moving to warmup_steps, "
        "which changes the ORM's LR schedule; both are decisions for a human."
    ),
)
def test_main_trains_a_step_and_saves_a_checkpoint_job_e_could_score(tmp_path):
    # End to end through the real HF Trainer on a ~19k-parameter backbone: config -> datasets
    # -> backbone -> train -> save. This is the wiring nothing else exercises, and the saved
    # reranker_head.json is what makes the result scoreable by job E at all.
    ps = [pre("p0", sid=0, cut_char=4), pre("p1", sid=1, cut_char=12)]
    campaign(tmp_path, ps, [one_list("val", 5)], train_rows=[one_list("train", 5)])
    model_dir = campaignfixture.tiny_backbone(tmp_path / "tiny")

    path = write_cfg(
        tmp_path,
        prm_rollout={"base_model": model_dir},
        model={"base_model": model_dir},
        train={"max_steps": 1, "eval_steps": 1, "save_steps": 1, "logging_steps": 1,
               "bf16": False, "gradient_checkpointing": False,
               "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
               "dataloader_num_workers": 0},
        mlflow={"db_file": str(tmp_path / "mlflow.db"), "experiment": "prm-test"},
    )
    train.main(["--config", path])

    final = tmp_path / "ckpt" / "final"
    assert (final / "reranker_head.json").is_file()
    assert json.loads((final / "reranker_head.json").read_text())["head_type"] == "seq_cls"
    assert (final / "tokenizer_config.json").is_file()


def test_the_eval_cap_reaches_the_val_split_and_never_the_train_split(tmp_path, monkeypatch):
    # Capping train would silently discard most of the training data -- the opposite of what
    # the knob is for. It is a measurement-cost decision, so main() is where it is applied and
    # the dataset never infers it from the split name.
    monkeypatch.setattr(train, "build_training_args", compat_training_args)
    seen = {}
    real = dataset.PRMListDataset

    def spy(cfg, split, tok, **kw):
        seen[split] = kw
        return real(cfg, split, tok, **kw)

    monkeypatch.setattr(train, "PRMListDataset", spy)

    ps = [pre("p0", sid=0, cut_char=4), pre("p1", sid=1, cut_char=12)]
    campaign(tmp_path, ps, [one_list("val", 5)], train_rows=[one_list("train", 5)])
    model_dir = campaignfixture.tiny_backbone(tmp_path / "tiny")
    path = write_cfg(
        tmp_path,
        prm_rollout={"base_model": model_dir},
        model={"base_model": model_dir},
        prm_train={"sigma": 1.0, "loss_alpha": 0.5, "eval_max_lists": 7,
                   "eval_subsample_seed": 5},
        train={"max_steps": 1, "eval_steps": 1, "save_steps": 1, "logging_steps": 1,
               "bf16": False, "gradient_checkpointing": False,
               "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
               "dataloader_num_workers": 0, "output_dir": str(tmp_path / "ckpt")},
        mlflow={"db_file": str(tmp_path / "mlflow.db"), "experiment": "prm-test"},
    )
    train.main(["--config", path])

    assert seen["val"] == {"max_lists": 7, "subsample_seed": 5}
    assert seen["train"] == {}


def test_gradient_checkpointing_is_non_reentrant_so_ddp_can_mark_grads_once(tmp_path,
                                                                            monkeypatch):
    # Reentrant checkpointing can make a parameter produce a gradient twice; DDP marks each
    # ready exactly once and raises "Expected to mark a variable ready only once". Invisible
    # on one GPU -- which is every run this repo has ever done -- and fatal on four.
    # Through the compat double for the same reason the end-to-end test below uses it: the
    # pre-existing warmup_ratio break, not anything this wrapper does.
    monkeypatch.setattr(train, "build_training_args", compat_training_args)
    # bf16 off only because this suite runs on CPU; the knob under test is unrelated.
    cfg = load_config(["--config", write_cfg(tmp_path, train={"bf16": False})])
    assert train.training_args_for(cfg).gradient_checkpointing_kwargs == {
        "use_reentrant": False
    }


def compat_training_args(cfg):
    """`build_training_args` minus the one kwarg transformers 5 dropped.

    A test double for a *pre-existing* incompatibility, not for anything this plan wrote: it
    exists so the wiring downstream of that call -- datasets, backbone, trainer, train, save --
    is exercised rather than blocked behind it. The skipped test above is what records that
    the real `build_training_args` cannot construct here.
    """
    t = cfg.train
    return TrainingArguments(
        output_dir=t.output_dir, num_train_epochs=t.epochs, max_steps=t.max_steps,
        learning_rate=t.lr, per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        weight_decay=t.weight_decay, bf16=t.bf16, fp16=t.fp16,
        gradient_checkpointing=t.gradient_checkpointing, logging_steps=t.logging_steps,
        eval_strategy="steps", eval_steps=t.eval_steps, save_strategy="steps",
        save_steps=t.save_steps, save_total_limit=t.save_total_limit,
        load_best_model_at_end=True, metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better, seed=t.seed,
        dataloader_num_workers=t.dataloader_num_workers, report_to=["mlflow"],
        run_name=cfg.mlflow.run_name, remove_unused_columns=False,
    )


def test_main_wires_config_to_a_trained_and_saved_checkpoint(tmp_path, monkeypatch):
    # End to end on a ~19k-parameter backbone: config -> datasets -> backbone -> HF train loop
    # -> save. This is the wiring nothing else exercises, and reranker_head.json is what makes
    # the result scoreable by job E at all.
    monkeypatch.setattr(train, "build_training_args", compat_training_args)

    ps = [pre("p0", sid=0, cut_char=4), pre("p1", sid=1, cut_char=12)]
    campaign(tmp_path, ps, [one_list("val", 5)], train_rows=[one_list("train", 5)])
    model_dir = campaignfixture.tiny_backbone(tmp_path / "tiny")

    path = write_cfg(
        tmp_path,
        prm_rollout={"base_model": model_dir},
        model={"base_model": model_dir},
        train={"max_steps": 1, "eval_steps": 1, "save_steps": 1, "logging_steps": 1,
               "bf16": False, "gradient_checkpointing": False,
               "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
               "dataloader_num_workers": 0, "output_dir": str(tmp_path / "ckpt")},
        mlflow={"db_file": str(tmp_path / "mlflow.db"), "experiment": "prm-test"},
    )
    train.main(["--config", path])

    final = tmp_path / "ckpt" / "final"
    assert (final / "reranker_head.json").is_file()
    assert json.loads((final / "reranker_head.json").read_text())["head_type"] == "seq_cls"
    assert (final / "tokenizer_config.json").is_file()
    assert (tmp_path / "mlflow.db").is_file()
