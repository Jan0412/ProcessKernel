"""LambdaRank training over job D's lists (PLAN_TRAINER §5, §7).

    reads   {out_dir}/lists_{train,val}.jsonl, prefixes.jsonl, v1's parts
    writes  {train.output_dir}/final/ + reranker_head.json, and MLflow

`PRMListwiseTrainer` overrides **only** the evaluation. `compute_loss` is inherited from
`ListwiseTrainer` and is already exactly right: it splits the flat score vector by
`group_sizes` and calls `lambdarank_loss` per list. Nothing under `reranker/src/listwise/` is
modified to get that (ARCHITECTURE S5).

The eval is overridden because four of the parent's eight metrics are wrong on PRM lists --
`top1_correct`, `speed_regret_at1`, `fast_at1` and the correctness/speed pair split all read
``rel > 0`` as "this kernel is correct" and ``rel - 1`` as its speed grade. For the PRM
``rel = 2 * v_graded`` is a mean over K rollouts: ``rel = 0.6`` means "30% of continuations
from here worked", not "wrong". They would still compute, and mean nothing.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import tempfile
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader

from reranker.src.config import _resolve, check_prm_budgets, load_config, to_flat_dict
from reranker.src.listwise.dataset import ListwiseCollator
from reranker.src.listwise.trainer import ListwiseTrainer
from reranker.src.model import build_backbone, load_tokenizer
from reranker.src.prm.rollout import rank_eval
from reranker.src.prm.rollout.dataset import PRMListDataset
from reranker.src.trainer import build_training_args, setup_mlflow


def metrics_from(rows, scores: dict[str, float], conf, prefix: str) -> dict[str, float]:
    """`rank_eval.report` -> the flat dict HF logs and selects on.

    Imported and called rather than reimplemented: that is what makes a training curve and a
    job-E report one statistic instead of two that agree until one is edited (§7).

    A band that measured nothing is **left out**, not written as 0.0 -- `rank_eval._band`
    already refuses that claim ("ranks nothing right here" is not "nothing was measured
    here"), and flattening its None into the metric dict would make it anyway. The overall
    numbers are never None in practice: `lists.py` drops all-equal lists, so every list that
    reaches training carries at least one ranking pair.
    """
    report = rank_eval.report(rows, scores, conf)
    out = {
        f"{prefix}_prm_lists": float(report["lists"]),
        f"{prefix}_prm_items": float(report["items"]),
        f"{prefix}_prm_pairs": float(report["pairs"]),
        f"{prefix}_prm_two_item_lists": float(report["two_item_lists"]),
        # Nonzero means the depth window moved since job A enumerated the cuts.
        f"{prefix}_prm_lists_outside_window": float(report["lists_outside_window"]),
    }
    for key in ("ndcg", "pairwise_acc"):
        if report[key] is not None:
            out[f"{prefix}_prm_{key}"] = float(report[key])
    for i, band in enumerate(report["buckets"]):
        for key in ("ndcg", "pairwise_acc"):
            if band[key] is not None:
                out[f"{prefix}_prm_{key}_b{i}"] = float(band[key])
    return out


class PRMListwiseTrainer(ListwiseTrainer):
    """`ListwiseTrainer` with the PRM's evaluation. The loss is the parent's, unchanged.

    `rollout_cfg` is the campaign section, and it is needed for one thing: `report` slices by
    the depth bands `bucket_edges` derives from the window job A actually cut in.
    """

    def __init__(self, *args, rollout_cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self._rollout_cfg = rollout_cfg

    @torch.no_grad()
    def _evaluate_lists(self, prefix: str) -> dict[str, float]:
        """Score every val list, then hand `rank_eval` the scores it reports on.

        `batch_size=1` and no shuffling, so batch *i* is `self._eval_lists.lists[i]`: the
        scores are attached to their own list's items positionally. A reordering here would
        score each list against another list's prefixes with every number still computing.
        """
        loader = DataLoader(
            self._eval_lists,
            batch_size=1,
            shuffle=False,
            collate_fn=self._list_collator,
            num_workers=self.args.dataloader_num_workers,
        )
        model = self.model
        was_training = model.training
        model.eval()

        scores: dict[str, float] = {}
        for lst, batch in zip(self._eval_lists.lists, loader):
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            values = self._forward_list_logits(model, batch).float().tolist()
            for item, value in zip(lst.items, values):
                scores[item.prefix_id] = value

        if was_training:
            model.train()
        return metrics_from(self._eval_lists.lists, scores, self._rollout_cfg, prefix)


# --- the entrypoint ----------------------------------------------------------------------


def _default_run_name(cfg) -> str:
    return f"{cfg.model.base_model.split('/')[-1]}_prm_listwise_{datetime.now():%Y%m%d_%H%M%S}"


def _log_config_artifact(mlflow, cfg) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "resolved_config.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
        mlflow.log_artifact(path, artifact_path="config")


def training_args_for(cfg):
    """`build_training_args` plus the one setting DDP needs and a single-GPU run never did.

    Reentrant checkpointing can make a parameter yield a gradient twice; DDP marks each ready
    exactly once and raises "Expected to mark a variable ready only once". This is the first
    job in the repo to run on more than one GPU, which is why nothing has hit it before.

    Set here rather than in `build_training_args`: that function is shared with the three ORM
    trainers, and non-reentrant checkpointing has different recompute behaviour -- not a thing
    to change under running experiments for a need only this job has.
    """
    args = build_training_args(cfg)
    args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    return args


def main(argv=None) -> None:
    cfg = load_config(None if argv is None else list(argv))

    # Both gates before anything expensive: a 4B backbone load would otherwise come first and
    # the run would die on memory instead of on the config error that caused it.
    cfg.prm_train.validate()
    for line in check_prm_budgets(cfg):
        print(f"[budget] {line}")

    # Imported here, not at module scope: `PRMListwiseTrainer` and `metrics_from` are imported
    # by the test suite and by anything that scores a checkpoint, and the cluster venv has no
    # mlflow today (ARCHITECTURE §9). A top-level import would make all of that unimportable.
    import mlflow

    setup_mlflow(cfg)
    if cfg.mlflow.run_name is None:
        cfg.mlflow.run_name = _default_run_name(cfg)

    tokenizer = load_tokenizer(cfg.model.base_model)
    train_ds = PRMListDataset(cfg, "train", tokenizer)
    # The cap is a measurement-cost decision, so it is applied here and only to val. Capping
    # train would discard most of the corpus while every log line still named the full one.
    eval_ds = PRMListDataset(
        cfg, "val", tokenizer,
        max_lists=cfg.prm_train.eval_max_lists,
        subsample_seed=cfg.prm_train.eval_subsample_seed,
    )
    print(f"[data] train lists: {len(train_ds)} | val lists: {len(eval_ds)} | "
          f"val items: {sum(len(l.items) for l in eval_ds.lists)}")

    backbone, head_info = build_backbone(cfg, tokenizer)
    # One collator for both dataloaders: there is no pointwise PRM path, so the parent's
    # train/eval collator swap is a no-op rather than something to work around.
    collator = ListwiseCollator(tokenizer)

    trainer = PRMListwiseTrainer(
        model=backbone,
        args=training_args_for(cfg),
        train_dataset=train_ds,
        # Passed even though `evaluate()` is overridden end to end and reads `_eval_lists`
        # instead: Trainer.__init__ refuses `eval_strategy != "no"` without an eval_dataset,
        # and it validates before any of this class's code runs.
        eval_dataset=eval_ds,
        data_collator=collator,
        head_info=head_info,
        sigma=cfg.prm_train.sigma,
        alpha=cfg.prm_train.loss_alpha,
        list_collator=collator,
        eval_lists_dataset=eval_ds,
        rollout_cfg=cfg.prm_rollout,
    )

    # Only rank zero touches MLflow -- under DDP every rank runs this script and racing to
    # create the experiment trips a SQLite UNIQUE constraint (listwise/train.py has the same
    # guard). Training itself runs on all ranks.
    is_main = trainer.is_world_process_zero()
    run = mlflow.start_run(run_name=cfg.mlflow.run_name) if is_main else contextlib.nullcontext()
    with run:
        if is_main:
            mlflow.set_tag("base_model", cfg.model.base_model)
            mlflow.set_tag("head_type", cfg.model.head_type)
            mlflow.set_tag("training", "prm_listwise")
            mlflow.set_tag("loss_type", "lambdarank")
            mlflow.log_params(to_flat_dict(cfg))
            mlflow.log_metrics({
                "data_train_lists": len(train_ds),
                "data_val_lists": len(eval_ds),
                "data_val_items": sum(len(l.items) for l in eval_ds.lists),
            })
            _log_config_artifact(mlflow, cfg)

        trainer.train()

        if is_main:
            final_dir = os.path.join(_resolve(cfg.train.output_dir), "final")
            trainer.save_model(final_dir)
            tokenizer.save_pretrained(final_dir)
            # rank_eval.head_of reads this back; without it a checkpoint cannot be scored
            # without guessing how its outputs become one scalar.
            with open(os.path.join(final_dir, "reranker_head.json"), "w") as f:
                json.dump(dataclasses.asdict(head_info), f)
            mlflow.log_artifacts(final_dir, artifact_path="model")
            print(f"[done] best model saved to {final_dir}")


if __name__ == "__main__":
    main()
