"""End-to-end training entrypoint for the *listwise* (LambdaRank) kernel reranker.

Steps:
  1. Load config, point MLflow at the SQLite database.
  2. Ensure the labeled source dataset exists; build speed-graded lists + a fresh
     problem-level train/val split (no test) if the listwise artifacts are missing.
  3. Load tokenizer + RerankerModel (full-weight fine-tune).
  4. Train with a LambdaRank loss over per-problem candidate lists.
  5. Validate on the val split: pointwise ranking metrics (model selection) plus
     listwise loss / pair-accuracy on held-out val lists.
  6. Log params, metrics, the resolved config, and the best model to MLflow.

Usage:
    python -m reranker.src.listwise.train --config configs/listwise_config.yaml
    python -m reranker.src.listwise.train --config configs/listwise_config.yaml listwise.sigma=2.0
    python -m reranker.src.listwise.train --config configs/listwise_config.yaml train.max_steps=20   # smoke test
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import tempfile
from datetime import datetime

import mlflow
import yaml

from reranker.src.config import _resolve, load_config, to_flat_dict
from reranker.src.data.build_dataset import build_dataset
from reranker.src.dataset import RerankerCollator, RerankerDataset
from reranker.src.listwise.dataset import ListwiseCollator, ListwiseDataset
from reranker.src.listwise.lists import build_lists
from reranker.src.listwise.trainer import ListwiseTrainer
from reranker.src.metrics import make_compute_metrics
from reranker.src.model import build_backbone, load_tokenizer
from reranker.src.trainer import build_training_args, setup_mlflow


def _ensure_lists(cfg) -> None:
    if not os.path.isfile(_resolve(cfg.data.dataset_jsonl)):
        print("[data] source dataset.jsonl missing — building it")
        build_dataset(cfg)
    lw = cfg.listwise
    artifacts = [lw.lists_train_jsonl, lw.lists_val_jsonl, lw.lists_splits_json]
    if not all(os.path.isfile(_resolve(p)) for p in artifacts):
        print("[data] listwise artifacts missing — building lists + splits")
        build_lists(cfg)


def _default_run_name(cfg) -> str:
    slug = cfg.model.base_model.split("/")[-1]
    return f"{slug}_listwise_{datetime.now():%Y%m%d_%H%M%S}"


def _log_config_artifact(cfg) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "resolved_config.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
        mlflow.log_artifact(path, artifact_path="config")


def main() -> None:
    cfg = load_config()
    setup_mlflow(cfg)
    if cfg.mlflow.run_name is None:
        cfg.mlflow.run_name = _default_run_name(cfg)

    _ensure_lists(cfg)

    tokenizer = load_tokenizer(cfg.model.base_model)

    train_ds = ListwiseDataset(
        lists_jsonl=cfg.listwise.lists_train_jsonl,
        dataset_jsonl=cfg.data.dataset_jsonl,
        tokenizer=tokenizer,
        max_length=cfg.model.max_length,
        reserve_ref_tokens=cfg.model.reserve_ref_tokens,
    )
    eval_lists = ListwiseDataset(
        lists_jsonl=cfg.listwise.lists_val_jsonl,
        dataset_jsonl=cfg.data.dataset_jsonl,
        tokenizer=tokenizer,
        max_length=cfg.model.max_length,
        reserve_ref_tokens=cfg.model.reserve_ref_tokens,
    )
    # Pointwise val set (scored one candidate at a time) for ranking metrics +
    # model selection — uses the fresh listwise split's "val" problems.
    val_ds = RerankerDataset(
        dataset_jsonl=cfg.data.dataset_jsonl,
        splits_json=cfg.listwise.lists_splits_json,
        split="val",
        tokenizer=tokenizer,
        max_length=cfg.model.max_length,
        reserve_ref_tokens=cfg.model.reserve_ref_tokens,
    )
    print(f"[data] train lists: {len(train_ds)} | val lists: {len(eval_lists)} | "
          f"val candidates: {len(val_ds)} | val problems: {len(set(val_ds.groups))}")

    backbone, head_info = build_backbone(cfg, tokenizer)
    training_args = build_training_args(cfg)

    trainer = ListwiseTrainer(
        model=backbone,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=RerankerCollator(tokenizer),   # eval (pointwise) collator
        compute_metrics=make_compute_metrics(val_ds.groups),
        head_info=head_info,
        sigma=cfg.listwise.sigma,
        alpha=cfg.listwise.loss_alpha,
        list_collator=ListwiseCollator(tokenizer),
        eval_lists_dataset=eval_lists,
    )

    # Only the main process touches MLflow / writes artifacts — under DDP every
    # rank runs this script, and racing to create the experiment/run trips a
    # SQLite UNIQUE constraint. Training itself runs on all ranks.
    is_main = trainer.is_world_process_zero()
    run_cm = mlflow.start_run(run_name=cfg.mlflow.run_name) if is_main else contextlib.nullcontext()
    with run_cm:
        if is_main:
            mlflow.set_tag("base_model", cfg.model.base_model)
            mlflow.set_tag("head_type", cfg.model.head_type)
            mlflow.set_tag("training", "listwise")
            mlflow.set_tag("loss_type", "lambdarank")
            mlflow.log_params(to_flat_dict(cfg))
            mlflow.log_metrics({
                "data_train_lists": len(train_ds),
                "data_val_lists": len(eval_lists),
                "data_val_problems": len(set(val_ds.groups)),
            })
            _log_config_artifact(cfg)

        trainer.train()

        # Save best model + tokenizer + head metadata, log as an MLflow artifact.
        if is_main:
            final_dir = os.path.join(_resolve(cfg.train.output_dir), "final")
            trainer.save_model(final_dir)
            tokenizer.save_pretrained(final_dir)
            with open(os.path.join(final_dir, "reranker_head.json"), "w") as f:
                json.dump(dataclasses.asdict(head_info), f)
            mlflow.log_artifacts(final_dir, artifact_path="model")
            print(f"[done] best model saved to {final_dir}")


if __name__ == "__main__":
    main()
