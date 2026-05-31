"""End-to-end training entrypoint for the kernel reranker.

Steps:
  1. Load config, point MLflow at the SQLite database.
  2. Build data/dataset.jsonl + data/splits.json if they don't exist yet.
  3. Load tokenizer + RerankerModel (full-weight fine-tune).
  4. Train with pointwise BCE; eval on the val split every `eval_steps`.
  5. Evaluate the best checkpoint on the held-out test split.
  6. Log params, metrics, the resolved config, and the final model to MLflow.

Usage:
    python -m reranker.train --config configs/default.yaml
    python -m reranker.train --config configs/default.yaml train.max_steps=20   # smoke test
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from datetime import datetime

import mlflow
import yaml

from reranker.src.config import _resolve, load_config, to_flat_dict
from reranker.src.data.build_dataset import build_dataset
from reranker.src.data.splits import build_splits
from reranker.src.dataset import RerankerCollator, build_datasets
from reranker.src.metrics import make_compute_metrics
from reranker.src.model import build_backbone, load_tokenizer
from reranker.src.trainer import (
    RerankerTrainer,
    build_training_args,
    compute_pos_weight,
    setup_mlflow,
)


def _ensure_data(cfg) -> None:
    if not os.path.isfile(_resolve(cfg.data.dataset_jsonl)):
        print("[data] dataset.jsonl missing — building it")
        build_dataset(cfg)
    if not os.path.isfile(_resolve(cfg.data.splits_json)):
        print("[data] splits.json missing — building it")
        build_splits(cfg)


def _default_run_name(cfg) -> str:
    slug = cfg.model.base_model.split("/")[-1]
    return f"{slug}_{datetime.now():%Y%m%d_%H%M%S}"


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

    _ensure_data(cfg)

    tokenizer = load_tokenizer(cfg.model.base_model)
    datasets = build_datasets(cfg, tokenizer, splits=("train", "val", "test"))
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    balance = {split: ds.label_balance() for split, ds in datasets.items()}
    print("[data] label balance:", json.dumps(balance, indent=2))

    pos_weight = cfg.train.pos_weight
    if pos_weight is None:
        pos_weight = compute_pos_weight(train_ds.labels)
    print(f"[train] pos_weight = {pos_weight:.3f}")

    backbone, head_info = build_backbone(cfg, tokenizer)
    collator = RerankerCollator(tokenizer)
    training_args = build_training_args(cfg)

    trainer = RerankerTrainer(
        model=backbone,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(val_ds.groups),
        head_info=head_info,
        pos_weight=pos_weight,
    )

    # Own the MLflow run so HF's MLflowCallback logs into it AND it stays open
    # for our post-training test metrics + artifact logging.
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.set_tag("base_model", cfg.model.base_model)
        mlflow.set_tag("head_type", cfg.model.head_type)
        mlflow.log_params(to_flat_dict(cfg))
        for split, b in balance.items():
            mlflow.log_metrics({f"data_{split}_{k}": v for k, v in b.items()})
        _log_config_artifact(cfg)

        trainer.train()

        # Evaluate the best checkpoint on the held-out test split.
        trainer.compute_metrics = make_compute_metrics(test_ds.groups)
        test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
        print("[test]", json.dumps(test_metrics, indent=2))
        mlflow.log_metrics(
            {k: v for k, v in test_metrics.items() if isinstance(v, (int, float))}
        )

        # Save best model + tokenizer + head metadata, log as an MLflow artifact.
        final_dir = os.path.join(_resolve(cfg.train.output_dir), "final")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        with open(os.path.join(final_dir, "reranker_head.json"), "w") as f:
            json.dump(dataclasses.asdict(head_info), f)
        mlflow.log_artifacts(final_dir, artifact_path="model")
        print(f"[done] best model saved to {final_dir}")


if __name__ == "__main__":
    main()
