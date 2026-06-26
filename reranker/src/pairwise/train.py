"""End-to-end training entrypoint for the *pairwise* kernel reranker.

Steps:
  1. Load config, point MLflow at the SQLite database.
  2. Ensure the labeled source dataset exists; build pairs + a fresh problem-level
     train/val split (no test) if the pairwise artifacts are missing.
  3. Load tokenizer + RerankerModel (full-weight fine-tune).
  4. Train with a pairwise loss (logistic | margin) over (positive, negative) pairs.
  5. Validate on the val split: pointwise ranking metrics (model selection) plus
     pairwise loss / accuracy on held-out val pairs.
  6. Log params, metrics, the resolved config, and the best model to MLflow.

Usage:
    python -m reranker.src.pairwise.train --config configs/pairwise_config.yaml
    python -m reranker.src.pairwise.train --config configs/pairwise_config.yaml pairwise.loss_type=margin
    python -m reranker.src.pairwise.train --config configs/pairwise_config.yaml train.max_steps=20   # smoke test
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
from reranker.src.dataset import RerankerCollator, RerankerDataset
from reranker.src.metrics import make_compute_metrics
from reranker.src.model import build_backbone, load_tokenizer
from reranker.src.pairwise.dataset import PairwiseCollator, PairwiseDataset
from reranker.src.pairwise.pairs import build_pairs
from reranker.src.pairwise.trainer import PairwiseTrainer
from reranker.src.trainer import build_training_args, setup_mlflow


def _ensure_pairs(cfg) -> None:
    if not os.path.isfile(_resolve(cfg.data.dataset_jsonl)):
        print("[data] source dataset.jsonl missing — building it")
        build_dataset(cfg)
    pw = cfg.pairwise
    artifacts = [pw.pairs_train_jsonl, pw.pairs_val_jsonl, pw.pairs_splits_json]
    if not all(os.path.isfile(_resolve(p)) for p in artifacts):
        print("[data] pairwise artifacts missing — building pairs + splits")
        build_pairs(cfg)


def _default_run_name(cfg) -> str:
    slug = cfg.model.base_model.split("/")[-1]
    return f"{slug}_pairwise_{cfg.pairwise.loss_type}_{datetime.now():%Y%m%d_%H%M%S}"


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

    _ensure_pairs(cfg)

    tokenizer = load_tokenizer(cfg.model.base_model)

    train_ds = PairwiseDataset(
        pairs_jsonl=cfg.pairwise.pairs_train_jsonl,
        dataset_jsonl=cfg.data.dataset_jsonl,
        tokenizer=tokenizer,
        max_length=cfg.model.max_length,
        reserve_ref_tokens=cfg.model.reserve_ref_tokens,
    )
    eval_pairs = PairwiseDataset(
        pairs_jsonl=cfg.pairwise.pairs_val_jsonl,
        dataset_jsonl=cfg.data.dataset_jsonl,
        tokenizer=tokenizer,
        max_length=cfg.model.max_length,
        reserve_ref_tokens=cfg.model.reserve_ref_tokens,
    )
    # Pointwise val set (scored one candidate at a time) for ranking metrics +
    # model selection — uses the fresh pairwise split's "val" problems.
    val_ds = RerankerDataset(
        dataset_jsonl=cfg.data.dataset_jsonl,
        splits_json=cfg.pairwise.pairs_splits_json,
        split="val",
        tokenizer=tokenizer,
        max_length=cfg.model.max_length,
        reserve_ref_tokens=cfg.model.reserve_ref_tokens,
    )
    print(f"[data] train pairs: {len(train_ds)} | val pairs: {len(eval_pairs)} | "
          f"val candidates: {len(val_ds)} | val problems: {len(set(val_ds.groups))}")

    backbone, head_info = build_backbone(cfg, tokenizer)
    training_args = build_training_args(cfg)

    trainer = PairwiseTrainer(
        model=backbone,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=RerankerCollator(tokenizer),   # eval (pointwise) collator
        compute_metrics=make_compute_metrics(val_ds.groups),
        head_info=head_info,
        loss_type=cfg.pairwise.loss_type,
        margin=cfg.pairwise.margin,
        weighted=cfg.pairwise.weighted_loss,
        pair_collator=PairwiseCollator(tokenizer),
        eval_pairs_dataset=eval_pairs,
    )

    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.set_tag("base_model", cfg.model.base_model)
        mlflow.set_tag("head_type", cfg.model.head_type)
        mlflow.set_tag("training", "pairwise")
        mlflow.set_tag("pair_mode", cfg.pairwise.pair_mode)
        mlflow.set_tag("loss_type", cfg.pairwise.loss_type)
        mlflow.log_params(to_flat_dict(cfg))
        mlflow.log_metrics({
            "data_train_pairs": len(train_ds),
            "data_val_pairs": len(eval_pairs),
            "data_val_problems": len(set(val_ds.groups)),
        })
        _log_config_artifact(cfg)

        trainer.train()

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
