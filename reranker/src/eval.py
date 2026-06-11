"""Evaluate a saved reranker checkpoint on data splits (train / val / test).

Prints the same metrics that MLflow tracks during training and (unless disabled)
logs them to the MLflow experiment, so you can evaluate any checkpoint — including
a pairwise-trained one — against any dataset / splits.json and have the per-split
numbers land in MLflow next to the training runs.

Point it at a specific dataset + splits with CLI overrides:
    data.dataset_jsonl=data/new20k.jsonl data.splits_json=data/new20k_splits.json

Usage:
    python -m reranker.src.eval --config configs/default.yaml
    python -m reranker.src.eval --config configs/default.yaml --checkpoint data/checkpoints/final
    python -m reranker.src.eval --config configs/pairwise_config.yaml \
        --checkpoint data/checkpoints_pairwise/final --splits test
    python -m reranker.src.eval --config configs/default.yaml --no-mlflow   # print only
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime

import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from reranker.src.config import _resolve, load_config, to_flat_dict
from reranker.src.dataset import RerankerCollator, build_datasets
from reranker.src.metrics import make_compute_metrics
from reranker.src.model import HeadInfo, _single_token_id, load_tokenizer
from reranker.src.trainer import RerankerTrainer, compute_pos_weight, setup_mlflow

_METRIC_ORDER = [
    "loss", "pr_auc", "roc_auc", "accuracy", "f1", "precision", "recall",
    "pass_at_1", "recall_at_1", "ndcg", "coverage", "positive_rate", "num_problems",
]


def _load_model(checkpoint_dir: str, cfg, tokenizer):
    head_json = os.path.join(checkpoint_dir, "reranker_head.json")
    if os.path.isfile(head_json):
        with open(head_json) as f:
            head_info = HeadInfo(**json.load(f))
    else:
        head_type = cfg.model.head_type
        if head_type == "yes_no_lm":
            head_info = HeadInfo(
                head_type,
                yes_id=_single_token_id(tokenizer, "yes"),
                no_id=_single_token_id(tokenizer, "no"),
            )
        else:
            head_info = HeadInfo(head_type)

    dtype = torch.bfloat16 if cfg.train.bf16 else (
        torch.float16 if cfg.train.fp16 else torch.float32
    )
    common = dict(dtype=dtype, attn_implementation=cfg.model.attn_implementation)

    if head_info.head_type == "seq_cls":
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir, num_labels=1, **common
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, **common)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, head_info


def _build_eval_args(cfg) -> TrainingArguments:
    return TrainingArguments(
        output_dir=_resolve(cfg.train.output_dir),
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        seed=cfg.train.seed,
        report_to=[],
        remove_unused_columns=False,
    )


def _print_split_metrics(split: str, metrics: dict) -> None:
    # Strip the split prefix from keys (e.g. "train_accuracy" -> "accuracy")
    prefix = split + "_"
    stripped = {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in metrics.items()
    }
    # Sort: preferred order first, then any remainder alphabetically
    ordered_keys = [k for k in _METRIC_ORDER if k in stripped]
    ordered_keys += sorted(k for k in stripped if k not in _METRIC_ORDER)

    col = max(len(k) for k in ordered_keys) + 2
    print(f"\n{'─' * (col + 12)}")
    print(f"  {split.upper()} METRICS")
    print(f"{'─' * (col + 12)}")
    for k in ordered_keys:
        v = stripped[k]
        if isinstance(v, float):
            print(f"  {k:<{col}} {v:.4f}")
        else:
            print(f"  {k:<{col}} {v}")
    print(f"{'─' * (col + 12)}")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", required=True)
    pre.add_argument("--checkpoint", default=None,
                     help="Checkpoint directory (default: <output_dir>/final)")
    pre.add_argument("--splits", default="train,val,test",
                     help="Comma-separated splits to evaluate (default: train,val,test)")
    pre.add_argument("--run-name", default=None,
                     help="MLflow run name (default: eval_<checkpoint>_<timestamp>)")
    pre.add_argument("--no-mlflow", action="store_true",
                     help="Disable MLflow logging (print metrics only)")
    pre_args, remaining = pre.parse_known_args()

    cfg = load_config(argv=["--config", pre_args.config] + remaining)
    splits = tuple(s.strip() for s in pre_args.splits.split(",") if s.strip())

    checkpoint_dir = pre_args.checkpoint or os.path.join(_resolve(cfg.train.output_dir), "final")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    print(f"[eval] checkpoint : {checkpoint_dir}")
    print(f"[eval] config     : {pre_args.config}")
    print(f"[eval] dataset    : {_resolve(cfg.data.dataset_jsonl)}")
    print(f"[eval] splits     : {_resolve(cfg.data.splits_json)}  {list(splits)}")

    if os.path.isfile(os.path.join(checkpoint_dir, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = load_tokenizer(cfg.model.base_model)

    datasets = build_datasets(cfg, tokenizer, splits=splits)
    model, head_info = _load_model(checkpoint_dir, cfg, tokenizer)

    if cfg.train.pos_weight is not None:
        pos_weight = cfg.train.pos_weight
    elif "train" in datasets:
        pos_weight = compute_pos_weight(datasets["train"].labels)
    else:
        pos_weight = 1.0
    trainer = RerankerTrainer(
        model=model,
        args=_build_eval_args(cfg),
        data_collator=RerankerCollator(tokenizer),
        head_info=head_info,
        pos_weight=pos_weight,
    )

    use_mlflow = not pre_args.no_mlflow
    if use_mlflow:
        setup_mlflow(cfg)
        run_name = pre_args.run_name or (
            f"eval_{os.path.basename(checkpoint_dir.rstrip('/'))}_{datetime.now():%Y%m%d_%H%M%S}"
        )
        mlflow.start_run(run_name=run_name)
        mlflow.set_tag("stage", "eval")
        mlflow.set_tag("base_model", cfg.model.base_model)
        mlflow.set_tag("head_type", head_info.head_type)
        mlflow.set_tag("eval_checkpoint", checkpoint_dir)
        mlflow.log_params(to_flat_dict(cfg))
        mlflow.log_param("eval_checkpoint", checkpoint_dir)
        mlflow.log_param("eval_splits", ",".join(splits))

    try:
        for split in splits:
            ds = datasets[split]
            trainer.compute_metrics = make_compute_metrics(ds.groups)
            metrics = trainer.evaluate(eval_dataset=ds, metric_key_prefix=split)
            _print_split_metrics(split, metrics)
            if use_mlflow:
                mlflow.log_metrics({
                    k: v for k, v in metrics.items()
                    if isinstance(v, (int, float)) and math.isfinite(v)
                })
    finally:
        if use_mlflow:
            mlflow.end_run()

    print("\n[eval] done")


if __name__ == "__main__":
    main()
