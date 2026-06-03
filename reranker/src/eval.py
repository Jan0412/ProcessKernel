"""Evaluate a saved reranker checkpoint on all data splits (train / val / test).

Prints the same metrics that MLflow tracks during training so you can compare
checkpoints without launching the full training loop.

Usage:
    python -m reranker.src.eval --config configs/default.yaml
    python -m reranker.src.eval --config configs/default.yaml --checkpoint data/checkpoints/final
    python -m reranker.src.eval --config configs/default.yaml --checkpoint data/checkpoints/checkpoint-200
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from reranker.src.config import _resolve, load_config
from reranker.src.dataset import RerankerCollator, build_datasets
from reranker.src.metrics import make_compute_metrics
from reranker.src.model import HeadInfo, _single_token_id, load_tokenizer
from reranker.src.trainer import RerankerTrainer, compute_pos_weight

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
    pre_args, remaining = pre.parse_known_args()

    cfg = load_config(argv=["--config", pre_args.config] + remaining)

    checkpoint_dir = pre_args.checkpoint or os.path.join(_resolve(cfg.train.output_dir), "final")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    print(f"[eval] checkpoint : {checkpoint_dir}")
    print(f"[eval] config     : {pre_args.config}")

    if os.path.isfile(os.path.join(checkpoint_dir, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = load_tokenizer(cfg.model.base_model)

    datasets = build_datasets(cfg, tokenizer, splits=("train", "val", "test"))
    model, head_info = _load_model(checkpoint_dir, cfg, tokenizer)

    pos_weight = cfg.train.pos_weight or compute_pos_weight(datasets["train"].labels)
    trainer = RerankerTrainer(
        model=model,
        args=_build_eval_args(cfg),
        data_collator=RerankerCollator(tokenizer),
        head_info=head_info,
        pos_weight=pos_weight,
    )

    for split in ("train", "val", "test"):
        ds = datasets[split]
        trainer.compute_metrics = make_compute_metrics(ds.groups)
        metrics = trainer.evaluate(eval_dataset=ds, metric_key_prefix=split)
        _print_split_metrics(split, metrics)

    print("\n[eval] done")


if __name__ == "__main__":
    main()
