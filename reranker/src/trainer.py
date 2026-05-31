"""Custom HuggingFace Trainer (pointwise BCE) plus MLflow wiring.

The model wrapper emits one scalar logit per (reference, kernel) pair; this
trainer applies `BCEWithLogitsLoss` with an optional `pos_weight` to counteract
class imbalance (most generated kernels are negative).

MLflow: we set the env vars HF's built-in `MLflowCallback` reads, so step-level
loss/metrics and config params are logged automatically when
`report_to=["mlflow"]`. `train.py` adds the extra artifact / final-test logging.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments

from reranker.src.config import RerankerConfig, _resolve
from reranker.src.model import HeadInfo


def setup_mlflow(cfg: RerankerConfig) -> None:
    """Point MLflow at the SQLite database and select the experiment.

    MLflow auto-creates and initialises the SQLite DB on first use, so no
    manual ``mlflow db upgrade`` step is required.
    """
    os.environ["MLFLOW_TRACKING_URI"] = cfg.mlflow.tracking_uri()
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow.experiment
    # Log checkpoints as artifacts via the HF callback.
    os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "false")


def compute_pos_weight(labels: list[int]) -> float:
    """neg / pos — the standard BCEWithLogitsLoss pos_weight for imbalance."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0:
        return 1.0
    return neg / pos


def build_training_args(cfg: RerankerConfig) -> TrainingArguments:
    t = cfg.train
    return TrainingArguments(
        output_dir=_resolve(t.output_dir),
        num_train_epochs=t.epochs,
        max_steps=t.max_steps,
        learning_rate=t.lr,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        warmup_ratio=t.warmup_ratio,
        weight_decay=t.weight_decay,
        bf16=t.bf16,
        fp16=t.fp16,
        gradient_checkpointing=t.gradient_checkpointing,
        logging_steps=t.logging_steps,
        eval_strategy="steps",
        eval_steps=t.eval_steps,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        seed=t.seed,
        dataloader_num_workers=t.dataloader_num_workers,
        report_to=["mlflow"],
        run_name=cfg.mlflow.run_name,
        remove_unused_columns=False,
    )


class RerankerTrainer(Trainer):
    """Trainer with a pointwise BCEWithLogitsLoss over the reranker's scalar logit.

    The model passed in is a raw HuggingFace backbone; the head-specific reduction
    to one scalar logit per example is applied via `head_info`.
    """

    def __init__(self, *args, head_info: HeadInfo, pos_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._head_info = head_info
        self._pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

    def _loss_fn(self, device, dtype):
        return nn.BCEWithLogitsLoss(
            pos_weight=self._pos_weight.to(device=device, dtype=dtype)
        )

    def _forward_logits(self, model, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone and reduce to per-example scalar logits."""
        labels = inputs["labels"]
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        logits = self._head_info.extract_logits(outputs, inputs["attention_mask"])
        return logits, labels

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        logits, labels = self._forward_logits(model, inputs)
        loss = self._loss_fn(logits.device, logits.dtype)(logits, labels.to(logits.dtype))
        return (loss, {"logits": logits}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = "labels" in inputs
        with torch.no_grad():
            logits, labels = self._forward_logits(model, inputs)
            loss = None
            if has_labels:
                loss = self._loss_fn(logits.device, logits.dtype)(
                    logits, labels.to(logits.dtype)
                ).detach()
            logits = logits.detach()
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)
