"""Pairwise-loss trainer for the reranker.

`PairwiseTrainer` subclasses the pointwise `RerankerTrainer` so it inherits the
single-candidate scoring path used for validation (`prediction_step`,
`_forward_logits`). Training, however, runs on (positive, negative) pairs and
optimizes a pairwise loss over the score difference `s_pos - s_neg`:

    logistic : softplus(-(s_pos - s_neg))   (BPR / RankNet, no hyperparameter)
    margin   : relu(margin - (s_pos - s_neg))

No BCE is involved. Validation reuses the pointwise ranking metrics (for model
selection / comparability) and additionally reports `eval_pair_loss` /
`eval_pair_acc` over a held-out set of validation pairs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reranker.src.trainer import RerankerTrainer

LOSS_TYPES = ("logistic", "margin")


class PairwiseTrainer(RerankerTrainer):
    def __init__(
        self,
        *args,
        loss_type: str = "logistic",
        margin: float = 1.0,
        pair_collator=None,
        eval_pairs_dataset=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {loss_type}")
        self._loss_type = loss_type
        self._margin = margin
        # `self.data_collator` (passed in) is the pointwise collator used for eval;
        # the pairwise collator is swapped in for the train dataloader.
        self._point_collator = self.data_collator
        self._pair_collator = pair_collator
        self._eval_pairs = eval_pairs_dataset

    # --- pairwise loss helpers ------------------------------------------------
    def _forward_pair(self, model, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        pos_out = model(
            input_ids=inputs["pos_input_ids"],
            attention_mask=inputs["pos_attention_mask"],
        )
        pos_logits = self._head_info.extract_logits(pos_out, inputs["pos_attention_mask"])
        neg_out = model(
            input_ids=inputs["neg_input_ids"],
            attention_mask=inputs["neg_attention_mask"],
        )
        neg_logits = self._head_info.extract_logits(neg_out, inputs["neg_attention_mask"])
        return pos_logits, neg_logits

    def _pairwise_loss(self, diff: torch.Tensor) -> torch.Tensor:
        if self._loss_type == "logistic":
            return F.softplus(-diff).mean()
        return F.relu(self._margin - diff).mean()  # margin

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_logits, neg_logits = self._forward_pair(model, inputs)
        diff = pos_logits - neg_logits
        loss = self._pairwise_loss(diff)
        return (loss, {"diff": diff}) if return_outputs else loss

    # --- dataloaders: pairwise for train, pointwise for eval ------------------
    def get_train_dataloader(self):
        self.data_collator = self._pair_collator
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        self.data_collator = self._point_collator
        return super().get_eval_dataloader(eval_dataset)

    # --- evaluation: pointwise ranking metrics + pairwise loss/acc ------------
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if self._eval_pairs is not None:
            pair_metrics = self._evaluate_pairs(metric_key_prefix)
            metrics.update(pair_metrics)
            self.log(pair_metrics)
        return metrics

    @torch.no_grad()
    def _evaluate_pairs(self, prefix: str) -> dict[str, float]:
        loader = DataLoader(
            self._eval_pairs,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self._pair_collator,
            num_workers=self.args.dataloader_num_workers,
        )
        model = self.model
        was_training = model.training
        model.eval()
        device = self.args.device
        total_loss, correct, n = 0.0, 0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pos_logits, neg_logits = self._forward_pair(model, batch)
            diff = pos_logits - neg_logits
            total_loss += self._pairwise_loss(diff).item() * diff.numel()
            correct += (diff > 0).sum().item()
            n += diff.numel()
        if was_training:
            model.train()
        return {
            f"{prefix}_pair_loss": total_loss / max(n, 1),
            f"{prefix}_pair_acc": correct / max(n, 1),
        }
