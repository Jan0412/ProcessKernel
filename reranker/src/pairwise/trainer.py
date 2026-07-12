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

from reranker.src.pairwise.dataset import GROUP_IDS
from reranker.src.trainer import RerankerTrainer

LOSS_TYPES = ("logistic", "margin")


def _group_factors(alpha: float, group_mass: dict, n_pairs: int) -> torch.Tensor:
    """Constant per-group multipliers for the group-split weighted loss.

    Each pair's effective weight is ``factor[group] * pair_weight``, where

        factor[g] = alpha_g * n_pairs / mass[g]

    ``mass[g]`` is the group's total pair weight (a dataset constant) so dividing
    by it turns each group's contribution into a *weighted average* of its pairs'
    losses — independent of the group's pair count and of the raw magnitude of
    its rel gaps. ``alpha_g`` (``alpha`` for correctness, ``1 - alpha`` for speed)
    then balances the two, renormalized over the groups actually present so a
    missing group hands its full weight to the other (mirrors the listwise loss).
    The result averages to 1 over the dataset, so the loss scale matches the
    unweighted mean. Normalization is by these constants, never the batch's own
    sums, so the weighting survives gradient accumulation / DDP.
    """
    raw_alpha = {"correctness": float(alpha), "speed": 1.0 - float(alpha)}
    present = {g: raw_alpha[g] for g, m in group_mass.items() if m > 0}
    norm = sum(present.values()) or 1.0
    factors = torch.zeros(len(GROUP_IDS), dtype=torch.float)
    for g, gid in GROUP_IDS.items():
        mass = group_mass.get(g, 0.0)
        if mass > 0:
            factors[gid] = (raw_alpha[g] / norm) * n_pairs / mass
    return factors


class PairwiseTrainer(RerankerTrainer):
    def __init__(
        self,
        *args,
        loss_type: str = "logistic",
        margin: float = 1.0,
        weighted: bool = False,
        alpha: float = 0.5,
        group_weight_mass: dict | None = None,
        n_pairs: int = 0,
        pair_collator=None,
        eval_pairs_dataset=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {loss_type}")
        self._loss_type = loss_type
        self._margin = margin
        self._weighted = weighted
        # Constant per-group multipliers (indexed by the batch's `group` ids).
        self._group_factor = (
            _group_factors(alpha, group_weight_mass or {}, max(int(n_pairs), 1))
            if weighted else None
        )
        # `self.data_collator` (passed in) is the pointwise collator used for eval;
        # the pairwise collator is swapped in for the train dataloader.
        self._point_collator = self.data_collator
        self._pair_collator = pair_collator
        self._eval_pairs = eval_pairs_dataset

    # --- pairwise loss helpers ------------------------------------------------
    def _forward_pair(self, model, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward over the stacked ``[pos; neg]`` batch, split back per side.

        The collator concatenates the two sides into a single ``(2B, L)`` batch
        (first B = positives, last B = negatives), so each parameter is used
        exactly once per step — required for DDP + gradient checkpointing.
        """
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = self._head_info.extract_logits(outputs, inputs["attention_mask"])  # (2B,)
        pos_logits, neg_logits = logits.chunk(2, dim=0)
        return pos_logits, neg_logits

    def _pairwise_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Per-pair loss (no reduction) over the score difference s_pos - s_neg."""
        if self._loss_type == "logistic":
            return F.softplus(-diff)
        return F.relu(self._margin - diff)  # margin

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_logits, neg_logits = self._forward_pair(model, inputs)
        diff = pos_logits - neg_logits
        losses = self._pairwise_loss(diff)
        if self._weighted and "weight" in inputs:
            # Group-split + alpha: each pair's weight is scaled by a constant,
            # per-group factor (correctness vs speed) that normalizes the group
            # by its own mass and balances the two by alpha — so the correctness
            # rel gap (>= 1) can't swamp the small speed gaps. Constants, not the
            # batch's own sums, so it survives gradient accumulation / DDP.
            factor = self._group_factor.to(device=losses.device, dtype=losses.dtype)
            w = inputs["weight"] * factor[inputs["group"]]
            loss = (losses * w).mean()
        else:
            loss = losses.mean()
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
            # Reported eval loss is unweighted (plain mean) for comparability.
            total_loss += self._pairwise_loss(diff).sum().item()
            correct += (diff > 0).sum().item()
            n += diff.numel()
        if was_training:
            model.train()
        return {
            f"{prefix}_pair_loss": total_loss / max(n, 1),
            f"{prefix}_pair_acc": correct / max(n, 1),
        }
