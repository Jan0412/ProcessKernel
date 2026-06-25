"""LambdaRank (listwise) trainer for the reranker.

`ListwiseTrainer` subclasses the pointwise `RerankerTrainer` so it inherits the
single-candidate scoring path used for validation (`prediction_step`,
`_forward_logits`). Training, however, runs on per-problem candidate *lists*: a
single forward over all candidates of the batch's lists, split back per list by
`group_sizes`, then a LambdaRank loss over each list optimizing the
NDCG-weighted ordering implied by the graded relevances.

The model stays a pointwise scorer (one scalar per candidate, no cross-candidate
attention), so a listwise-trained model is validated through the pointwise
`RerankerDataset` and ranks candidate pools of any size at inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reranker.src.trainer import RerankerTrainer


def lambdarank_loss(scores: torch.Tensor, rels: torch.Tensor, sigma: float = 1.0):
    """LambdaRank loss for one query's candidate list.

    scores: (n,) model logits; rels: (n,) graded relevance (>= 0).
    Returns a scalar loss, or None if the list has no valid pair to rank
    (fewer than 2 candidates, no positive relevance, or all-equal relevance).
    """
    n = scores.shape[0]
    if n < 2:
        return None
    scores = scores.float()
    rels = rels.float()

    gains = torch.pow(2.0, rels) - 1.0                                   # (n,)
    ideal, _ = torch.sort(gains, descending=True)
    idx = torch.arange(n, device=scores.device, dtype=torch.float)
    discounts = 1.0 / torch.log2(idx + 2.0)
    idcg = (ideal * discounts).sum()
    if idcg <= 0:                                                        # no positives
        return None

    # Discount at each item's current (score-induced) rank; 0 = best.
    order = torch.argsort(scores, descending=True)
    rank = torch.empty(n, device=scores.device, dtype=torch.long)
    rank[order] = torch.arange(n, device=scores.device)
    disc_at = 1.0 / torch.log2(rank.float() + 2.0)                      # (n,)

    pos = (rels.unsqueeze(1) > rels.unsqueeze(0)).float()              # i more relevant than j
    n_pairs = pos.sum()
    if n_pairs == 0:                                                    # all-equal relevance
        return None

    # |ΔNDCG_ij| from swapping i and j — a detached constant weight (LambdaRank).
    delta_ndcg = (torch.abs(gains.unsqueeze(1) - gains.unsqueeze(0))
                  * torch.abs(disc_at.unsqueeze(1) - disc_at.unsqueeze(0)) / idcg).detach()

    s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)                  # s_i - s_j
    losses = pos * delta_ndcg * F.softplus(-sigma * s_diff)            # log(1 + exp(-sigma*(s_i-s_j)))
    return losses.sum() / n_pairs


class ListwiseTrainer(RerankerTrainer):
    def __init__(
        self,
        *args,
        sigma: float = 1.0,
        list_collator=None,
        eval_lists_dataset=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._sigma = sigma
        # `self.data_collator` (passed in) is the pointwise collator used for eval;
        # the listwise collator is swapped in for the train dataloader.
        self._point_collator = self.data_collator
        self._list_collator = list_collator
        self._eval_lists = eval_lists_dataset

    # --- forward / loss -------------------------------------------------------
    def _forward_list_logits(self, model, inputs) -> torch.Tensor:
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        return self._head_info.extract_logits(outputs, inputs["attention_mask"])

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # `**kwargs` absorbs HF's `num_items_in_batch` (transformers>=5.9.0); we
        # average over queries, not tokens, so it is intentionally unused.
        scores = self._forward_list_logits(model, inputs)              # (total_cand,)
        sizes = inputs["group_sizes"].tolist()
        losses = []
        for s_q, r_q in zip(torch.split(scores, sizes), torch.split(inputs["rels"], sizes)):
            loss_q = lambdarank_loss(s_q, r_q, sigma=self._sigma)
            if loss_q is not None:
                losses.append(loss_q)
        if losses:
            loss = torch.stack(losses).mean()
        else:
            # Whole batch degenerate (no rankable pair): keep the graph, zero grad.
            loss = scores.sum() * 0.0
        return (loss, {"scores": scores}) if return_outputs else loss

    # --- dataloaders: listwise for train, pointwise for eval ------------------
    def get_train_dataloader(self):
        self.data_collator = self._list_collator
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        self.data_collator = self._point_collator
        return super().get_eval_dataloader(eval_dataset)

    # --- evaluation: pointwise ranking metrics + listwise loss/pair-acc -------
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if self._eval_lists is not None:
            list_metrics = self._evaluate_lists(metric_key_prefix)
            metrics.update(list_metrics)
            self.log(list_metrics)
        return metrics

    @torch.no_grad()
    def _evaluate_lists(self, prefix: str) -> dict[str, float]:
        loader = DataLoader(
            self._eval_lists,
            batch_size=1,                       # one list per batch (variable sizes)
            collate_fn=self._list_collator,
            num_workers=self.args.dataloader_num_workers,
        )
        model = self.model
        was_training = model.training
        model.eval()
        device = self.args.device
        total_loss, n_lists = 0.0, 0
        correct_pairs, total_pairs = 0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            scores = self._forward_list_logits(model, batch)
            sizes = batch["group_sizes"].tolist()
            for s_q, r_q in zip(torch.split(scores, sizes), torch.split(batch["rels"], sizes)):
                loss_q = lambdarank_loss(s_q, r_q, sigma=self._sigma)
                if loss_q is not None:
                    total_loss += loss_q.item()
                    n_lists += 1
                s, r = s_q.float(), r_q.float()
                pos = r.unsqueeze(1) > r.unsqueeze(0)
                better = (s.unsqueeze(1) - s.unsqueeze(0)) > 0
                correct_pairs += (pos & better).sum().item()
                total_pairs += pos.sum().item()
        if was_training:
            model.train()
        return {
            f"{prefix}_list_loss": total_loss / max(n_lists, 1),
            f"{prefix}_list_pair_acc": correct_pairs / max(total_pairs, 1),
        }
