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


def lambdarank_loss(scores: torch.Tensor, rels: torch.Tensor, sigma: float = 1.0,
                    alpha: float = 0.5):
    """LambdaRank loss for one query's candidate list.

    scores: (n,) model logits; rels: (n,) graded relevance (>= 0).

    ``alpha`` splits the ranking pairs into two groups and weights them:
    ``loss = alpha * correctness_term + (1 - alpha) * speed_term``, each term the
    mean over its own pairs. **correctness** pairs have the lower item wrong
    (``rel_j == 0``); **speed** pairs are both-correct (``rel_j > 0``). This stops
    the abundant correct-vs-wrong pairs from drowning the fast-vs-slow signal —
    ``alpha=0.5`` weights the two objectives equally regardless of their counts;
    lower ``alpha`` pushes harder on speed. Still LambdaRank: each pair keeps its
    ``ΔNDCG`` weight and the list-level ``idcg`` normalization. If only one group
    is present, that group carries the whole loss.

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
    if pos.sum() == 0:                                                  # all-equal relevance
        return None

    # |ΔNDCG_ij| from swapping i and j — a detached constant weight (LambdaRank).
    delta_ndcg = (torch.abs(gains.unsqueeze(1) - gains.unsqueeze(0))
                  * torch.abs(disc_at.unsqueeze(1) - disc_at.unsqueeze(0)) / idcg).detach()

    s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)                  # s_i - s_j
    pair_loss = delta_ndcg * F.softplus(-sigma * s_diff)               # log(1 + exp(-sigma*(s_i-s_j)))

    # Split ranking pairs into correctness (lower item wrong) and speed (both correct).
    j_wrong = (rels.unsqueeze(0) == 0).float()
    corr = pos * j_wrong                                                # correct-vs-wrong pairs
    speed = pos * (1.0 - j_wrong)                                       # fast-vs-slow pairs
    n_corr, n_speed = corr.sum(), speed.sum()

    terms, weights = [], []
    if n_corr > 0:
        terms.append((corr * pair_loss).sum() / n_corr)
        weights.append(alpha)
    if n_speed > 0:
        terms.append((speed * pair_loss).sum() / n_speed)
        weights.append(1.0 - alpha)
    # Renormalize weights so a missing group hands its full weight to the other.
    wsum = sum(weights)
    if wsum <= 0:                                                       # e.g. alpha=1 but only speed pairs
        return torch.stack(terms).mean()
    return sum(w / wsum * t for w, t in zip(weights, terms))


def _graded_ndcg(scores: torch.Tensor, rels: torch.Tensor) -> float:
    """NDCG of the score-induced ranking against the graded relevances.

    Uses the same ``gain = 2^rel - 1`` and ``1/log2(rank+2)`` discount as the
    LambdaRank loss, so it rewards ranking *fast* correct kernels above slow ones
    (unlike the binary NDCG in metrics.py). Returns 0 when there is no positive
    relevance (idcg == 0).
    """
    n = scores.shape[0]
    if n < 1:
        return 0.0
    gains = torch.pow(2.0, rels.float()) - 1.0
    discounts = 1.0 / torch.log2(torch.arange(n, device=scores.device, dtype=torch.float) + 2.0)
    idcg = (torch.sort(gains, descending=True).values * discounts).sum()
    if idcg <= 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    dcg = (gains[order] * discounts).sum()
    return float((dcg / idcg).item())


class ListwiseTrainer(RerankerTrainer):
    def __init__(
        self,
        *args,
        sigma: float = 1.0,
        alpha: float = 0.5,
        list_collator=None,
        eval_lists_dataset=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._sigma = sigma
        self._alpha = alpha
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
            loss_q = lambdarank_loss(s_q, r_q, sigma=self._sigma, alpha=self._alpha)
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

    # --- evaluation: listwise-only, speed-aware / @1 metrics ------------------
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Listwise-only evaluation.

        We deliberately do NOT call the parent (pointwise) eval loop: the binary
        classification / binary-NDCG metrics are not the objective and doubled the
        eval cost. Instead we run a single pass over the val *lists* and report
        speed-aware / @1 metrics. We still log the metrics and fire the eval
        callbacks so HF's best-model / early-stopping bookkeeping keeps working.
        """
        metrics = self._evaluate_lists(metric_key_prefix)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    @torch.no_grad()
    def _evaluate_lists(self, prefix: str) -> dict[str, float]:
        """Per-problem (per-list) speed-aware metrics, averaged over val lists.

        A candidate's relevance encodes both objectives: ``rel == 0`` is wrong,
        ``rel > 0`` is correct with speed grade ``p = rel - 1``.
        """
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

        total_loss, n_loss_lists = 0.0, 0
        ndcgs: list[float] = []
        speed_correct = speed_total = 0
        corr_correct = corr_total = 0
        n_lists = 0
        top1_correct = 0
        regret_sum, n_regret = 0.0, 0
        fast_sum, n_fast = 0.0, 0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            scores = self._forward_list_logits(model, batch)
            sizes = batch["group_sizes"].tolist()
            for s_q, r_q in zip(torch.split(scores, sizes), torch.split(batch["rels"], sizes)):
                loss_q = lambdarank_loss(s_q, r_q, sigma=self._sigma, alpha=self._alpha)
                if loss_q is not None:
                    total_loss += loss_q.item()
                    n_loss_lists += 1

                s, r = s_q.float(), r_q.float()
                if r.shape[0] < 1:
                    continue
                n_lists += 1

                # Graded NDCG of the score-induced order vs the graded relevances.
                ndcgs.append(_graded_ndcg(s, r))

                # Pairwise accuracy, split by pair type (correct-vs-wrong vs fast-vs-slow).
                more = r.unsqueeze(1) > r.unsqueeze(0)             # i more relevant than j
                better = (s.unsqueeze(1) - s.unsqueeze(0)) > 0     # i scored above j
                j_wrong = (r.unsqueeze(0) == 0)                    # lower side is wrong
                corr_mask = more & j_wrong
                speed_mask = more & (~j_wrong)
                corr_correct += (corr_mask & better).sum().item()
                corr_total += corr_mask.sum().item()
                speed_correct += (speed_mask & better).sum().item()
                speed_total += speed_mask.sum().item()

                # @1 metrics: the model picks the single top-scored candidate.
                top_rel = r[int(torch.argmax(s).item())].item()
                picked_p = top_rel - 1.0 if top_rel > 0 else 0.0   # 0 if the gate fails
                top1_correct += 1 if top_rel > 0 else 0
                best_rel = r.max().item()
                if best_rel > 0:                                   # list has a correct kernel
                    best_p = best_rel - 1.0
                    regret_sum += best_p - picked_p
                    n_regret += 1
                    if best_p > 0:                                 # some speed headroom to earn
                        fast_sum += picked_p / best_p
                        n_fast += 1

        if was_training:
            model.train()

        return {
            f"{prefix}_list_loss": total_loss / max(n_loss_lists, 1),
            f"{prefix}_list_ndcg_graded": float(sum(ndcgs) / max(len(ndcgs), 1)),
            f"{prefix}_speed_pair_acc": speed_correct / max(speed_total, 1),
            f"{prefix}_correctness_pair_acc": corr_correct / max(corr_total, 1),
            f"{prefix}_top1_correct": top1_correct / max(n_lists, 1),
            f"{prefix}_speed_regret_at1": regret_sum / max(n_regret, 1),
            f"{prefix}_fast_at1": fast_sum / max(n_fast, 1),
        }
