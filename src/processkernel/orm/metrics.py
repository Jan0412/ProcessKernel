"""Evaluation metrics for the reranker.

Two families:
  * Classification metrics on the pointwise good/bad labels (accuracy, F1, AUCs).
  * Per-problem ranking metrics — the numbers that reflect real usage: for each
    problem we rank its candidate kernels by predicted score and ask whether the
    top pick is actually good. These are computed even though training is pointwise.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = _sigmoid(logits)
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)
    out = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "positive_rate": float(labels.mean()),
    }
    # AUCs are undefined when only one class is present in the eval split.
    if len(np.unique(labels)) == 2:
        out["roc_auc"] = roc_auc_score(labels, probs)
        out["pr_auc"] = average_precision_score(labels, probs)
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ranking_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: list[tuple],
) -> dict[str, float]:
    """Group candidates by problem, rank by score, score the top pick.

    pass_at_1   : fraction of problems whose top-scored candidate is good.
    recall_at_1 : among problems that HAVE a good candidate, fraction where the
                  top-scored pick is good (i.e. the reranker recovers a good one).
    ndcg        : mean NDCG of the score-induced ranking against binary labels.
    coverage    : fraction of problems that have at least one good candidate
                  (an upper bound on pass_at_1).
    """
    by_group: dict[tuple, list[tuple[float, int]]] = {}
    for score, label, group in zip(logits, labels, groups):
        by_group.setdefault(group, []).append((float(score), int(label)))

    pass1, ndcgs = [], []
    recall1, n_have_good = [], 0
    n_have_good_total = 0

    for cands in by_group.values():
        cands_sorted = sorted(cands, key=lambda x: x[0], reverse=True)
        ranked_labels = [lbl for _, lbl in cands_sorted]
        top_good = ranked_labels[0]
        pass1.append(top_good)

        has_good = any(lbl == 1 for lbl in ranked_labels)
        if has_good:
            n_have_good_total += 1
            recall1.append(top_good)

        ideal = sorted(ranked_labels, reverse=True)
        idcg = _dcg(ideal)
        ndcgs.append(_dcg(ranked_labels) / idcg if idcg > 0 else 0.0)

    n_groups = len(by_group)
    return {
        "pass_at_1": float(np.mean(pass1)) if pass1 else 0.0,
        "recall_at_1": float(np.mean(recall1)) if recall1 else 0.0,
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "coverage": n_have_good_total / n_groups if n_groups else 0.0,
        "num_problems": float(n_groups),
    }


def make_compute_metrics(groups: list[tuple]):
    """Build a `compute_metrics` callable bound to the eval set's group order.

    HuggingFace Trainer only hands `compute_metrics` the predictions and labels,
    not per-row group ids. Eval runs in dataset order without shuffling, so we
    capture the eval dataset's group list here in a closure.
    """

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        logits = np.asarray(logits).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        metrics = classification_metrics(logits, labels)
        metrics.update(ranking_metrics(logits, labels, groups))
        return metrics

    return compute_metrics
