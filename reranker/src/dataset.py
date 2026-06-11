"""Torch dataset + collator for the reranker.

Each example is a (reference architecture, candidate kernel) pair formatted as a
single sequence for a cross-encoder. The (ref, kernel) -> input_ids encoding
(including truncation) lives in `reranker.src.encoding.SequenceEncoder`, shared
with the pairwise dataset so both code paths encode identically.
"""

from __future__ import annotations

import json
from typing import Optional

import torch
from torch.utils.data import Dataset

from reranker.src.config import _resolve
from reranker.src.data.splits import load_splits
from reranker.src.encoding import INSTRUCTION, SEPARATOR, SequenceEncoder  # noqa: F401


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class RerankerDataset(Dataset):
    """Loads JSONL rows for one split and tokenizes (ref, kernel) pairs lazily."""

    def __init__(
        self,
        dataset_jsonl: str,
        splits_json: str,
        split: str,
        tokenizer,
        max_length: int,
        reserve_ref_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reserve_ref_tokens = reserve_ref_tokens
        self.encoder = SequenceEncoder(tokenizer, max_length, reserve_ref_tokens)

        splits = load_splits(_resolve(splits_json))
        all_rows = _read_jsonl(_resolve(dataset_jsonl))
        self.rows = [
            r for r in all_rows
            if splits.get((r["level"], r["problem_id"])) == split
        ]
        if not self.rows:
            raise ValueError(f"No rows for split '{split}' — check splits.json / dataset.jsonl")

    # --- grouping / label helpers (used by compute_metrics) -------------------
    @property
    def groups(self) -> list[tuple[int, int]]:
        return [(r["level"], r["problem_id"]) for r in self.rows]

    @property
    def labels(self) -> list[int]:
        return [r["label"] for r in self.rows]

    def label_balance(self) -> dict[str, int]:
        pos = sum(self.labels)
        return {"total": len(self.rows), "positive": pos, "negative": len(self.rows) - pos}

    # --- torch Dataset API ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def _encode_pair(self, ref_src: str, kernel_src: str) -> list[int]:
        return self.encoder.encode(ref_src, kernel_src)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        input_ids = self._encode_pair(row["ref_arch_src"], row["kernel_src"])
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": float(row["label"]),
        }


def pad_sequences(
    seqs: list[list[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of token-id sequences; return (input_ids, attention_mask)."""
    max_len = max(len(s) for s in seqs)
    input_ids, attention_mask = [], []
    for s in seqs:
        pad = max_len - len(s)
        input_ids.append(s + [pad_id] * pad)
        attention_mask.append([1] * len(s) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


class RerankerCollator:
    """Pads a batch of variable-length examples; keeps `labels` as a float tensor."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __call__(self, batch: list[dict]) -> dict:
        input_ids, attention_mask = pad_sequences(
            [ex["input_ids"] for ex in batch], self.pad_id
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor([ex["labels"] for ex in batch], dtype=torch.float),
        }


def build_datasets(
    cfg,
    tokenizer,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, RerankerDataset]:
    """Build a RerankerDataset per requested split."""
    return {
        split: RerankerDataset(
            dataset_jsonl=cfg.data.dataset_jsonl,
            splits_json=cfg.data.splits_json,
            split=split,
            tokenizer=tokenizer,
            max_length=cfg.model.max_length,
            reserve_ref_tokens=cfg.model.reserve_ref_tokens,
        )
        for split in splits
    }
