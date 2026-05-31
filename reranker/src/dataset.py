"""Torch dataset + collator for the reranker.

Each example is a (reference architecture, candidate kernel) pair formatted as a
single sequence for a cross-encoder. Truncation keeps the full reference where
possible (up to `reserve_ref_tokens`) and truncates the candidate kernel's tail,
since kernels can be long and the head carries the imports / structure.
"""

from __future__ import annotations

import json
from typing import Optional

import torch
from torch.utils.data import Dataset

from reranker.src.config import _resolve
from reranker.src.data.splits import load_splits

INSTRUCTION = (
    "You are judging whether a generated GPU kernel is a correct and faster "
    "drop-in replacement for the given PyTorch reference architecture.\n"
    "Reference architecture:\n"
)
SEPARATOR = "\n\nCandidate kernel:\n"


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

        splits = load_splits(_resolve(splits_json))
        all_rows = _read_jsonl(_resolve(dataset_jsonl))
        self.rows = [
            r for r in all_rows
            if splits.get((r["level"], r["problem_id"])) == split
        ]
        if not self.rows:
            raise ValueError(f"No rows for split '{split}' — check splits.json / dataset.jsonl")

        # Precompute static instruction / separator token ids (no special tokens).
        self._instr_ids = tokenizer.encode(INSTRUCTION, add_special_tokens=False)
        self._sep_ids = tokenizer.encode(SEPARATOR, add_special_tokens=False)

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
        tok = self.tokenizer
        ref_ids = tok.encode(ref_src, add_special_tokens=False)
        kernel_ids = tok.encode(kernel_src, add_special_tokens=False)

        # Terminate each example with a single EOS: the seq-cls head scores the
        # last non-pad token, so we give every sequence a consistent sentinel.
        # (Qwen tokenizers no longer expose build_inputs_with_special_tokens, so
        # we assemble the input ids explicitly rather than relying on it.)
        eos_id = tok.eos_token_id
        num_special = 1 if eos_id is not None else 0
        budget = self.max_length - num_special - len(self._instr_ids) - len(self._sep_ids)
        budget = max(budget, 0)

        ref_keep = min(len(ref_ids), self.reserve_ref_tokens, budget)
        ref_ids = ref_ids[:ref_keep]
        kernel_keep = max(budget - len(ref_ids), 0)
        kernel_ids = kernel_ids[:kernel_keep]

        content = self._instr_ids + ref_ids + self._sep_ids + kernel_ids
        if eos_id is not None:
            content.append(eos_id)
        return content

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        input_ids = self._encode_pair(row["ref_arch_src"], row["kernel_src"])
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": float(row["label"]),
        }


class RerankerCollator:
    """Pads a batch of variable-length examples; keeps `labels` as a float tensor."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            pad = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.pad_id] * pad)
            attention_mask.append(ex["attention_mask"] + [0] * pad)
            labels.append(ex["labels"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
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
