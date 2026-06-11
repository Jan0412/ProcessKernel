"""Torch dataset + collator for pairwise reranker training.

Each item is a (positive, negative) pair for the *same* problem, each side
encoded as a (reference architecture, candidate kernel) cross-encoder sequence
via the shared `SequenceEncoder` — identical to the pointwise encoding, so a
pairwise-trained model can be validated through the pointwise `RerankerDataset`.
"""

from __future__ import annotations

import json

from torch.utils.data import Dataset

from reranker.src.config import _resolve
from reranker.src.dataset import pad_sequences
from reranker.src.encoding import SequenceEncoder


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _row_key(run_name: str, level: int, problem_id: int, sample_id: int) -> tuple:
    return (run_name, int(level), int(problem_id), int(sample_id))


class PairwiseDataset(Dataset):
    """Loads a pairs JSONL and the source dataset; encodes each side lazily."""

    def __init__(
        self,
        pairs_jsonl: str,
        dataset_jsonl: str,
        tokenizer,
        max_length: int,
        reserve_ref_tokens: int,
    ):
        self.encoder = SequenceEncoder(tokenizer, max_length, reserve_ref_tokens)

        rows = _read_jsonl(_resolve(dataset_jsonl))
        self._by_key = {
            _row_key(r["run_name"], r["level"], r["problem_id"], r["sample_id"]): r
            for r in rows
        }
        self.pairs = _read_jsonl(_resolve(pairs_jsonl))
        if not self.pairs:
            raise ValueError(f"No pairs in {pairs_jsonl} — run reranker.src.pairwise.pairs first")

    def __len__(self) -> int:
        return len(self.pairs)

    def _lookup(self, pair: dict, side: str) -> dict:
        ref = pair[side]
        key = _row_key(ref["run_name"], pair["level"], pair["problem_id"], ref["sample_id"])
        row = self._by_key.get(key)
        if row is None:
            raise KeyError(f"Pair references a row not in the source dataset: {key}")
        return row

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        pos = self._lookup(pair, "pos")
        neg = self._lookup(pair, "neg")
        return {
            "pos_input_ids": self.encoder.encode(pos["ref_arch_src"], pos["kernel_src"]),
            "neg_input_ids": self.encoder.encode(neg["ref_arch_src"], neg["kernel_src"]),
        }


class PairwiseCollator:
    """Pads the positive and negative sides of a pair batch independently."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __call__(self, batch: list[dict]) -> dict:
        pos_ids, pos_mask = pad_sequences([ex["pos_input_ids"] for ex in batch], self.pad_id)
        neg_ids, neg_mask = pad_sequences([ex["neg_input_ids"] for ex in batch], self.pad_id)
        return {
            "pos_input_ids": pos_ids,
            "pos_attention_mask": pos_mask,
            "neg_input_ids": neg_ids,
            "neg_attention_mask": neg_mask,
        }
