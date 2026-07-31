"""Torch dataset + collator for pairwise reranker training.

Each item is a (positive, negative) pair for the *same* problem, each side
encoded as a (reference architecture, candidate kernel) cross-encoder sequence
via the shared `SequenceEncoder` — identical to the pointwise encoding, so a
pairwise-trained model can be validated through the pointwise `RerankerDataset`.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset

from reranker.src.config import _resolve
from reranker.src.dataset import pad_sequences
from reranker.src.encoding import SequenceEncoder


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _row_key(run_name: str, level: int, problem_id: int, sample_id: int) -> tuple:
    return (run_name, int(level), int(problem_id), int(sample_id))


# Integer ids for the two pair groups the weighted loss balances separately.
GROUP_IDS = {"correctness": 0, "speed": 1}


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

    @property
    def group_weight_mass(self) -> dict[str, float]:
        """Total pair weight per group (correctness vs speed).

        These dataset-level constants let the trainer normalize each group by its
        own mass — so the correctness offset (rel gap >= 1) can't swamp the far
        smaller speed gaps — and then balance the two with `alpha` (group-split,
        the pairwise analogue of the listwise loss). Constant, not per-batch, so
        the weighting survives gradient accumulation / DDP.
        """
        masses = {g: 0.0 for g in GROUP_IDS}
        for p in self.pairs:
            masses[p.get("group", "correctness")] += float(p.get("weight", 1.0))
        return masses

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
            "weight": float(pair.get("weight", 1.0)),
            "group": GROUP_IDS[pair.get("group", "correctness")],
        }


class PairwiseCollator:
    """Stacks the positive and negative sides into one ``(2B, L)`` batch.

    Both sides are padded to a single common length and concatenated as
    ``[pos_0..pos_{B-1}, neg_0..neg_{B-1}]`` so the trainer scores a pair with a
    **single** forward pass (then splits the logits back). One forward keeps each
    parameter used exactly once per step, which DDP + gradient checkpointing
    require (two separate forwards mark params ready twice and crash under DDP).
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __call__(self, batch: list[dict]) -> dict:
        seqs = [ex["pos_input_ids"] for ex in batch] + [ex["neg_input_ids"] for ex in batch]
        input_ids, attention_mask = pad_sequences(seqs, self.pad_id)
        return {
            "input_ids": input_ids,            # (2B, L): rows [0:B]=pos, [B:2B]=neg
            "attention_mask": attention_mask,
            "weight": torch.tensor([ex["weight"] for ex in batch], dtype=torch.float),
            "group": torch.tensor([ex["group"] for ex in batch], dtype=torch.long),
        }
