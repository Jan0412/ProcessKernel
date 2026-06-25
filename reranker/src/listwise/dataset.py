"""Torch dataset + collator for listwise reranker training.

Each item is one problem's candidate *list*: a set of (reference architecture,
candidate kernel) cross-encoder sequences encoded via the shared
`SequenceEncoder` (identical to the pointwise/pairwise encoding, so a
listwise-trained model is validated through the pointwise `RerankerDataset`),
plus a graded relevance per candidate.

The collator flattens all candidates of all lists in a batch into one padded
tensor and carries a `group_sizes` vector so the trainer can split the per-list
scores back out for the LambdaRank loss.
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


class ListwiseDataset(Dataset):
    """Loads a lists JSONL and the source dataset; encodes each candidate lazily."""

    def __init__(
        self,
        lists_jsonl: str,
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
        self.lists = _read_jsonl(_resolve(lists_jsonl))
        if not self.lists:
            raise ValueError(f"No lists in {lists_jsonl} — run reranker.src.listwise.lists first")

    def __len__(self) -> int:
        return len(self.lists)

    def _lookup(self, lst: dict, cand: dict) -> dict:
        key = _row_key(cand["run_name"], lst["level"], lst["problem_id"], cand["sample_id"])
        row = self._by_key.get(key)
        if row is None:
            raise KeyError(f"List references a row not in the source dataset: {key}")
        return row

    def __getitem__(self, idx: int) -> dict:
        lst = self.lists[idx]
        cand_input_ids, rels = [], []
        for cand in lst["candidates"]:
            row = self._lookup(lst, cand)
            cand_input_ids.append(self.encoder.encode(row["ref_arch_src"], row["kernel_src"]))
            rels.append(float(cand["rel"]))
        return {"cand_input_ids": cand_input_ids, "rels": rels}


class ListwiseCollator:
    """Flatten all candidates of all lists in the batch into one padded tensor.

    `group_sizes[i]` is the number of candidates in list `i`; it sums to the
    flattened batch size, letting the trainer `torch.split` scores per list.
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __call__(self, batch: list[dict]) -> dict:
        all_seqs: list[list[int]] = []
        all_rels: list[float] = []
        group_sizes: list[int] = []
        for ex in batch:
            all_seqs.extend(ex["cand_input_ids"])
            all_rels.extend(ex["rels"])
            group_sizes.append(len(ex["cand_input_ids"]))
        input_ids, attention_mask = pad_sequences(all_seqs, self.pad_id)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "rels": torch.tensor(all_rels, dtype=torch.float),
            "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
        }
