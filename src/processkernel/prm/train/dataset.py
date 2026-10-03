"""Job D's lists -> the encoded lists a LambdaRank step reads (PLAN_TRAINER §5).

    reads   {out_dir}/lists_{split}.jsonl, {out_dir}/prefixes.jsonl, v1's parts_glob
    yields  one item per LIST: ``{"cand_input_ids": [...], "rels": [...]}``

The unit is the list, not the prefix: LambdaRank ranks within a group, so every item of a group
has to reach the same batch. `ListwiseCollator` flattens them and carries `group_sizes` so the
trainer can split the scores back out -- which is why this yields exactly the two keys that
collator reads, and is used unchanged.

Unlike `orm/list_dataset.py`, the items of one list share no text: each is a prefix of a
different completion, so each needs its own forward pass and its own last-token score. That is
also why v1's per-position head machinery is not needed here (PLAN_v2 §5).
"""

from __future__ import annotations

import os
import random

from torch.utils.data import Dataset

from processkernel.config import RerankerConfig, _resolve
from processkernel.prm.train import encoding, rank_eval
from processkernel.prm.rollout import lists, prefixes, stage


class PRMListDataset(Dataset):
    """One campaign split's lists, each encoded lazily through `PrefixEncoder`."""

    def __init__(self, cfg: RerankerConfig, split: str, tokenizer, *,
                 max_lists: int = 0, subsample_seed: int = 42):
        conf = cfg.prm_rollout
        out_dir = _resolve(conf.out_dir)

        self.lists = lists.read_lists(os.path.join(out_dir, lists.LISTS.format(split=split)))
        if not self.lists:
            raise ValueError(
                f"no lists in {out_dir} for split {split!r} -- run processkernel.prm.rollout."
                "lists first. Training on an empty split runs zero steps and reports a "
                "finished run"
            )

        # Before the parts below, not after: `sources_for` parses every part a referenced
        # prefix touches and then holds a Source per completion for the life of the run, so
        # subsampling afterwards would pay both costs for lists it is about to discard.
        # Sorted back into file order because `_evaluate_lists` zips self.lists against an
        # unshuffled loader positionally -- the draw picks WHICH lists, never their order.
        if max_lists and len(self.lists) > max_lists:
            keep = sorted(random.Random(subsample_seed).sample(range(len(self.lists)), max_lists))
            self.lists = [self.lists[i] for i in keep]

        by_id = {p.prefix_id: p for p in stage.read_prefixes(
            os.path.join(out_dir, prefixes.PREFIXES)
        )}
        wanted = [item.prefix_id for lst in self.lists for item in lst.items]
        missing = [pid for pid in wanted if pid not in by_id]
        if missing:
            raise KeyError(
                f"{missing[0]} is an item of a {split} list with no row in "
                f"{prefixes.PREFIXES} -- the lists and the prefixes come from different "
                "builds, and the text to score it by is on the prefix"
            )
        self.prefixes = {pid: by_id[pid] for pid in wanted}
        # Resolved exactly the way job B and job E resolve them, through the same function:
        # a second lookup convention is how the trainer and the campaign drift apart with
        # nothing to notice (PLAN_v2 §6). Prefixes cut from one completion share one Source.
        self.sources = rank_eval.sources_for(conf, [by_id[pid] for pid in wanted])

        # The RERANKER's budget, never `model.max_length`, which bounds the ORM's ref + kernel
        # (ARCHITECTURE S4). The same budget job E encodes with, so the two are comparable.
        self.encoder = encoding.PrefixEncoder(tokenizer, conf.max_length)

    def __len__(self) -> int:
        return len(self.lists)

    def __getitem__(self, idx: int) -> dict:
        lst = self.lists[idx]
        return {
            "cand_input_ids": [
                self.encoder.encode(self.prefixes[it.prefix_id], self.sources[it.prefix_id])
                for it in lst.items
            ],
            "rels": [float(it.rel) for it in lst.items],
        }
