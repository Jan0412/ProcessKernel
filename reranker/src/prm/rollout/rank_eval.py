"""Does an existing checkpoint rank the measured prefixes? Job E (PLAN_v2 §6, §11).

    python -m reranker.src.prm.rollout.rank_eval --config <cfg> --checkpoint <v1>

**Nothing here trains.** One inference pass over ``lists_val.jsonl``, reporting pairwise
accuracy and graded NDCG per depth bucket. It is the acceptance test for the data product:
the V̂-spread gate says the lists *contain* variation, this says whether that variation is
learnable signal or noise -- and a campaign that passes the first and fails the second has
produced an expensive dataset with nothing in it.

Every headline number is sliced by depth, never averaged over it: shallow prefixes have tiny
true gaps and maximal estimator noise, deep ones the reverse, and one mean hides both.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from reranker.src.config import _resolve, load_config
from reranker.src.dataset import pad_sequences

# Private on purpose, and imported anyway (§6): reusing the trainer's own NDCG is what makes
# this plan's number and a future trainer's one statistic rather than two that agree until
# one is edited. Nothing under reranker/src/listwise/ is modified to get it.
from reranker.src.listwise.trainer import _graded_ndcg
from reranker.src.prm.build import write_atomic
from reranker.src.prm.rollout import encoding, lists, prefixes, rollout, stage
# Re-exported, not redefined: stats.py slices by the same bands and reads the same files, and
# must not import torch to do either -- so the bands live in prefixes.py, which already owns
# the depth window, and the reader beside the writer in lists.py (§6).
from reranker.src.prm.rollout.lists import read_lists
from reranker.src.prm.rollout.prefixes import VAL, bucket_edges, bucket_of

REPORT = "rank_eval_report.json"


@dataclass(frozen=True)
class ListResult:
    """One list's ranking, as the report accumulates it."""

    ndcg: float
    pairs_correct: int
    pairs_total: int


def list_metrics(scores: list[float], rels: list[float]) -> ListResult:
    """One list's NDCG and its ranking-pair tally.

    A pair is (i, j) with ``rel_i > rel_j``; it is right when ``s_i > s_j``. Strictly, as in
    `listwise/trainer.py::_evaluate_lists` -- a model that cannot separate two prefixes has
    not ranked them, and crediting the tie would flatter a scorer that returns a constant.
    """
    s = torch.tensor(scores, dtype=torch.float)
    r = torch.tensor(rels, dtype=torch.float)
    more = r.unsqueeze(1) > r.unsqueeze(0)
    better = (s.unsqueeze(1) - s.unsqueeze(0)) > 0
    return ListResult(
        ndcg=_graded_ndcg(s, r),
        pairs_correct=int((more & better).sum().item()),
        pairs_total=int(more.sum().item()),
    )


def report(rows, scores: dict[str, float], cfg) -> dict:
    """The ranking report: overall, then one entry per depth band.

    Accuracy **pools pairs** rather than averaging lists. Pairs are the unit of evidence -- a
    4-item list carries six and a 2-item list one -- so a mean of per-list accuracies would
    weight the pair as heavily as the list.
    """
    edges = bucket_edges(cfg)
    per_band: list[list[ListResult]] = [[] for _ in range(len(edges) - 1)]
    n_items = [0] * len(per_band)
    outside = two_item = 0

    for lst in rows:
        missing = [it.prefix_id for it in lst.items if it.prefix_id not in scores]
        if missing:
            raise KeyError(
                f"{missing[0]} is an item of {lst.list_key} that was never scored -- the "
                "report would then be computed over part of a list while reporting the "
                "whole one"
            )
        band = bucket_of(lst.rel_depth_mean, edges)
        per_band[band].append(
            list_metrics([scores[it.prefix_id] for it in lst.items],
                         [it.rel for it in lst.items])
        )
        n_items[band] += len(lst.items)
        outside += not edges[0] <= lst.rel_depth_mean <= edges[-1]
        two_item += len(lst.items) == 2

    every = [m for band in per_band for m in band]
    return {
        "lists": len(rows),
        "items": sum(n_items),
        "pairs": sum(m.pairs_total for m in every),
        "pairwise_acc": _acc(every),
        "ndcg": _ndcg(every),
        # A corpus that is mostly pairs is a pairwise dataset wearing a listwise schema (§2).
        "two_item_lists": two_item,
        # Nonzero means the report's depth window is not the one the campaign was cut under.
        "lists_outside_window": outside,
        "buckets": [
            _band(edges[i], edges[i + 1], per_band[i], n_items[i]) for i in range(len(per_band))
        ],
    }


def _band(lo: float, hi: float, results: list[ListResult], n_items: int) -> dict:
    """One depth band. Reported even when empty, with ``None`` where a metric has no input.

    ``0.0`` would read as "the model ranks nothing right here", which is a different claim
    from "nothing was measured here" -- and the second is the one an empty band makes.
    """
    return {
        "lo": lo,
        "hi": hi,
        "n_lists": len(results),
        "n_items": n_items,
        "n_pairs": sum(m.pairs_total for m in results),
        "pairwise_acc": _acc(results),
        "ndcg": _ndcg(results),
    }


def _acc(results: list[ListResult]) -> float | None:
    total = sum(m.pairs_total for m in results)
    return sum(m.pairs_correct for m in results) / total if total else None


def _ndcg(results: list[ListResult]) -> float | None:
    return sum(m.ndcg for m in results) / len(results) if results else None


# --- the pass: val lists + a checkpoint -> the ranking report -----------------------------


def sources_for(rollout_cfg, prefix_rows) -> dict[str, rollout.Source]:
    """``prefix_id -> the v1 texts it was cut from``, one part read per unit.

    Resolved exactly the way job B resolves them, through the same three functions: the PRM
    must be shown at scoring time the text the sampler held at generation time, and a second
    lookup convention here is a way for the two to drift apart with nothing to notice.
    """
    prompts = rollout.load_prompts(rollout_cfg.parts_glob)
    by_name = {os.path.basename(p): p for p in glob.glob(_resolve(rollout_cfg.parts_glob))}
    out: dict[str, rollout.Source] = {}
    for unit, ps in rollout.units(prefix_rows).items():
        part = by_name.get(f"{unit}.jsonl")
        if part is None:
            raise FileNotFoundError(
                f"{unit} has prefixes but no v1 part under {rollout_cfg.parts_glob} to read "
                "their texts from -- job A and job E are reading different builds"
            )
        srcs = rollout.load_sources(part, prompts)
        for p in ps:
            key = rollout.source_key(p)
            if key not in srcs:
                raise KeyError(
                    f"{p.prefix_id} names {key}, which {os.path.basename(part)} does not "
                    "hold -- point parts_glob at the build job A read"
                )
            out[p.prefix_id] = srcs[key]
    return out


def batched(ids: list[list[int]], forward, batch_size: int) -> list[float]:
    """Score every sequence, longest first, and hand the scores back in the caller's order.

    Sorted by length because a batch pads to its longest member and `max_length` is 16,384:
    mixing a 400-token prefix with a 12,000-token one pays for the difference on every row.
    The realignment is the part that has to be right -- a permuted return would score each
    prefix with another prefix's number and every metric downstream would still compute.
    """
    if batch_size < 1:
        raise ValueError(f"rank_eval needs a batch size >= 1, got {batch_size}")
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]), reverse=True)
    out: list[float] = [0.0] * len(ids)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        for i, score in zip(chunk, forward([ids[i] for i in chunk])):
            out[i] = score
    return out


def rank_eval(cfg, *, scorer=None, tokenizer=None, checkpoint: str | None = None) -> dict:
    """Score `lists_val.jsonl` with an existing checkpoint. Trains nothing (§6).

    **Val only.** N3 fixes val selection to random, which is what makes the number
    trustworthy; the train half may be scored-selected in a later campaign, and an accuracy
    computed over prefixes a PRM chose because it was confident about them would be inflated
    with no way to tell from the output.
    """
    conf = cfg.prm_rollout
    conf.validate()
    out_dir = _resolve(conf.out_dir)

    rows = read_lists(os.path.join(out_dir, lists.LISTS.format(split=VAL)))
    by_id = {p.prefix_id: p for p in stage.read_prefixes(os.path.join(out_dir, prefixes.PREFIXES))}
    wanted = [item.prefix_id for lst in rows for item in lst.items]
    missing = [pid for pid in wanted if pid not in by_id]
    if missing:
        raise KeyError(
            f"{missing[0]} is an item of a val list with no row in {prefixes.PREFIXES} -- the "
            f"lists and the prefixes come from different builds, and the text to score it by "
            "is on the prefix"
        )

    srcs = sources_for(conf, [by_id[pid] for pid in wanted])
    if scorer is None:
        scorer, tokenizer = load_scorer(cfg, checkpoint)
    elif tokenizer is None:
        raise ValueError(
            "a scorer arrives with the tokenizer that encoded for it: it is handed ids, and "
            "which tokenizer produced them is not recoverable from them"
        )
    # The RERANKER's tokenizer, never the generation model's that sizes a rollout's budget in
    # rollout.py -- they are different models and mixing them mis-sizes silently (§13 step 2).
    encoder = encoding.PrefixEncoder(tokenizer, conf.max_length)
    values = scorer([encoder.encode(by_id[pid], srcs[pid]) for pid in wanted])
    if len(values) != len(wanted):
        # zip stops at the shorter side, so the tail would go unscored and `report` would
        # blame the lists for a scorer that returned short.
        raise ValueError(
            f"the scorer returned {len(values)} scores for {len(wanted)} items"
        )
    scores = dict(zip(wanted, values))

    out = report(rows, scores, conf)
    out |= {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": checkpoint,
        "config": dataclasses.asdict(conf),
    }
    write_atomic(os.path.join(out_dir, REPORT), json.dumps(out, indent=2))
    return out


# --- the one part that needs a GPU -------------------------------------------------------


def head_of(cfg, checkpoint: str, tokenizer):
    """How this checkpoint's outputs become one scalar -- read, not guessed.

    ``reranker_head.json`` is what a training run wrote down, and it wins: that is the whole
    of "checkpoint-agnostic". Only a checkpoint that saved none falls back to the config in
    hand, which is a guess about a model somebody else trained.
    """
    from reranker.src.model import HeadInfo, _single_token_id

    head_json = os.path.join(checkpoint, "reranker_head.json")
    if os.path.isfile(head_json):
        with open(head_json) as f:
            return HeadInfo(**json.load(f))
    if cfg.model.head_type == "yes_no_lm":
        return HeadInfo(cfg.model.head_type,
                        yes_id=_single_token_id(tokenizer, "yes"),
                        no_id=_single_token_id(tokenizer, "no"))
    return HeadInfo(cfg.model.head_type)


def load_scorer(cfg, checkpoint: str | None) -> tuple[Callable, object]:
    """``(scorer, tokenizer)`` for a saved checkpoint -- the only GPU in this module.

    Not `reranker.src.eval::_load_model`, which is the same twenty lines: that module imports
    mlflow at the top and the cluster venv job E runs in does not have it. `HeadInfo` is
    shared, and it is the part that matters -- how a backbone's outputs become one scalar is
    where a checkpoint-agnostic scorer would otherwise guess.
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    from reranker.src.model import load_tokenizer

    if not checkpoint:
        raise ValueError("rank_eval needs --checkpoint: it scores an existing model, and "
                         "there is no default worth guessing")
    conf = cfg.prm_rollout
    # The reranker's tokenizer. `conf.gen_model` is the generation model's and belongs to
    # rollout.py's token budget; reaching for it here would encode with the wrong vocabulary.
    if os.path.isfile(os.path.join(checkpoint, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = load_tokenizer(conf.base_model)
    head = head_of(cfg, checkpoint, tokenizer)

    dtype = torch.bfloat16 if cfg.train.bf16 else (torch.float16 if cfg.train.fp16 else torch.float32)
    common = dict(dtype=dtype, attn_implementation=cfg.model.attn_implementation)
    if head.head_type == "seq_cls":
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1, **common)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **common)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @torch.no_grad()
    def forward(chunk: list[list[int]]) -> list[float]:
        # Right-padded, which both heads assume -- but they locate the last real token by
        # different means, so both have to hold. `yes_no_lm` reads it off the attention mask
        # (`extract_logits`, model.py). `seq_cls` never sees the mask: HF has already pooled,
        # taking the highest index where `input_ids != config.pad_token_id`. The two agree
        # only while padding is on the right *and* `pad_id` does not occur inside a prefix --
        # true here, since it is the Qwen end-of-text token and these sequences are prompt
        # text and Triton code.
        input_ids, attention_mask = pad_sequences(chunk, pad_id)
        device = next(model.parameters()).device
        outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        return head.extract_logits(outputs, attention_mask.to(device)).float().tolist()

    size = cfg.train.per_device_eval_batch_size
    return (lambda ids: batched(ids, forward, size)), tokenizer


def main(argv=None) -> None:
    # `load_config` passes unknown args through as `key=value` overrides, so --checkpoint has
    # to come off first or it is read as one and raises.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", required=True)
    pre.add_argument("--checkpoint", required=True, help="the checkpoint to score; trains nothing")
    args, rest = pre.parse_known_args(argv)
    out = rank_eval(
        load_config(["--config", args.config] + rest), checkpoint=args.checkpoint
    )

    print(f"{out['lists']} val lists, {out['items']} items, {out['pairs']} ranking pairs "
          f"({out['two_item_lists']} lists are pairs)")
    print(f"  overall   pairwise {_pct(out['pairwise_acc'])}  ndcg {_pct(out['ndcg'])}")
    for band in out["buckets"]:
        print(f"  {band['lo']:.2f}-{band['hi']:.2f}  pairwise {_pct(band['pairwise_acc'])}  "
              f"ndcg {_pct(band['ndcg'])}  ({band['n_lists']} lists, {band['n_pairs']} pairs)")
    if out["lists_outside_window"]:
        print(f"  WARNING {out['lists_outside_window']} lists sit outside "
              f"[{out['buckets'][0]['lo']}, {out['buckets'][-1]['hi']}] and were clamped into "
              "the edge bands -- the depth window has moved since job A")


def _pct(value: float | None) -> str:
    """``--`` rather than ``0.000`` for a band nothing was measured in."""
    return "  --  " if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
