"""A finished campaign on disk — v1's parts plus job A's and job D's outputs.

Not a test module: pytest collects ``test_*.py`` only. Written once here rather than copied
into `test_rank_eval.py` and `test_train.py`, which would let job E's fixture and the
trainer's drift apart — and the point of the trainer's eval is that it computes *the same
statistic* over *the same lists* as job E (ARCHITECTURE S10, PLAN_TRAINER §7).
"""

from __future__ import annotations

import dataclasses
import json
import os

from reranker.src.config import PRMRolloutConfig, RerankerConfig
from reranker.src.prm import build
from reranker.src.prm.rollout import lists, prefixes

TAG = "ar"
RUN, SHARD = "a_run", "shard_00"
SYS, USER = "SYSTEM PROMPT", "USER PROMPT\n"
RAW = "0123456789abcdefghij"


def cfg(**over) -> PRMRolloutConfig:
    base = PRMRolloutConfig(run_tags={"a_run": TAG}, baseline_timing_json=__file__)
    for k, v in over.items():
        setattr(base, k, v)
    base.validate()
    return base


class CharTokenizer:
    """One id per character, so a test can read back exactly what the scorer was shown."""

    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


def pre(pid: str, *, sid: int, cut_char: int, split="val", **over) -> prefixes.Prefix:
    fields = dict(
        prefix_id=pid, source="cut", run_name=RUN, run_tag=TAG, shard=SHARD, round=0,
        level=2, problem_id=37, sample_id=sid,
        stem=f"level_2_problem_37_sample_{sid}_kernel",
        cut_char=cut_char, cut_index=5, cut_kind="code", n_cuts_total=20, rel_depth=0.5,
        list_key=f"{TAG}:2:37:0:5", split=split, selection="random", selection_score=None,
        K=4, min_rollouts=2,
    )
    fields.update(over)
    return prefixes.Prefix(**fields)


def part_row(sid: int, **over) -> dict:
    r = {
        "run_name": RUN, "shard": SHARD, "round": 0, "level": 2, "problem_id": 37,
        "sample_id": sid, "stem": f"level_2_problem_37_sample_{sid}_kernel",
        build.ROW_SHA1: "a" * 40, "prompt": USER, "raw": RAW,
        "cuts": [4, 10], "cut_kinds": ["prose", "code"], "cut_index": [0, 1], "correct": True,
    }
    r.update(over)
    return r


def campaign(tmp_path, ps, list_rows, *, part_rows=None, train_rows=(), **over) -> RerankerConfig:
    """v1's parts plus job A's and job D's outputs, laid out as the campaign writes them."""
    parts = tmp_path / build.PARTS
    parts.mkdir(exist_ok=True)
    (parts / f"{RUN}__{SHARD}__round0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in (part_rows or [part_row(p.sample_id) for p in ps]))
    )
    (tmp_path / build.PROMPTS).write_text(json.dumps({"a" * 40: SYS}))

    out = tmp_path / "campaign"
    out.mkdir(exist_ok=True)
    (out / prefixes.PREFIXES).write_text(
        "".join(json.dumps(dataclasses.asdict(p)) + "\n" for p in ps)
    )
    for split, rows in (("val", list_rows), ("train", train_rows)):
        (out / lists.LISTS.format(split=split)).write_text(
            "".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in rows)
        )
    knobs = {
        "parts_glob": os.path.join(str(tmp_path), build.PARTS, "*.jsonl"),
        "out_dir": str(out),
        "depth_buckets": 1,
    }
    return RerankerConfig(prm_rollout=cfg(**(knobs | over)))


def two_prefix_campaign(tmp_path, rel_depth_mean=0.5, **over):
    """One val list of two prefixes cut at 4 and 12 characters."""
    ps = [pre("p0", sid=0, cut_char=4), pre("p1", sid=1, cut_char=12)]
    lst = lists.ListRow(
        list_key=f"{TAG}:2:37:0:5", run_tag=TAG, level=2, problem_id=37, round=0,
        cut_index=5, rel_depth_mean=rel_depth_mean, split="val", source="cut",
        items=[lists.Item("p0", rel=2.0, n_rollouts=4, se=0.2),
               lists.Item("p1", rel=0.0, n_rollouts=4, se=0.2)],
    )
    return ps, campaign(tmp_path, ps, [lst], **over)


class StubScorer:
    """Scores by the length of what it was shown, and keeps every sequence for inspection."""

    def __init__(self):
        self.seen: list[list[int]] = []

    def __call__(self, ids: list[list[int]]) -> list[float]:
        self.seen.extend(ids)
        return [float(len(x)) for x in ids]


def tiny_backbone(path) -> str:
    """A ~19k-parameter Qwen3 + a char tokenizer, saved where `from_pretrained` can load them.

    Hermetic on purpose: `build_backbone` and `load_tokenizer` both go through
    `from_pretrained`, which takes a local directory as happily as a hub id. Pointing them at
    one lets `main()` be tested end to end with no network, no hub cache and no 4B download.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        AutoModelForSequenceClassification,
        PreTrainedTokenizerFast,
        Qwen3Config,
    )

    path = str(path)
    conf = Qwen3Config(
        vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
        num_labels=1,
    )
    AutoModelForSequenceClassification.from_config(conf).save_pretrained(path)

    # Contiguous ids: a vocab with holes makes `save_pretrained` warn that it may be corrupt.
    vocab = {"[UNK]": 0, "[PAD]": 1} | {chr(i): i - 30 for i in range(32, 127)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[PAD]"
    ).save_pretrained(path)
    return path
