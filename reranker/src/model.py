"""Reranker backbone construction.

The reranker produces one scalar logit per (reference, kernel) pair. Two head
types are supported:

  seq_cls   AutoModelForSequenceClassification(num_labels=1) — model-agnostic.
  yes_no_lm AutoModelForCausalLM — score = logit("yes") - logit("no") at the last
            real token. The native Qwen3-Reranker formulation.

We deliberately return the *raw* HuggingFace model (a `PreTrainedModel`) rather
than an `nn.Module` wrapper, so `Trainer` checkpoint save/load works natively.
The head-specific reduction from model outputs to a scalar logit lives in
`HeadInfo.extract_logits`, which the trainer calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

HEAD_TYPES = ("seq_cls", "yes_no_lm")


def load_tokenizer(base_model: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@dataclass
class HeadInfo:
    """How to reduce a backbone's outputs to one scalar logit per example."""

    head_type: str
    yes_id: int = -1
    no_id: int = -1

    def extract_logits(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.head_type == "seq_cls":
            return outputs.logits.squeeze(-1)  # (batch, 1) -> (batch,)
        # yes_no_lm: pick the last real token, contrast yes vs no vocab logits.
        last_idx = attention_mask.long().sum(dim=1) - 1
        batch_idx = torch.arange(attention_mask.size(0), device=attention_mask.device)
        last_logits = outputs.logits[batch_idx, last_idx]  # (batch, vocab)
        return last_logits[:, self.yes_id] - last_logits[:, self.no_id]


def build_backbone(cfg, tokenizer):
    """Return (backbone PreTrainedModel, HeadInfo) for the configured head type."""
    model_cfg = cfg.model
    if model_cfg.head_type not in HEAD_TYPES:
        raise ValueError(f"head_type must be one of {HEAD_TYPES}, got {model_cfg.head_type}")

    dtype = torch.bfloat16 if cfg.train.bf16 else (
        torch.float16 if cfg.train.fp16 else torch.float32
    )
    common = dict(dtype=dtype, attn_implementation=model_cfg.attn_implementation)

    if model_cfg.head_type == "seq_cls":
        backbone = AutoModelForSequenceClassification.from_pretrained(
            model_cfg.base_model, num_labels=1, **common
        )
        if backbone.config.pad_token_id is None:
            backbone.config.pad_token_id = tokenizer.pad_token_id
        return backbone, HeadInfo("seq_cls")

    backbone = AutoModelForCausalLM.from_pretrained(model_cfg.base_model, **common)
    if backbone.config.pad_token_id is None:
        backbone.config.pad_token_id = tokenizer.pad_token_id
    yes_id = _single_token_id(tokenizer, "yes")
    no_id = _single_token_id(tokenizer, "no")
    return backbone, HeadInfo("yes_no_lm", yes_id=yes_id, no_id=no_id)


def _single_token_id(tokenizer, word: str) -> int:
    """Token id for a word, trying a few common encodings; fail loudly otherwise."""
    for variant in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(
        f"Could not map '{word}' to a single token for this tokenizer; "
        f"yes_no_lm head needs single-token 'yes'/'no'. Use head_type=seq_cls instead."
    )
