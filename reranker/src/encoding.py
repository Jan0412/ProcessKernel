"""Shared cross-encoder sequence encoding for the reranker.

Both the pointwise dataset (`RerankerDataset`) and the pairwise dataset
(`PairwiseDataset`) must turn a (reference architecture, candidate kernel) pair
into input ids *identically*, otherwise scores from a pairwise-trained model are
not comparable across the two code paths (the pairwise model is validated via
the pointwise dataset). This module is the single source of truth for that
format.

Layout of one encoded example:

    INSTRUCTION + ref_ids[:reserve] + SEPARATOR + kernel_ids[:budget] + EOS

Truncation keeps the full reference where possible (up to `reserve_ref_tokens`)
and truncates the candidate kernel's tail, since kernels can be long and the
head carries the imports / structure.
"""

from __future__ import annotations

INSTRUCTION = (
    "You are judging whether a generated GPU kernel is a correct and faster "
    "drop-in replacement for the given PyTorch reference architecture.\n"
    "Reference architecture:\n"
)
SEPARATOR = "\n\nCandidate kernel:\n"


class SequenceEncoder:
    """Encode a (reference, kernel) pair into a single cross-encoder token sequence."""

    def __init__(self, tokenizer, max_length: int, reserve_ref_tokens: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reserve_ref_tokens = reserve_ref_tokens
        # Precompute static instruction / separator token ids (no special tokens).
        self._instr_ids = tokenizer.encode(INSTRUCTION, add_special_tokens=False)
        self._sep_ids = tokenizer.encode(SEPARATOR, add_special_tokens=False)

    def encode(self, ref_src: str, kernel_src: str) -> list[int]:
        return self.encode_with_meta(ref_src, kernel_src)[0]

    def encode_with_meta(self, ref_src: str, kernel_src: str) -> tuple[list[int], dict]:
        """Like :meth:`encode`, but also return truncation diagnostics.

        The second element is ``{"ref_tokens", "kernel_tokens", "ref_total",
        "kernel_total", "truncated"}`` where ``*_tokens`` are the counts actually
        kept in the sequence, ``*_total`` the untruncated counts, and
        ``truncated`` True iff either side was cut. Scoring uses this to record
        how much of each side the model actually saw. The token layout is
        identical to :meth:`encode` — this is the single source of truth.
        """
        tok = self.tokenizer
        ref_ids = tok.encode(ref_src, add_special_tokens=False)
        kernel_ids = tok.encode(kernel_src, add_special_tokens=False)
        ref_total, kernel_total = len(ref_ids), len(kernel_ids)

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
        meta = {
            "ref_tokens": len(ref_ids),
            "kernel_tokens": len(kernel_ids),
            "ref_total": ref_total,
            "kernel_total": kernel_total,
            "truncated": (ref_total > len(ref_ids)) or (kernel_total > len(kernel_ids)),
        }
        return content, meta
