"""A prefix -> the ids a PRM scores it by: §5's one built contract piece (PLAN_v2 §5).

Deliberately **not** `reranker/src/encoding.py::SequenceEncoder`, which lays out
``INSTRUCTION + ref + SEPARATOR + kernel``. That shape assumes a finished candidate kernel to
put after the separator, and a PRM is handed a half-written generation instead -- so v2 scores
the *stored* prompt plus the prefix, verbatim, exactly what v1 froze. The pointwise and
pairwise paths depend on the other layout; folding the two together would change what every
existing checkpoint was trained on.

It lives here, in its own module, so `rank_eval.py` and whatever trainer comes later encode
through one function rather than two copies that drift.
"""

from __future__ import annotations

from reranker.src.prm.rollout.rollout import Source, prefix_text


def scored_text_of(prompt: str, generated: str) -> str:
    """The PRM's layout from its two pieces directly, for a live candidate with no stored row."""
    return prompt + generated


def scored_text(prefix, src: Source) -> str:
    """What the PRM reads: the stored prompt, then the generation so far.

    The prompt is v1's raw text, not `render_chat`'s rendering of it -- v1 §2 decision 2, and
    at inference there is no chat header around the fragment being judged. The generated half
    comes from job B's `prefix_text`, so the slice (and its range check) has one definition.
    """
    return scored_text_of(src.prompt, prefix_text(prefix, src))


class PrefixEncoder:
    """`scored_text` -> input ids, truncated from the head.

    Truncation is the whole reason this is a class and not one call to
    ``tok(text, truncation=True)``: the tokenizer's default drops the *tail*, which is the cut
    point -- the single position whose value is being measured. Dropping from the head instead
    spends the budget on the prompt first and keeps the prefix whole; a prefix that alone
    overruns `max_length` loses its own head and still ends where it was cut.

    No EOS is appended, unlike `SequenceEncoder`. A prefix is by construction unfinished, and
    a terminator would both claim otherwise and displace the cut point from the pooled last
    position.
    """

    def __init__(self, tokenizer, max_length: int):
        # ids[-0:] is the whole list, so a zero budget would encode everything untruncated.
        if max_length < 1:
            raise ValueError(f"PrefixEncoder needs max_length >= 1, got {max_length}")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode_text(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids[-self.max_length :]

    def encode(self, prefix, src: Source) -> list[int]:
        return self.encode_text(scored_text(prefix, src))
