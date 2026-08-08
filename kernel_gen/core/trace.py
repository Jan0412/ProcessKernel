"""Per-token model internals: what was sampled, and how sure the model was.

The generation loop throws away everything except the completion text. That is fine
for producing kernels and useless for training a process reward model, which needs to
know *where* in a generation the model started to go wrong. This module is the durable
record: token ids, the top-K alternatives the model considered at each step, and a
handful of precomputed confidence scalars.

Two rules shape it.

**Pure numpy, no vLLM.** Nothing here imports a backend, so the whole module -- and
every test over it -- runs on a login node. The backend hands over plain Python lists;
the packing, the arithmetic and the file format all live here.

**Raw top-K is stored alongside the scalars.** The scalars are the measures we know we
want today (Shannon entropy, DeepConf token confidence, self-certainty, margin). A
measure nobody has thought of yet is one line of offline numpy over ``topk_lp`` -- and
regenerating a trace costs GPU-hours, so the raw material is never discarded to save
kilobytes.

A note on truncation, because every scalar here inherits it: the model's distribution
is over ~150k tokens and we keep 20. ``tail_mass`` (the probability NOT in the top-K)
is stored per token so that inheritance is measurable rather than assumed -- an
``entropy`` of 0.4 means something very different when the tail holds 1e-6 than when it
holds 0.3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

#: what ``topk_ids`` holds where the model returned fewer than K alternatives
PAD_ID = -1

#: Segment codes for :attr:`TokenTrace.seg`. The two-pass sampler generates the plan
#: and the code in separate calls at different temperatures, and a PRM that cannot tell
#: them apart would compare a prose token's confidence against a code token's.
SEG_PLAN = 0
SEG_CODE = 1


@dataclass
class TokenTrace:
    """One completion's token-level record. All arrays are indexed by token position.

    ``token_ids`` is what was *sampled*, which is not always ``topk_ids[:, 0]`` -- at
    temperature 1.0 the model regularly emits its second or fifth choice, and which one
    it took is exactly the signal a PRM is being asked to learn from. ``sampled_lp``
    carries that token's own logprob so surprisal never has to be looked up by id.
    """

    token_ids: np.ndarray  # int32[T]
    topk_ids: np.ndarray  # int32[T, K], PAD_ID where the model returned fewer
    topk_lp: np.ndarray  # float16[T, K], -inf on the padding
    sampled_lp: np.ndarray  # float32[T]
    sampled_rank: np.ndarray  # int16[T], 1-based; 0 = unknown
    seg: np.ndarray  # int8[T], SEG_PLAN / SEG_CODE
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def k(self) -> int:
        return int(self.topk_lp.shape[1])


def dedupe_rows(
    topk_ids: np.ndarray, topk_lp: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Drop each row's repeated ids, keep descending order, return width ``k``.

    vLLM's sampler emits ``(num tokens) x (num_logprobs + 1)`` -- ``torch.cat`` of the
    sampled token's own id onto the top-K indices -- and the engine appends that row
    into :class:`FlatLogprobs` verbatim. Its own source says a duplicate is harmless
    because "inserting duplicated data into a dictionary twice is the same as doing it
    once", which is true of the legacy ``dict`` container and false of the flat one
    :func:`kernel_gen.core.backend._topk` actually reads. So the sampled token arrives
    twice whenever it ranked inside the top-K, and truncating to K without deduping
    keeps a repeat and discards the real rank-K alternative (KGEN-25).

    First occurrence wins: it sits at the token's true rank, which is what
    ``sampled_rank`` already records, so deduping never moves it. Exact float ties can
    sort the two copies apart, so the check is on ids, not on adjacency, and the sort
    is stable to leave the surviving order untouched.
    """
    ids = np.asarray(topk_ids)
    same = (ids[:, :, None] == ids[:, None, :]) & (ids[:, :, None] != PAD_ID)
    keep = (ids != PAD_ID) & ~np.triu(same, 1).any(axis=1)
    order = np.argsort(~keep, axis=1, kind="stable")
    slot = np.arange(ids.shape[1])[None, :]
    n_keep = keep.sum(axis=1)[:, None]

    out_ids = np.where(slot < n_keep, np.take_along_axis(ids, order, 1), PAD_ID)
    out_lp = np.where(
        slot < n_keep,
        np.take_along_axis(np.asarray(topk_lp, dtype=np.float32), order, 1),
        -np.inf,
    )
    # Pad rather than narrow: a caller asking for K columns gets K, even from a backend
    # that returned fewer alternatives than that -- including one that returned none.
    if out_ids.shape[1] < k:
        short = k - out_ids.shape[1]
        out_ids = np.hstack([out_ids, np.full((len(ids), short), PAD_ID, dtype=out_ids.dtype)])
        out_lp = np.hstack([out_lp, np.full((len(ids), short), -np.inf, dtype=out_lp.dtype)])
    return out_ids[:, :k].astype(np.int32), out_lp[:, :k].astype(np.float16)


def pack(
    token_ids: list[int],
    topk: list[list[tuple[int, float]]] | None,
    k: int,
    seg: int = SEG_PLAN,
    meta: dict | None = None,
) -> TokenTrace:
    """Backend output -> a :class:`TokenTrace`, one segment, no seam handling.

    ``topk[t]`` is the alternatives at step ``t`` as ``(token_id, logprob)``. Three
    properties of vLLM's actual output are handled here rather than trusted:

    **Order.** vLLM prepends the *sampled* token to the top-K, so when sampling took a
    runner-up -- routine at ``--think-temperature 1.0`` -- the first entry is NOT the
    argmax. Rows are therefore sorted by logprob here, descending. Trusting the incoming
    order would put the sampled token in the ``top1_lp`` column for exactly the tokens
    where the distinction matters.

    **Duplicates.** That same prepend repeats the sampled token whenever it ranked
    inside the top-K, which is almost always. :func:`dedupe_rows` drops the repeat, so a
    row of K columns holds K *distinct* alternatives.

    **Width.** The row therefore arrives K+1 wide. It is deduped and truncated to K
    *after* the sampled token's logprob and rank have been read off the full sorted row,
    so a token sampled outside the top-K still records its true rank and logprob and
    nothing downstream has to reason about ragged rows.

    ``topk=None`` (a backend with no internals to give) still produces a valid trace --
    the arrays are PAD rather than absent, so callers never branch.

    ``meta["trace_k"]`` records the K actually stored. Traces outlive the code that
    wrote them, K-dependent scalars (``entropy``, ``deepconf_c``, ``tail_mass``) are not
    comparable across different K, and a repaired corpus sits at a different K from a
    natively-generated one -- so the file says which it is rather than leaving a reader
    to infer it from an array shape that cannot distinguish the two.
    """
    n = len(token_ids)
    ids = np.asarray(token_ids, dtype=np.int32)

    rows = list(topk or [])[:n]  # a backend that returned more logprob rows than tokens
    width = max((len(row) for row in rows), default=0)
    staged_ids = np.full((n, width), PAD_ID, dtype=np.int32)
    staged_lp = np.full((n, width), -np.inf, dtype=np.float32)
    sampled_lp = np.zeros(n, dtype=np.float32)
    sampled_rank = np.zeros(n, dtype=np.int16)

    for t, row in enumerate(rows):
        ordered = sorted(row, key=lambda entry: entry[1], reverse=True)
        for rank, (token_id, logprob) in enumerate(ordered, start=1):
            if token_id == ids[t]:
                sampled_lp[t] = logprob
                sampled_rank[t] = rank
                break
        staged_ids[t, : len(ordered)] = [i for i, _ in ordered]
        staged_lp[t, : len(ordered)] = [lp for _, lp in ordered]

    topk_ids, topk_lp = dedupe_rows(staged_ids, staged_lp, k)

    return TokenTrace(
        token_ids=ids,
        topk_ids=topk_ids,
        topk_lp=topk_lp,
        sampled_lp=sampled_lp,
        sampled_rank=sampled_rank,
        seg=np.full(n, seg, dtype=np.int8),
        meta={**(meta or {}), "trace_k": int(k)},
    )


def concat_passes(plan: TokenTrace, code: TokenTrace, meta: dict | None = None) -> TokenTrace:
    """Stitch the two-pass sampler's halves into one trace, and record where the seam is.

    The completion string the loop stores is ``PLAN_PREFIX + plan + CODE_FENCE + code``,
    but ``PLAN_PREFIX`` is a pass-1 *prefill* and ``CODE_FENCE`` is part of pass 2's
    *prompt*. Neither was sampled, so neither has a confidence value and neither appears
    in these arrays. Concatenating without saying so leaves every downstream character
    offset silently wrong by the length of two strings, which is the kind of bug that
    produces a plausible-looking PRM trained on misaligned labels.

    So the token counts and the seam's character offsets go in ``meta``, and ``seg``
    flips from :data:`SEG_PLAN` to :data:`SEG_CODE` at exactly ``n_plan_tokens``.
    """
    if plan.k != code.k:
        raise ValueError(f"passes disagree on K: {plan.k} vs {code.k}")

    merged = dict(plan.meta)
    merged.update(code.meta)
    merged.update(meta or {})
    merged["n_plan_tokens"] = len(plan)
    merged["n_code_tokens"] = len(code)

    return TokenTrace(
        token_ids=np.concatenate([plan.token_ids, code.token_ids]),
        topk_ids=np.concatenate([plan.topk_ids, code.topk_ids]),
        topk_lp=np.concatenate([plan.topk_lp, code.topk_lp]),
        sampled_lp=np.concatenate([plan.sampled_lp, code.sampled_lp]),
        sampled_rank=np.concatenate([plan.sampled_rank, code.sampled_rank]),
        seg=np.concatenate([
            np.full(len(plan), SEG_PLAN, dtype=np.int8),
            np.full(len(code), SEG_CODE, dtype=np.int8),
        ]),
        meta=merged,
    )


def derive_scalars(
    topk_lp: np.ndarray, sampled_lp: np.ndarray | None = None, vocab_size: int | None = None
) -> dict[str, np.ndarray]:
    """Per-token confidence measures from the top-K logprobs. All float32[T].

    ``entropy``
        Shannon entropy of the top-K *renormalized* to sum to 1. Bounded by log K, so
        it is comparable across tokens but is a truncated estimate -- read it next to
        ``tail_mass``. This is EDU-PRM's anchor signal.
    ``top1_lp``, ``margin``
        The best alternative's logprob, and ``p1 - p2`` -- the cheapest possible
        "was this a close call" feature.
    ``surprisal``
        ``-log P(sampled)``. Mean over a span is that span's log-perplexity. Distinct
        from ``-top1_lp`` whenever sampling did not take the argmax, which at
        ``--think-temperature 1.0`` is most interesting tokens.
    ``deepconf_c``
        DeepConf token confidence, ``-(1/K) sum_j log P(j)`` over the top-K. Higher is
        *more* confident: when the model is sure, the runners-up collapse to ~1e-8 and
        their logprobs dominate the mean. Note it is only monotone against
        "probability mass concentrated in the top-K" -- a distribution smeared over the
        whole vocabulary also scores high, which is precisely the case ``entropy`` and
        ``tail_mass`` catch. Keep all three.
    ``tail_mass``
        ``1 - sum_j P(j)``: how much of the distribution the top-K did not see.
    ``self_cert``
        Self-certainty, ``KL(P || uniform) = log V - H(P)``, the one measure of the set
        that is length-invariant. The unseen tail is charged at maximum entropy, so
        this is a *lower bound* on the true value -- conservative in the useful
        direction, since it can only understate confidence. Needs ``vocab_size``;
        omitted from the result when that is unknown rather than guessed.
    """
    lp = np.asarray(topk_lp, dtype=np.float64)
    if lp.ndim != 2:
        raise ValueError(f"topk_lp must be 2-D [T, K], got shape {lp.shape}")
    n, k = lp.shape
    valid = np.isfinite(lp)
    n_valid = valid.sum(axis=1)

    p = np.where(valid, np.exp(np.where(valid, lp, 0.0)), 0.0)
    mass = p.sum(axis=1)
    tail_mass = np.clip(1.0 - mass, 0.0, 1.0)

    # Renormalized top-K entropy. `mass` is 0 only for a token with no logprobs at all
    # (a backend that gave none); guard so an untraced run yields zeros, not NaNs.
    safe_mass = np.where(mass > 0, mass, 1.0)
    q = p / safe_mass[:, None]
    entropy = -(q * np.log(np.where(q > 0, q, 1.0))).sum(axis=1)

    order_lp = np.where(valid, lp, -np.inf)
    top1_lp = order_lp[:, 0] if k else np.full(n, -np.inf)
    p1 = p[:, 0] if k else np.zeros(n)
    p2 = p[:, 1] if k > 1 else np.zeros(n)

    # Mean over the entries that exist, not over K: a short row must not be dragged
    # toward zero by padding it never had.
    deepconf_c = np.where(
        n_valid > 0, -np.where(valid, lp, 0.0).sum(axis=1) / np.maximum(n_valid, 1), 0.0
    )

    out = {
        "entropy": entropy,
        "top1_lp": top1_lp,
        "margin": p1 - p2,
        "deepconf_c": deepconf_c,
        "tail_mass": tail_mass,
    }
    if sampled_lp is not None:
        out["surprisal"] = -np.asarray(sampled_lp, dtype=np.float64)
    if vocab_size:
        # Full-distribution entropy, upper-bounded: the observed top-K terms, plus the
        # tail at its maximum entropy (mass spread evenly over the V-K unseen tokens).
        h_seen = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1)
        n_unseen = max(vocab_size - k, 1)
        h_tail = np.where(
            tail_mass > 0, tail_mass * (np.log(n_unseen) - np.log(np.maximum(tail_mass, 1e-300))), 0.0
        )
        out["self_cert"] = np.log(vocab_size) - (h_seen + h_tail)

    return {name: value.astype(np.float32) for name, value in out.items()}


def summarize(scalars: dict[str, np.ndarray], window: int = 512) -> dict[str, float]:
    """DeepConf's trace-level statistics, as flat scalars for the sidecar journal.

    DeepConf's finding is that a trace's *worst stretch* predicts its correctness far
    better than its average -- one confidently-wrong span is not diluted by a thousand
    boilerplate tokens. So the sliding-window minimum (``c_least``) and the bottom
    decile are the headline numbers; the means are kept because they cost nothing.

    ``window`` defaults well below DeepConf's 2048, which was tuned on math traces: a
    plan here is 300-800 tokens, and a 2048-wide window would average it away entirely.

    Taken over the finite entries only: ``top1_lp`` is ``-inf`` for a token whose row held
    no alternatives, and one such token would drag the whole record's mean to ``-inf``. A
    name with nothing finite is dropped rather than faked to 0.0 (which reads as p=1).
    """
    out: dict[str, float] = {}
    for name, values in scalars.items():
        finite = values[np.isfinite(values)]
        if finite.size:
            out[f"mean_{name}"] = float(finite.mean())

    conf = scalars.get("deepconf_c")
    if conf is None:
        return out
    conf = conf[np.isfinite(conf)]  # also collapses an already-empty array
    if conf.size == 0:
        return out

    groups = _sliding_mean(conf, window)
    out["c_least"] = float(groups.min())
    out["c_bottom10"] = float(np.sort(groups)[: max(1, int(0.1 * groups.size))].mean())
    out["c_tail"] = float(conf[-window:].mean())
    out["n_tokens"] = int(conf.size)
    return out


def rank1_calibration(topk_lp: np.ndarray, sampled_rank: np.ndarray) -> tuple[float, float]:
    """``(recorded mean p1, observed P(sampled was rank 1))`` -- is the record tempered?

    This is the one check that can tell ``raw_logprobs`` from ``processed_logprobs``
    after the fact, using nothing but data already on disk, and it matters because the
    two are indistinguishable by eye: both produce plausible confidence numbers, and
    only one of them is a property of the model rather than of a CLI flag.

    The argument is simple. Sampling at temperature ``T`` draws from the distribution
    sharpened by ``1/T``. If the *recorded* distribution is raw (pre-temperature), then
    at ``T = 1`` the sampler drew from exactly what was recorded and the two numbers
    must agree, while at ``T < 1`` it drew from something sharper and the observed
    rank-1 rate must come out *higher* than the record predicts. If instead the record
    were processed, the temperature would already be baked into it and both halves would
    agree regardless of ``T``.

    So: agreement on a ``T = 1`` span and a ratio above 1 on a ``T < 1`` span is proof
    the record is raw. Measured on the first traced run (Qwen3.6-27B): 1.008 on the plan
    at ``T = 1.0``, 1.032 on the code at ``T = 0.6``.
    """
    lp = np.asarray(topk_lp, dtype=np.float64)
    if lp.size == 0 or lp.shape[1] == 0:
        return float("nan"), float("nan")
    recorded = float(np.exp(lp[:, 0]).mean())
    observed = float((np.asarray(sampled_rank) == 1).mean())
    return recorded, observed


def _sliding_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean over every window of ``window`` tokens, stride 1. Short input -> one group."""
    w = min(window, values.size)
    if w <= 1:
        return values.astype(np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    return (cumulative[w:] - cumulative[:-w]) / w


def write_trace(path: str, trace: TokenTrace) -> None:
    """One ``.npz`` per completion, uncompressed on purpose.

    Measured on a realistic trace (5,300 tokens, K=20): ``savez_compressed`` produces
    506 KiB in 161 ms, ``savez`` 680 KiB in 6.2 ms. Over the 330,000 attempts a full
    KernelBook run implies that is 171 GB at 14.7 CPU-hours against 230 GB at 0.6 --
    and the writing happens in the between-rounds checkpoint, with the GPU idle. Saving
    59 GB out of 332 TB free is not worth 14 CPU-hours of a job the GPU is waiting on.
    The bulk is ``topk_ids``, which is near-random and barely compresses anyway.

    ``meta`` goes in as a JSON string rather than as object-dtype arrays, so the file
    loads without ``allow_pickle`` -- these files outlive the code that wrote them and
    a pickle in the load path would make reading them a trust decision.
    """
    np.savez(
        path,
        token_ids=trace.token_ids,
        topk_ids=trace.topk_ids,
        topk_lp=trace.topk_lp,
        sampled_lp=trace.sampled_lp,
        sampled_rank=trace.sampled_rank,
        seg=trace.seg,
        meta=np.array(json.dumps(trace.meta)),
    )


def read_trace(path: str) -> TokenTrace:
    with np.load(path) as data:
        return TokenTrace(
            token_ids=data["token_ids"],
            topk_ids=data["topk_ids"],
            topk_lp=data["topk_lp"],
            sampled_lp=data["sampled_lp"],
            sampled_rank=data["sampled_rank"],
            seg=data["seg"],
            meta=json.loads(str(data["meta"])),
        )
