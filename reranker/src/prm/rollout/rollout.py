"""Prefixes -> K measured continuations each: job B's generation half (PLAN_v2 §6).

v1 asks what a completion ended up being worth. This asks what a *prefix* is worth, by
continuing it K times from the exact context the sampler held and evaluating what comes
back. Nothing here writes a file -- stage.py owns the rollout rows, because `staged_as` is
its to assign and the map job D reads is a projection of them.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from kernel_gen.core.sampling import CODE_FENCE
from kernel_gen.core.text import extract_code_block
from reranker.src.config import _resolve
from reranker.src.prm import build, corpus
from reranker.src.prm.chunks import CODE, PROSE

# §6 wants a unit's rollouts in one call, and it wants max_tokens sized per prefix. The
# Backend takes one max_tokens per call, so the two cannot both hold exactly; bucketing the
# budget this coarsely keeps the call count in single digits (measured 7 buckets over a real
# shard, against 4,700 prefixes) while a call still takes its bucket's *minimum*, so no
# prompt is ever given more than its own prefix left it. A member loses at most this many
# tokens off a budget measured at 9,855-16,249 on that shard.
BUDGET_QUANTUM = 1024


@dataclass(frozen=True)
class Source:
    """The three texts v1 froze for one completion; the prefix is a slice of ``raw``."""

    system_prompt: str
    prompt: str
    raw: str


def source_key(x) -> tuple:
    """A completion's identity, shared by a v1 row and every prefix cut from it.

    ``stem`` alone is not it: the same sample is regenerated every round and by every run,
    so three of the four components are needed before a stem names one completion. Strict on
    both branches -- a tolerant ``.get`` would turn a renamed v1 field into ``None``, collapse
    a part's rows onto keys differing only by stem, and resolve a prefix to another round's
    text with every guard below still passing.
    """
    get = x.__getitem__ if isinstance(x, dict) else lambda k: getattr(x, k)
    return (get("run_name"), get("shard"), get("round"), get("stem"))


def load_prompts(parts_glob: str) -> dict[str, str]:
    """v1's ``system_prompt_sha1`` -> text side table, which sits beside the parts dir."""
    return json.loads(_read(os.path.join(_v1_dir(_resolve(parts_glob)), build.PROMPTS)))


def load_sources(part_path: str, prompts: dict[str, str]) -> dict[tuple, Source]:
    """One part's texts, keyed by :func:`source_key` -- job B resolves them once per unit."""
    out = {}
    with open(part_path) as f:
        for line in f:
            row = json.loads(line)
            sha = row[build.ROW_SHA1]
            if sha not in prompts:
                raise ValueError(
                    f"{os.path.basename(part_path)} carries system prompt {sha}, which "
                    f"{build.PROMPTS} does not resolve -- the side table is derived from the "
                    "parts and a build killed before it was written leaves shas pointing "
                    "nowhere; rerun reranker.src.prm.build over this out_dir"
                )
            out[source_key(row)] = Source(prompts[sha], row["prompt"], row["raw"])
    return out


def reconstruct(backend, prefix, src: Source) -> str:
    """The exact string the model held at this cut: chat header, then the generated prefix.

    Byte-exact except for the ``Current date:`` line gpt-oss's template stamps, which the
    source run did not record and which is identical for every prefix in a list (§13 step 2).
    """
    return backend.render_chat(src.system_prompt, src.prompt) + prefix_text(prefix, src)


def prefix_text(prefix, src: Source) -> str:
    """The generated half of the context: ``raw[:cut_char]``, plus a beam's branch.

    Range-checked at both ends because a slice is not: past the end Python returns the whole
    completion and a negative one counts back from it, and either records a prefix other than
    the one job A enumerated. `source_key` compares (run, shard, round, stem), so a part
    rebuilt with different text keeps its key and reaches here.
    """
    if not 0 <= prefix.cut_char <= len(src.raw):
        raise ValueError(
            f"{prefix.prefix_id} has cut_char={prefix.cut_char}, outside its v1 row's "
            f"0..{len(src.raw)} -- the prefixes and the parts came from different builds; "
            "point the campaign's parts_glob at the build job A read"
        )
    return src.raw[: prefix.cut_char] + (prefix.beam_text or "")


def n_prefix_tokens(prefix, src: Source, count: Callable[[str], int]) -> int:
    """How much of the generation budget this prefix has already spent.

    ``count`` is the **generation** model's tokenizer, never the reranker's that bounds
    `max_length` in §7 -- they are different models and mixing them mis-sizes the budget
    with nothing to notice it.
    """
    return count(prefix_text(prefix, src))


def gen_counter(cfg) -> Callable[[str], int]:
    """The **generation** model's tokenizer, which is what sizes a rollout's budget.

    Named so the call site says which model it is: `cfg.base_model` is the reranker's and
    bounds `max_length` in §7, and reaching for the wrong one here mis-sizes every budget
    with nothing downstream to notice.
    """
    return build.token_counter(cfg.gen_model)


def budget(spent: int, cfg) -> int:
    """The continuation's ``max_tokens``: what is left of the source run's own cap.

    Uniformly conservative rather than an exact mirror. Past the seam the prefix's *plan*
    tokens are charged against pass 2, which the source run did not do (it gave each pass its
    own `max_new_tokens`), so a rollout is never handed more room than the source had and
    sometimes less. Measured: rooms 9,855-16,249 against completions of ~1,550 tokens.

    Floored at zero rather than allowed negative: `raw` is two passes stitched together, so a
    deep cut can have spent more than one pass' worth. Callers drop a prefix with no room.
    """
    return max(0, cfg.max_new_tokens - spent)


@dataclass(frozen=True)
class Rollout:
    """One row of ``rollouts.jsonl.gz`` (PLAN_v2 §5).

    ``continuation`` is the generated tail only: ``prompt + raw[:cut_char] + continuation``
    reconstructs the whole text, and storing the head would copy v1's corpus once per rollout.
    """

    rollout_id: str
    prefix_id: str
    j: int
    continuation: str
    # Extracted here and stored verbatim, never re-extracted later: KGEN-20 measured
    # re-extraction drifting on 292/10,510 (2.78%) of the source run's completions.
    code: str
    code_sha1: str          # the campaign-wide eval dedup key
    n_prefix_tokens: int
    n_gen_tokens: int
    finish_reason: dict     # {"plan": ..., "code": ...}; plan is null on a code cut
    truncation: str         # ok | truncated | unknown -- v1's truncation_state, unchanged
    staged_as: dict | None = None   # stage.py's to assign; null when deduped away


@dataclass
class _Job:
    """One rollout in flight: its context, its budget, and the passes as they come back."""

    prefix: object
    j: int
    head: str          # render_chat(...) + raw[:cut_char] + beam
    text: str          # the generated half of `head`, which is what gets extracted from
    spent: int
    room: int
    mode: str          # which pass the source run was in here -- see _mode
    plan: object = None
    code: object = None


def generate(backend, prefixes, sources: dict, cfg, count, counts=None) -> list[Rollout]:
    """``K`` continuations of every prefix, sampled in the regime each prefix came from."""
    counts = Counter() if counts is None else counts
    jobs: list[_Job] = []
    for prefix in prefixes:
        if prefix.cut_kind not in (PROSE, CODE):
            raise ValueError(
                f"{prefix.prefix_id} has cut_kind={prefix.cut_kind!r}, which selects no "
                f"continuation mode. Falling through to {CODE!r} would continue a plan at "
                "the code temperature -- a different distribution from the one the prefix "
                "was generated under"
            )
        key = source_key(prefix)
        if key not in sources:
            raise KeyError(
                f"{prefix.prefix_id} names {key}, which this unit's part does not hold. "
                "Prefixes and texts have to come from one v1 build -- point the campaign's "
                "parts_glob at the build job A read, or re-run job A over this one"
            )
        src = sources[key]
        text = prefix_text(prefix, src)
        spent = n_prefix_tokens(prefix, src, count)
        room = budget(spent, cfg)
        if room < 1:
            # Every rollout would stop at length and be dropped by job D anyway; and a
            # zero-budget member would drag its whole batch's max_tokens down with it.
            counts["prefix_no_budget"] += 1
            continue
        counts["prefixes"] += 1
        # reconstruct(), not an inline copy of it: the tests that pin the prompt exact are
        # written against that function, and a second implementation here would drift silently.
        head = reconstruct(backend, prefix, src)
        # Once per prefix, not once per rollout: the K siblings share a context, and this
        # scans it for fences.
        mode = _mode(prefix, text, counts)
        jobs.extend(_Job(prefix, j, head, text, spent, room, mode) for j in range(prefix.K))

    prose = [job for job in jobs if job.mode == PROSE]
    plans = _call(backend, prose, lambda job: job.head, cfg.think_temperature, [CODE_FENCE])
    for job, plan in zip(prose, plans):
        job.plan = plan
    for job, code in zip(jobs, _call(backend, jobs, _code_prompt, cfg.temperature, None)):
        job.code = code

    counts["rollouts"] += len(jobs)
    return [_row(job, count) for job in jobs]


def _mode(prefix, text: str, counts: Counter) -> str:
    """Which pass the source run was in at this character -- not what the text looks like.

    `cut_kind` answers a different question. `chunks.py` labels a chunk by what it *is*, and
    its prose regions are the gaps between fenced spans, so they carry the ```` ```python ````
    marker line and every stretch of prose pass 2 wrote between blocks. Both are `prose`, and
    both are past the seam.

    The seam is recoverable exactly, with no heuristic: pass 1 stops at CODE_FENCE and vLLM
    drops the stop string from the text, so a kept row's plan cannot contain one and the first
    ``` ```python ``` in `raw` *is* where pass 2 began. A prefix holding it is therefore in
    pass 2, whatever its label says. Measured over three gpt-oss shards: 637 of 11,477 prose
    cuts (5.6%) -- 263 standing at the fence itself, 374 in prose written after pass 2 closed
    a block. Continuing those at `think_temperature` would sample them from a distribution
    they never came from, and would splice a second opening fence into the completion.
    """
    if prefix.cut_kind != PROSE:
        return CODE
    if CODE_FENCE in text:
        counts["prose_cut_past_seam"] += 1
        return CODE
    return PROSE


def _call(backend, jobs: list[_Job], prompt_of, temperature: float, stop):
    """One backend call per budget bucket, realigned to ``jobs``. Never one call per job."""
    out: list[object] = [None] * len(jobs)
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, job in enumerate(jobs):
        buckets[job.room // BUDGET_QUANTUM].append(i)
    for _, idx in sorted(buckets.items()):
        completions = backend.complete_traced(
            [prompt_of(jobs[i]) for i in idx],
            temperature=temperature,
            max_tokens=min(jobs[i].room for i in idx),
            stop=stop,
        )
        for i, completion in zip(idx, completions):
            out[i] = completion
    return out


def _code_prompt(job: _Job) -> str:
    """What pass 2 continues from -- the *bare* fence, exactly as sampling.py writes it."""
    return job.head if job.plan is None else job.head + job.plan.text + CODE_FENCE


def _continuation(job: _Job) -> str:
    if job.plan is None:
        return job.code.text
    # CODE_FENCE carries no newline of its own, so when pass 2 opens mid-line the seam reads
    # ```pythonimport torch, no fence matches, and the import sharing that line is dropped
    # (KGEN-21). The prompt above keeps the bare fence; only the assembled text gains this.
    seam = CODE_FENCE if job.code.text.startswith("\n") else CODE_FENCE + "\n"
    return job.plan.text + seam + job.code.text


def _row(job: _Job, count: Callable[[str], int]) -> Rollout:
    continuation = _continuation(job)
    # From the whole completion, not from the continuation: on a code cut the fence opened
    # before it and the imports are on the prefix's side of the seam.
    code = extract_code_block(job.text + continuation)
    return Rollout(
        rollout_id=f"{job.prefix.prefix_id}__j{job.j:02d}",
        prefix_id=job.prefix.prefix_id,
        j=job.j,
        continuation=continuation,
        code=code,
        code_sha1=build._text_sha1(code),
        n_prefix_tokens=job.spent,
        n_gen_tokens=_n_gen_tokens(job, continuation, count),
        finish_reason={
            "plan": None if job.plan is None else job.plan.finish_reason,
            "code": job.code.finish_reason,
        },
        truncation=corpus.truncation_state(_trace(job)),
    )


def _n_gen_tokens(job: _Job, continuation: str, count: Callable[[str], int]) -> int:
    """Off the sampled ids where the backend has them, off the tokenizer where it does not.

    The ids are the honest count and the text is not: pass 1's run through the stop string
    it was truncated before, and the seam holds characters nobody generated.
    """
    passes = [p for p in (job.plan, job.code) if p is not None]
    ids = [len(p.token_ids) for p in passes if p.token_ids is not None]
    return sum(ids) if len(ids) == len(passes) else count(continuation)


def _trace(job: _Job) -> dict:
    """The shape v1's ``truncation_state`` reads, so both versions decide this one way.

    Never inferred from an unterminated fence: that fires on 99% of the source run, and a
    campaign that dropped those rollouts would be dropping almost all of them.
    """
    if job.plan is None:
        return {"passes": 1, "code_finish_reason": job.code.finish_reason}
    return {
        "passes": 2,
        "plan_finish_reason": job.plan.finish_reason,
        "code_finish_reason": job.code.finish_reason,
    }


def _v1_dir(parts_glob: str) -> str:
    return os.path.dirname(os.path.dirname(parts_glob))


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()
