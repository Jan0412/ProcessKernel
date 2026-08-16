"""Prefixes -> K measured continuations each: job B's generation half (PLAN_v2 §6).

v1 asks what a completion ended up being worth. This asks what a *prefix* is worth, by
continuing it K times from the exact context the sampler held and evaluating what comes
back.

`generate` itself is pure -- it is handed a list of prefixes and returns rows. The driver
below writes each unit's part (through stage.py's writer, so both jobs name and open it one
way), its `.meta` and the manifest; `staged_as` stays stage.py's to assign, because the map
job D reads is a projection of it.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import yaml

from kernel_gen.core.sampling import CODE_FENCE
from kernel_gen.core.text import extract_code_block
from reranker.src.config import _resolve, load_config
from reranker.src.prm import build, corpus
from reranker.src.prm.chunks import CODE, PROSE
from reranker.src.prm.rollout import prefixes

# §6 wants max_tokens sized per prefix, and the Backend takes one max_tokens per call, so the
# two meet only through a bucket. Bucketing this coarsely keeps the calls per *batch* in
# single digits (measured 7 buckets over a real shard) while a call still takes its bucket's
# *minimum*, so no prompt is ever given more than its own prefix left it. A member loses at
# most this many tokens off a budget measured at 9,855-16,249 on that shard. How much of a
# unit reaches one call is `batches`, below -- a unit is not a batch.
BUDGET_QUANTUM = 1024


ROLLOUT_MANIFEST = "rollout_manifest.json"   # job B's, beside job A's manifest.json
# What a rollout is sampled from and under, and so what a reused unit has to agree with. Not
# the whole config, deliberately: the batching, eval and grading knobs move without changing a
# single row, and a guard that refused those would be one operators route around by deleting
# the manifest. v1's build.py draws the same line with ROW_KNOBS.
SAMPLE_KNOBS = ("gen_model", "K", "temperature", "think_temperature", "max_new_tokens",
                "parts_glob")
UNIT_META = ".meta"                      # one unit's counters, beside its part
GEN_CONFIG = "generation_config.yaml"    # what lintloop.sh leaves in every shard dir
# The regime a rollout has to be sampled in, as {config attribute: generation_config key}.
# `gen_model` is the one that actually drifts -- three of the four candidate runs are
# DeepSeek and the default is gpt-oss -- but the other three cost nothing to check and a
# future run may move them.
SAMPLER = {
    "gen_model": "model",
    "temperature": "temperature",
    "think_temperature": "think_temperature",
    "max_new_tokens": "max_new_tokens",
}


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


def unit_name(prefix) -> str:
    """The ``(run, shard, round)`` this prefix belongs to, spelled as v1 spells its parts.

    Job B resolves a unit's texts by looking this name up among the v1 parts, so it is
    `build.part_name` without the suffix rather than a second convention beside it.
    """
    return f"{prefix.run_name}__{prefix.shard}__round{prefix.round}"


def units(prefix_rows) -> dict[str, list]:
    """``unit name -> its prefixes``, in the order job A wrote them.

    One part per unit, as in v1: it is the resume granularity, and the texts a unit needs
    are exactly the ones its v1 part holds -- so a unit is also all a worker has to load.
    """
    out: dict[str, list] = {}
    for p in prefix_rows:
        out.setdefault(unit_name(p), []).append(p)
    return out


def batches(prefixes, size: int):
    """``size`` prefixes at a time: what one ``generate`` call is handed.

    A unit is not a batch. Measured on a real one, 14,005 prefixes came to 70,025 rollouts in
    a single call -- ~770 MB of pass-2 prompt strings plus every Completion held at once, and
    nothing written until all of it returns. The K siblings of a prefix stay together, which
    is what lets vLLM's prefix cache collapse their shared prefill.
    """
    if size < 1:
        raise ValueError(f"prm_rollout.prefixes_per_batch must be >= 1, got {size!r}")
    for i in range(0, len(prefixes), size):
        yield prefixes[i : i + size]


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


def source_run_dirs(cfg) -> dict[str, str]:
    """``run_name -> run dir``, off v1's manifest: the only pointer back to the source run.

    v1 froze the texts, not the settings they were sampled under, so the run dir is where
    the regime a prefix came from is still recorded.
    """
    path = os.path.join(_v1_dir(_resolve(cfg.parts_glob)), build.MANIFEST)
    run_dirs = json.loads(_read(path))["config"]["run_dirs"]
    # v1's corpus.units refuses colliding basenames, so this cannot silently drop a run.
    return {os.path.basename(os.path.normpath(d)): d for d in run_dirs}


def check_gen_model(cfg, shards) -> dict[str, str]:
    """The campaign's sampler settings against the run each shard was generated with.

    Before generation, not after: by staging time the rollouts have already been sampled,
    and a `gen_model` naming another model has by then tokenized every budget with the wrong
    tokenizer and measured V-hat under a model that never wrote the prefix. Neither shows up
    in the output -- the rollouts look exactly as valid as correct ones.
    """
    dirs = source_run_dirs(cfg)
    out = {}
    for run_name, shard in shards:
        if run_name not in dirs:
            raise ValueError(
                f"{run_name} is not among the runs v1 built ({sorted(dirs)}), so the model "
                "that wrote its prefixes cannot be checked -- point parts_glob at the build "
                "job A read"
            )
        path = os.path.join(dirs[run_name], shard, GEN_CONFIG)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"no {path}: {run_name}/{shard} carries prefixes but the run dir holding the "
                "settings they were sampled under is gone. Restore it, or the campaign "
                "cannot show it continues them in the regime they came from"
            )
        source = yaml.safe_load(_read(path))
        for attr, key in SAMPLER.items():
            want, got = getattr(cfg, attr), source.get(key)
            if want != got:
                raise ValueError(
                    f"prm_rollout.{attr} is {want!r}, but {run_name}/{shard} was generated "
                    f"with {key}={got!r} ({path}). A rollout has to continue its prefix in "
                    "the regime that wrote it, and nothing downstream can see that it did not"
                )
        out[f"{run_name}/{shard}"] = source["model"]
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


# --- job B: prefixes.jsonl -> one part per unit ------------------------------------------


def prefix_caching(backend) -> bool | None:
    """Whether vLLM's prefill cache is on, read off the engine. ``None`` if unaskable.

    Not assumed: the K prompts of a prefix are byte-identical so that this cache collapses
    their shared prefill, and without it the campaign quietly pays K prefills per prefix.
    """
    try:
        return bool(backend.llm.llm_engine.vllm_config.cache_config.enable_prefix_caching)
    except AttributeError:
        return None


def unit_names(conf) -> list[str]:
    """The campaign's units, sorted -- the index space an array task selects from.

    Absolute, never a stride: PLAN_v2 §6 requires a partial resubmit to re-run the units it
    names, and a stride read off SLURM_ARRAY_TASK_COUNT would re-slice them all.
    """
    from reranker.src.prm.rollout import stage

    out_dir = _resolve(conf.out_dir)
    return sorted(units(stage.read_prefixes(os.path.join(out_dir, prefixes.PREFIXES))))


def run_rollouts(cfg, backend=None, count=None, only=None) -> dict:
    """Every unit's prefixes -> ``K`` continuations each; returns the manifest it wrote.

    Resume is by part and never by content: `code_sha1` hashes text sampled at temperature
    0.6, so a re-run unit produces different kernels and a full set of new evals. A unit whose
    part is already there is therefore skipped outright (§9).
    """
    # Deferred: stage.py imports this module, so importing it at the top would be a cycle.
    from reranker.src.prm.rollout import stage

    conf = cfg.prm_rollout
    conf.validate()
    out_dir = _resolve(conf.out_dir)
    rows = stage.read_prefixes(os.path.join(out_dir, prefixes.PREFIXES))
    # Before anything is sampled, and over every unit rather than only the ones left to do:
    # it is provenance as much as a check, and a resumed campaign records it too.
    models = check_gen_model(conf, sorted({(p.run_name, p.shard) for p in rows}))
    by_unit = units(rows)
    if only is not None and only not in by_unit:
        raise KeyError(
            f"{only} is not a unit of this campaign -- {len(by_unit)} units are, and job A's "
            f"prefixes.jsonl under {out_dir} is the only thing that names them"
        )
    done = [u for u in by_unit if os.path.exists(stage.unit_path(out_dir, u))]
    todo = {
        unit: ps
        for unit, ps in by_unit.items()
        if unit not in set(done) and (only is None or unit == only)
    }
    if done:
        _check_resume(conf, out_dir)

    def _publish(generated: int) -> dict:
        m = _job_b_manifest(conf, _metas(out_dir), models, generated)
        build.write_atomic(os.path.join(out_dir, ROLLOUT_MANIFEST), json.dumps(m, indent=2))
        return m

    caching = None
    generated = 0
    if todo:
        backend = _backend(conf) if backend is None else backend
        caching = prefix_caching(backend)
        if caching is False:
            raise ValueError(
                "vLLM prefix caching is off: the K rollouts of a prefix share one prompt "
                "precisely so their prefill is computed once, and this campaign would pay "
                f"for {conf.K} of them per prefix instead"
            )
        count = gen_counter(conf) if count is None else count
        prompts = load_prompts(conf.parts_glob)
        by_name = {os.path.basename(p): p for p in glob.glob(_resolve(conf.parts_glob))}
        for unit, ps in todo.items():
            part = by_name.get(f"{unit}.jsonl")
            if part is None:
                raise FileNotFoundError(
                    f"{unit} has prefixes but no v1 part under {conf.parts_glob} to read "
                    "their texts from -- job A and job B are reading different builds"
                )
            _run_unit(
                backend, unit, ps, load_sources(part, prompts), conf, count, out_dir, caching
            )
            generated += 1
            # After each unit, not once at the end: a job killed on its wall clock still has
            # to leave the settings its finished parts were sampled under.
            _publish(generated)

    return _publish(generated)


def _check_resume(conf, out_dir: str) -> None:
    """A finished unit is reused on its part's name alone, so the sampler must not have moved.

    Without this a job killed on its wall clock and resubmitted after a `temperature` edit
    keeps its finished units and stamps the new settings over all of them -- two regimes in
    one campaign, and a manifest that names only the second.
    """
    path = os.path.join(out_dir, ROLLOUT_MANIFEST)
    if not os.path.isfile(path):
        raise ValueError(
            f"{out_dir} holds finished units but no {ROLLOUT_MANIFEST}, so what they were "
            "sampled under cannot be checked against this run. Delete the parts, or generate "
            "into a new out_dir"
        )
    was = json.loads(_read(path)).get("config", {})
    moved = {k: (was.get(k), getattr(conf, k)) for k in SAMPLE_KNOBS if was.get(k) != getattr(conf, k)}
    if moved:
        raise ValueError(
            f"{out_dir} already holds units generated with {moved} (was, now) -- they would be "
            "reused as they are and this run's settings recorded for them. Generate into a new "
            "out_dir, or delete the parts these knobs no longer describe"
        )


def _run_unit(backend, unit, ps, sources: dict, conf, count, out_dir: str, caching) -> None:
    """One unit, batch by batch, published only when the last batch is back."""
    from reranker.src.prm.rollout import stage

    counts: Counter = Counter()
    started = time.time()
    path = stage.unit_path(out_dir, unit)
    with stage.open_part(path) as f:
        for batch in batches(ps, conf.prefixes_per_batch):
            stage.dump_rollouts(generate(backend, batch, sources, conf, count, counts), f)
        # Inside the context, so the counters are on disk before the part is renamed into
        # place: the manifest is rebuilt from them, and a resumed run has no other memory.
        build.write_atomic(
            path + UNIT_META,
            json.dumps(
                {
                    "seconds": round(time.time() - started, 1),
                    "counts": dict(counts),
                    # With the unit and not with the run: a resume has no engine to ask, and
                    # the manifest it rewrites would otherwise forget what this one sampled under.
                    "prefix_caching": caching,
                }
            ),
        )
    print(f"  {unit}: {counts['rollouts']} rollouts from {counts['prefixes']} prefixes")


def _metas(out_dir: str) -> dict[str, dict]:
    """Each finished unit's counters, keyed by unit -- this campaign's and every earlier one's."""
    from reranker.src.prm.rollout import stage

    out = {}
    for part in sorted(glob.glob(os.path.join(out_dir, stage.ROLLOUTS, "*.jsonl.gz"))):
        unit = os.path.basename(part)[: -len(".jsonl.gz")]
        meta = part + UNIT_META
        out[unit] = json.loads(_read(meta)) if os.path.isfile(meta) else {"counts": {}}
    return out


def _job_b_manifest(conf, metas: dict, models: dict, generated: int) -> dict:
    counts: Counter = Counter()
    for meta in metas.values():
        counts.update(meta["counts"])
    # True only if every unit says so; unknown as soon as one could not be asked. False cannot
    # appear -- run_rollouts refuses to start on it.
    seen = [meta.get("prefix_caching") for meta in metas.values()]
    caching = None if not seen or None in seen else all(seen)
    dirty = build._git("status", "--porcelain")
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": build._git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "config": dataclasses.asdict(conf),
        # Which model actually wrote the prefixes: V̂ is only interpretable against it.
        "gen_models": models,
        # §6: verified rather than assumed, and recorded so a later run can be compared.
        "prefix_caching": caching,
        "units": {
            unit: {
                "prefixes": meta["counts"].get("prefixes", 0),
                "rollouts": meta["counts"].get("rollouts", 0),
                "seconds": meta.get("seconds"),
                "prefix_caching": meta.get("prefix_caching"),
            }
            for unit, meta in metas.items()
        },
        "units_generated": generated,
        "units_reused": len(metas) - generated,
        "prefixes": counts["prefixes"],
        "rollouts": counts["rollouts"],
        # prose_cut_past_seam and prefix_no_budget among them: "counted, not silent" only
        # holds once something writes them down.
        "counts": dict(sorted(counts.items())),
    }


def _backend(conf):
    """vLLM with the source run's own model kwargs -- the regime the prefixes came from."""
    from kernel_gen.core.backend import VLLMBackend

    return VLLMBackend(
        conf.gen_model, max_model_len=conf.max_model_len, max_num_seqs=conf.max_num_seqs
    )


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--units" in argv:
        argv.remove("--units")
        print(len(unit_names(load_config(argv).prm_rollout)))
        return

    cfg = load_config(argv)
    only, task = None, os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is not None:
        names = unit_names(cfg.prm_rollout)
        if int(task) >= len(names):
            print(f"task {task}: only {len(names)} units, nothing to do")
            return
        only = names[int(task)]
        print(f"task {task}: unit {only}")

    manifest = run_rollouts(cfg, only=only)
    print(
        f"Rollouts: {manifest['rollouts']} from {manifest['prefixes']} prefixes over "
        f"{manifest['units_generated']} units ({manifest['units_reused']} already done)"
    )
    print(f"  prefix_caching {manifest['prefix_caching']}   counts {manifest['counts']}")


def _v1_dir(parts_glob: str) -> str:
    return os.path.dirname(os.path.dirname(parts_glob))


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


if __name__ == "__main__":
    main()
