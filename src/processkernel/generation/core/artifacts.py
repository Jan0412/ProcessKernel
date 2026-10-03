"""Everything generation puts on disk, and the four contracts that constrain it.

**1. The kernel goes flat, under the canonical name.** KernelBench's eval resolves each
sample by exact path and ``src/processkernel/checker/scan.py`` scandirs the run dir non-recursively; the
file stem is the primary key joining generation to eval.

**2. One run, one folder.** The run dir holds the kernels, ``generation_config.yaml``
and, once graded, ``eval_results.json``; traces go to ``traces/``. Every downstream
reader (eval, the PRM corpus, the ORM dataset builder) reads that folder, found through
:func:`layout`.

**3. ``generation_config.yaml`` is a public API with a hand-rolled parser.**
``src/processkernel/checker/runs.py`` reads it with a flat ``key: value`` scanner that skips any line
starting with a space or a dash -- so nested values and block lists silently vanish --
and it resolves the level as ``pseudo_level or level``, *pseudo_level winning*. A run
that wrote both keys would report level 5 on a KernelBench run at level 1, and every
downstream filename lookup would be built against files that do not exist. Hence
:func:`write_config` emits exactly one of the two.

**4. One trace record per stem, describing its own arrays.**
:func:`write_traces` appends while :func:`~.trace.write_trace` overwrites, so a slot
generated twice leaves the first record pointing at the second's tokens.
:func:`prune_traces` drops the stale half before the run starts.

**Traces live under ``traces/``, never flat.** ``scan.py``'s single non-recursive
``scandir`` pass over a run folder is load-bearing at 100k+ files; a ``.npz`` sitting flat
would be invisible to the ``_kernel.py`` globs and still walked by every one of them.
"""

from __future__ import annotations

import json
import os

from processkernel.checker.core.naming import staged_kernel_filename

from ..gen_config import write_generation_config
from .model import Trajectory


def eval_run_name(out_dir: str) -> str:
    """The name eval resolves as ``runs/<name>``. NOT basename: a sharded run is
    ``runs/<run>/shard_07``, two levels deep. Last ``runs`` wins; falls back to basename.
    """
    parts = os.path.normpath(os.path.abspath(out_dir)).split(os.sep)
    if "runs" in parts:
        tail = parts[len(parts) - 1 - parts[::-1].index("runs") + 1 :]
        if tail:
            return "/".join(tail)
    return os.path.basename(os.path.normpath(out_dir))


def trace_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "traces")


def layout(run_dir: str) -> tuple[str, str]:
    """``(kernel dir, trace dir)`` of a run: the run dir itself and its ``traces/``.

    Runs made by the retired lint loop, the paper's runs among them, kept this generation
    in ``rounds/round_0/`` and ``traces/round_0/``, with a later, repaired kernel at the
    run root; those two are read instead where they exist.
    """
    kernels = os.path.join(run_dir, "rounds", "round_0")
    traces = os.path.join(run_dir, "traces", "round_0")
    return (
        kernels if os.path.isdir(kernels) else run_dir,
        traces if os.path.isdir(traces) else trace_dir(run_dir),
    )


def write_config(out_dir: str, config: dict, dataset: str) -> str:
    """Write ``generation_config.yaml`` to the run dir.

    Renames ``level`` -> ``pseudo_level`` for KernelBook, matching what the legacy
    kernelbook script writes, and guarantees the other key is absent. See this
    module's docstring for why writing both would be silently destructive.
    """
    config = dict(config)
    if dataset == "kernelbook":
        config["pseudo_level"] = config.pop("level", None)
    else:
        config.pop("pseudo_level", None)

    return write_generation_config(out_dir, config)


def write_kernels(out_dir: str, trajectories: list[Trajectory]) -> int:
    """Persist each slot's kernel flat in the run dir -- what eval scores.

    Every slot gets a file, even an empty one. "N samples per problem" is a contract the
    whole downstream (pass@k, the reranker's list construction) is built on; silently
    dropping slots would bias every one of them toward the easy problems.
    """
    os.makedirs(out_dir, exist_ok=True)

    written = 0
    for traj in trajectories:
        attempt = traj.last
        if attempt is None:
            print(
                f"[WARN] problem {traj.problem.problem_id} sample {traj.sample_id} "
                f"produced no attempt at all -- no file written"
            )
            continue
        name = staged_kernel_filename(
            traj.problem.level, traj.problem.problem_id, traj.sample_id
        )
        with open(os.path.join(out_dir, name), "w") as fh:
            fh.write(attempt.code)
        written += 1
    return written


def write_trace_config(out_dir: str, config: dict) -> str:
    """Run-level facts about the capture, written once to ``traces/trace_config.json``.

    Model id, ``logprobs_mode``, top-K and vocabulary size are constant for a run, so
    they live here rather than on every one of ~330,000 attempt records. ``vocab_size``
    in particular is not decoration: self-certainty is a KL against the uniform
    distribution over the vocabulary, so a reader that guesses it wrong gets a plausible
    number that is off by a constant nobody can recover later.
    """
    target = os.path.join(out_dir, "traces")
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, "trace_config.json")
    with open(path, "w") as fh:
        json.dump(config, fh, indent=2)
    return path


def write_traces(
    out_dir: str,
    trajectories: list[Trajectory],
    *,
    window: int = 512,
    vocab_size: int | None = None,
    system_prompt: str = "",
) -> int:
    """Each attempt's trace: a ``.npz`` of arrays plus a line of context in ``attempts.jsonl``.

    The two halves answer different questions and are stored apart on purpose. The
    ``.npz`` is bulk numeric data nobody reads without a reason; ``attempts.jsonl`` is
    the index over it, small enough to load whole and rich enough to decide *which*
    traces are worth opening -- which is the point of the DeepConf summary statistics on
    each record. It also carries the **full raw completion**, including the ``## Plan``
    prose that the kernel file drops in favour of the extracted code.

    An attempt whose trace failed to assemble still gets a record, with ``trace: null``.
    Its prose is worth keeping regardless, and a silently missing line would make the
    journal disagree with the kernels on disk.
    """
    from .trace import derive_scalars, summarize, write_trace

    target = trace_dir(out_dir)
    records: list[dict] = []
    written = 0
    for traj in trajectories:
        for attempt in traj.attempts:
            os.makedirs(target, exist_ok=True)
            stem = staged_kernel_filename(
                traj.problem.level, traj.problem.problem_id, traj.sample_id
            )[: -len(".py")]

            record = {
                "stem": stem,
                "level": traj.problem.level,
                "problem_id": traj.problem.problem_id,
                "sample_id": traj.sample_id,
                "problem_name": traj.problem.name,
                # The conversation in order: the system message, the user turn the model
                # saw, then the assistant completion. `system_prompt` is constant across the
                # run and repeated per record on purpose, so each line is a self-contained
                # training example.
                "system_prompt": system_prompt,
                "prompt": attempt.prompt,
                "raw": attempt.raw,
                # The extracted kernel, verbatim. Re-extracting `raw` at read time returns a
                # DIFFERENT string whenever the extractor's ranking has changed since the
                # capture (KGEN-20: 292 of 10,510 real records had drifted).
                "code": attempt.code,
                "trace": None,
                "confidence": {},
            }

            if attempt.trace is not None:
                scalars = derive_scalars(
                    attempt.trace.topk_lp, attempt.trace.sampled_lp, vocab_size=vocab_size
                )
                record["trace"] = {"file": f"{stem}.npz", **attempt.trace.meta}
                record["confidence"] = summarize(scalars, window=window)
                write_trace(os.path.join(target, f"{stem}.npz"), attempt.trace)
                written += 1

            records.append(record)

    if records:
        append_jsonl(os.path.join(target, "attempts.jsonl"), records)
    return written


def prune_traces(out_dir: str, stems: set[str]) -> int:
    """Drop every record and ``.npz`` for ``stems``, enforcing contract 4 before a rerun.

    Keyed on the slots about to run, not on ``--skip-existing``: re-running a traced run
    without that flag regenerates everything. Returns the number of records dropped.
    """
    target = trace_dir(out_dir)
    if not stems or not os.path.isdir(target):
        return 0

    dropped = 0
    journal = os.path.join(target, "attempts.jsonl")
    stale_files = {f"{stem}.npz" for stem in stems}

    if os.path.exists(journal):
        kept = []
        for record in read_jsonl(journal):
            if record.get("stem") not in stems:
                kept.append(record)
                continue
            dropped += 1
            if record.get("trace"):  # the name that record claims, not the default
                stale_files.add(record["trace"]["file"])
        # Via a temp file: a crash mid-prune must not truncate the journal.
        tmp = journal + ".tmp"
        with open(tmp, "w") as fh:
            for record in kept:
                fh.write(json.dumps(record) + "\n")
        os.replace(tmp, journal)

    # Unconditional, not only for dropped records: write_traces writes each .npz in its
    # loop and appends the journal at the end, so a crash between leaves orphans.
    for name in stale_files:
        path = os.path.join(target, name)
        if os.path.exists(path):
            os.unlink(path)

    return dropped


def append_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
