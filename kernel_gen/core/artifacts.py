"""Everything the loop puts on disk, and the four contracts that constrain it.

**1. The final kernel goes flat, under the canonical name.** KernelBench's eval resolves
each sample by exact path and ``triton_lint/scan.py`` scandirs the run dir
non-recursively; the file stem is the primary key joining generation to eval. So the
per-round intermediates live in ``rounds/round_{r}/`` where those globs cannot see
them, and only :meth:`Trajectory.final` is written flat. A dirty round-0 kernel
landing in the run dir would be scored as if it were the answer.

**2. ``rounds/round_0/`` is itself a run dir.** It holds an unrefined generation --
same prompt, same sampler, no feedback existed yet -- so it *is* the baseline arm, and
it gets its own copy of ``generation_config.yaml`` so eval and ``load_run()`` can be
pointed straight at it.

**3. ``generation_config.yaml`` is a public API with a hand-rolled parser.**
``triton_lint/runs.py`` reads it with a flat ``key: value`` scanner that skips any line
starting with a space or a dash -- so nested values and block lists silently vanish --
and it resolves the level as ``pseudo_level or level``, *pseudo_level winning*. An arm
that wrote both keys would report level 5 on a KernelBench run at level 1, and every
downstream filename lookup would be built against files that do not exist. Hence
:func:`write_config` emits exactly one of the two.

**4. One trace record per ``(round, stem)``, describing its own arrays.**
:func:`write_traces` appends while :func:`~.trace.write_trace` overwrites, so a slot
generated twice leaves the first record pointing at the second's tokens.
:func:`prune_traces` drops the stale half before the round runs.

**Traces obey contract 1 by living outside its reach.** Everything the trace writer
produces goes under ``traces/``, never flat and never in ``rounds/``. That is not
tidiness: ``scan.py``'s docstring records that a run folder already holds 140k-175k
files and that its single non-recursive ``scandir`` pass is load-bearing, and this run
is heading for four times that. A ``.npz`` sitting flat would be invisible to the
globs (they are anchored on ``_kernel.py``) and would still be walked by every one of
them.
"""

from __future__ import annotations

import json
import os

from triton_lint.model import staged_kernel_filename

from ..gen_config import write_generation_config
from .model import Trajectory


def round_dir(out_dir: str, round_index: int) -> str:
    return os.path.join(out_dir, "rounds", f"round_{round_index}")


def trace_dir(out_dir: str, round_index: int) -> str:
    return os.path.join(out_dir, "traces", f"round_{round_index}")


def write_config(out_dir: str, config: dict, dataset: str) -> str:
    """Write ``generation_config.yaml`` to the run dir and to ``rounds/round_0/``.

    Renames ``level`` -> ``pseudo_level`` for KernelBook, matching what the legacy
    kernelbook script writes, and guarantees the other key is absent. See this
    module's docstring for why writing both would be silently destructive.
    """
    config = dict(config)
    if dataset == "kernelbook":
        config["pseudo_level"] = config.pop("level", None)
    else:
        config.pop("pseudo_level", None)

    path = write_generation_config(out_dir, config)
    write_generation_config(round_dir(out_dir, 0), config)
    return path


def write_attempts(out_dir: str, trajectories: list[Trajectory], round_index: int) -> int:
    """Persist one round's generations under ``rounds/round_{r}/``, canonically named.

    Canonical names (not ``attempt_3.txt``) so a round dir can be handed to eval or to
    the linter's scanner unchanged -- which is what makes round 0 a free baseline.
    """
    target = round_dir(out_dir, round_index)
    os.makedirs(target, exist_ok=True)

    written = 0
    for traj in trajectories:
        attempt = next((a for a in traj.attempts if a.round == round_index), None)
        if attempt is None:
            continue
        name = staged_kernel_filename(
            traj.problem.level, traj.problem.problem_id, traj.sample_id
        )
        with open(os.path.join(target, name), "w") as fh:
            fh.write(attempt.code)
        written += 1
    return written


def write_kernels(out_dir: str, trajectories: list[Trajectory]) -> int:
    """Persist each slot's final attempt flat in the run dir -- what eval scores.

    Every slot gets a file even when it never went clean. "N samples per problem" is a
    contract the whole downstream (pass@k, the sweep, the reranker's list construction)
    is built on; silently dropping the slots that stayed dirty would bias every one of
    them toward the easy problems.
    """
    os.makedirs(out_dir, exist_ok=True)

    written = 0
    for traj in trajectories:
        final = traj.final()
        if final is None:
            print(
                f"[WARN] problem {traj.problem.problem_id} sample {traj.sample_id} "
                f"produced no attempt at all -- no file written"
            )
            continue
        name = staged_kernel_filename(
            traj.problem.level, traj.problem.problem_id, traj.sample_id
        )
        with open(os.path.join(out_dir, name), "w") as fh:
            fh.write(final.code)
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
    round_index: int,
    *,
    window: int = 512,
    vocab_size: int | None = None,
    system_prompt: str = "",
) -> int:
    """One round's traces: a ``.npz`` of arrays plus a line of context per attempt.

    The two halves answer different questions and are stored apart on purpose. The
    ``.npz`` is bulk numeric data nobody reads without a reason; ``attempts.jsonl`` is
    the index over it, small enough to load whole and rich enough to decide *which*
    traces are worth opening -- which is the entire point of the DeepConf summary
    statistics on each record.

    The jsonl also carries the two things this pipeline has been discarding since it was
    written: the **full raw completion**, including the ``## Plan`` prose that
    ``write_attempts`` drops in favour of the extracted code, and the **full findings**
    with their line numbers.

    An attempt whose trace failed to assemble still gets a record, with ``trace: null``.
    Its prose and findings are worth keeping regardless, and a silently missing line
    would make the journal disagree with the kernels on disk.
    """
    from .trace import derive_scalars, summarize, write_trace

    target = trace_dir(out_dir, round_index)
    os.makedirs(target, exist_ok=True)

    records, written = [], 0
    for traj in trajectories:
        attempt = next((a for a in traj.attempts if a.round == round_index), None)
        if attempt is None:
            continue
        stem = staged_kernel_filename(
            traj.problem.level, traj.problem.problem_id, traj.sample_id
        )[: -len(".py")]

        record = {
            "stem": stem,
            "level": traj.problem.level,
            "problem_id": traj.problem.problem_id,
            "sample_id": traj.sample_id,
            "problem_name": traj.problem.name,
            "round": round_index,
            # The conversation this round, in order: the system message, the user turn the
            # model actually saw (base prompt at round 0; base + previous kernel + feedback
            # after), then the assistant completion. Stored so the trace reconstructs the
            # whole exchange without replaying the prompt builders against a pinned dataset.
            # `system_prompt` is constant across the run and repeated per record on purpose,
            # so each line is a self-contained training example.
            "system_prompt": system_prompt,
            "prompt": attempt.prompt,
            "raw": attempt.raw,
            "n_chars_code": len(attempt.code),
            "clean": bool(attempt.review and attempt.review.clean),
            # `feedback` is the rendered critic text -- the exact string folded into the
            # NEXT round's prompt; `findings` is the same verdict structured, with a lineno
            # per entry. Both kept: the text is what the model sees, the structure is what a
            # reward model reads.
            #
            # Each lineno is 1-based into `extract_code_block(raw)` -- the string the critic
            # was handed -- NOT into `raw`, and NOT into raw sliced at `code_char_start`.
            # That slice keeps the newline after the fence, the closing fence and any
            # trailing prose, so it is off by at least one line and for a single-pass run
            # is the whole completion. Resolve a lineno by re-extracting.
            "feedback": attempt.review.text if attempt.review else "",
            "findings": attempt.review.findings if attempt.review else [],
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

    append_jsonl(os.path.join(target, "attempts.jsonl"), records)
    return written


def prune_traces(out_dir: str, stems: set[str]) -> int:
    """Drop every record and ``.npz`` for ``stems``, enforcing contract 4 before a rerun.

    Keyed on the slots about to run, not on ``--skip-existing``: re-running a traced run
    without that flag regenerates everything. Returns the number of records dropped.
    """
    traces_root = os.path.join(out_dir, "traces")
    if not stems or not os.path.isdir(traces_root):
        return 0

    dropped = 0
    for entry in sorted(os.scandir(traces_root), key=lambda e: e.name):
        if not entry.is_dir() or not entry.name.startswith("round_"):
            continue
        journal = os.path.join(entry.path, "attempts.jsonl")
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
            path = os.path.join(entry.path, name)
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
