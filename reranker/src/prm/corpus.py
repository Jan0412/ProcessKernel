"""Walk the runs, join every attempt to its verdict, report what was missing (PLAN §6)."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from reranker.src.config import _resolve
from reranker.src.prm.targets import MEAN, MIN

FINISHED = "stop"
OK, TRUNCATED, UNKNOWN = "ok", "truncated", "unknown"
# sampling.py stamps passes=1 (no plan pass) or 2; a record predating the field is two-pass.
_REASONS = {1: ("code_finish_reason",), 2: ("plan_finish_reason", "code_finish_reason")}
NO_ATTEMPTS, NO_EVAL = "no_attempts", "no_eval"
# One entry in build.py's drop ledger, spelled here because this is where it is counted.
NO_EVAL_ENTRY = "no_eval_entry"

# A file that does not read cleanly is corruption, never a counted skip (PLAN §13).
_CORRUPT = "corrupt, or written by an evaluation that never finished (PLAN §13)"


@dataclass(frozen=True)
class Unit:
    """One (run, shard, round): the unit of parallelism and of one output part."""

    run_name: str
    run_dir: str
    shard: str
    round: int
    attempts_path: str
    eval_path: str


@dataclass(frozen=True)
class Labeled:
    """One attempt joined to its verdict -- everything a row needs except the cuts."""

    unit: Unit
    stem: str
    level: int
    problem_id: int
    sample_id: int
    prompt: str
    raw: str
    compiled: bool
    correct: bool
    runtimes: dict[str, float | None]  # {mean, min}: the baseline's keys, so targets divides
    truncation: str


def truncation_state(trace: dict | None) -> str:
    """``ok`` | ``truncated`` | ``unknown``, read off vLLM's own verdict (§2)."""
    # Never infer it from an unterminated fence -- that fires on 99% of gpt-oss.
    if not trace:
        return UNKNOWN
    # Keyed on the shape: a two-pass record missing its code reason must stay unknown, and a
    # pass count no sampler here writes is unknown rather than read off the two it does.
    passes = trace.get("passes", 2)
    # An exact int, as prm.rounds demands: bool is an int and 1.0 hashes as 1, so either
    # would otherwise select the single-pass table.
    if not isinstance(passes, int) or isinstance(passes, bool) or passes not in _REASONS:
        return UNKNOWN
    reasons = [trace.get(k) for k in _REASONS[passes]]
    if None in reasons:
        return UNKNOWN
    return OK if all(r == FINISHED for r in reasons) else TRUNCATED


def units(
    run_dirs: Sequence[str], rounds: Sequence[int], *, max_shards: int | None = None
) -> tuple[list[Unit], list[tuple[str, str, int, str]]]:
    """Units that have both files, and the ``(run, shard, round, reason)`` that have not."""
    # _resolve anchors relatives to the project root; abspath then strips a trailing slash.
    resolved = [os.path.abspath(_resolve(d)) for d in run_dirs]
    names = [os.path.basename(d) for d in resolved]
    collide = sorted({n for n in names if names.count(n) > 1})
    if collide:
        raise ValueError(f"run_dirs collide on basename {collide}; part files are named from it")
    rnds = [int(r) for r in rounds]
    repeat = sorted({r for r in rnds if rnds.count(r) > 1})
    if repeat:
        raise ValueError(f"rounds repeat {repeat}; two workers would write one part file")

    found: list[Unit] = []
    skipped: list[tuple[str, str, int, str]] = []
    for run_dir, run_name in zip(resolved, names):
        shards = sorted(
            d
            for d in os.listdir(run_dir)
            if d.startswith("shard_") and os.path.isdir(os.path.join(run_dir, d))
        )
        if not shards:
            raise FileNotFoundError(f"no shard_* directories in {run_dir}")
        for shard in shards[:max_shards]:
            for rnd in rnds:
                attempts = os.path.join(
                    run_dir, shard, "traces", f"round_{rnd}", "attempts.jsonl"
                )
                eval_file = os.path.join(
                    run_dir, shard, "rounds", f"round_{rnd}", "eval_results.json"
                )
                if not os.path.isfile(attempts):
                    skipped.append((run_name, shard, rnd, NO_ATTEMPTS))
                elif not os.path.isfile(eval_file):
                    skipped.append((run_name, shard, rnd, NO_EVAL))
                else:
                    found.append(
                        Unit(
                            run_name=run_name,
                            run_dir=run_dir,
                            shard=shard,
                            round=rnd,
                            attempts_path=attempts,
                            eval_path=eval_file,
                        )
                    )
    return found, skipped


def iter_labeled(unit: Unit, counts: Counter) -> Iterator[Labeled]:
    """One ``Labeled`` per attempt that has a verdict; the rest counted into ``counts``."""
    joined = verdicts(unit.eval_path)
    with open(unit.attempts_path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = _loads(line, unit.attempts_path, lineno)
            row = _row(unit, rec, joined, lineno)
            if row is None:
                # An unfinished eval rewrites a complete file with fewer entries, not a
                # broken one, so it lands here rather than in a parse error. Counted out
                # here: a bad `counts` is the caller's bug, not corruption in the data.
                counts[NO_EVAL_ENTRY] += 1
            else:
                yield row


def _row(
    unit: Unit, rec: dict, joined: dict[tuple[int, int], dict], lineno: int
) -> Labeled | None:
    """One row, or ``None`` for an attempt with no verdict; every field read named by line."""
    try:
        key = (int(rec["problem_id"]), int(rec["sample_id"]))
        verdict = joined.get(key)
        if verdict is None:
            return None
        stats = verdict.get("runtime_stats") or {}
        return Labeled(
            unit=unit,
            stem=rec["stem"],
            # From the record, never from config: a mis-set data.level cannot mislabel.
            level=int(rec["level"]),
            problem_id=key[0],
            sample_id=key[1],
            prompt=rec["prompt"],
            raw=rec["raw"],
            compiled=bool(verdict.get("compiled", False)),
            correct=bool(verdict.get("correctness", False)),
            # `runtime` is the mean and is what build_dataset grades, so both agree.
            runtimes={MEAN: _num(verdict.get("runtime")), MIN: _num(stats.get("min"))},
            truncation=truncation_state(rec.get("trace")),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{unit.attempts_path}:{lineno}: {e} -- {_CORRUPT}") from e


def verdicts(path: str) -> dict[tuple[int, int], dict]:
    """``eval_results.json`` as ``{(problem_id, sample_id): entry}`` -- the join key."""
    raw = _load_json(path)
    out: dict[tuple[int, int], dict] = {}
    # Wrapped like the reads above; a repeated key is fatal -- no way to pick a verdict.
    try:
        for problem_id, entries in raw.items():
            for entry in entries:
                key = (int(problem_id), int(entry["sample_id"]))
                if key in out:
                    raise ValueError(f"two verdicts for {key}")
                out[key] = entry
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path}: {e} -- {_CORRUPT}") from e
    return out


def _load_json(path: str) -> dict:
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: {e} -- {_CORRUPT}") from e


def _loads(line: str, path: str, lineno: int) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}:{lineno}: {e} -- {_CORRUPT}") from e


def _num(value) -> float | None:
    return None if value is None else float(value)
