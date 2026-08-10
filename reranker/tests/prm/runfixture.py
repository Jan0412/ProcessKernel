"""A run directory on disk in the shape ``corpus.units()`` walks — shared by the corpus tests.

Not a test module: pytest collects ``test_*.py`` only. Written once here rather than copied
into `test_corpus.py` and `test_truncation.py`, which would let the two shapes drift.
"""

from __future__ import annotations

import json
from pathlib import Path

from reranker.src.prm.corpus import Unit, units

STOP = {"passes": 2, "plan_finish_reason": "stop", "code_finish_reason": "stop"}
LENGTH = {"passes": 2, "plan_finish_reason": "stop", "code_finish_reason": "length"}
# --think-temperature 0: one call, so sampling.py writes no plan reason at all.
SINGLE_STOP = {"passes": 1, "code_finish_reason": "stop"}
SINGLE_LENGTH = {"passes": 1, "code_finish_reason": "length"}


def attempt(problem_id=0, sample_id=0, *, level=6, rnd=0, raw="x = 1\n", trace=STOP, **over):
    """One ``attempts.jsonl`` record, carrying the fields corpus.py reads."""
    rec = {
        "stem": f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel",
        "level": level,
        "problem_id": problem_id,
        "sample_id": sample_id,
        "round": rnd,
        "prompt": "PROMPT",
        "raw": raw,
        "trace": trace,
    }
    rec.update(over)
    return rec


def verdict(sample_id=0, *, compiled=True, correctness=True, runtime=1.0, min_runtime=1.0):
    """One ``eval_results.json`` entry; ``min_runtime=None`` leaves ``runtime_stats`` empty."""
    entry = {"sample_id": sample_id, "compiled": compiled, "correctness": correctness}
    if runtime is not None:
        entry["runtime"] = runtime
    if min_runtime is not None:
        entry["runtime_stats"] = {"mean": runtime, "min": min_runtime}
    return entry


def write_round(run_dir: Path, shard: str, rnd: int, attempts, verdicts) -> None:
    """Lay out one (shard, round); ``None`` on either side leaves that file out."""
    traces = run_dir / shard / "traces" / f"round_{rnd}"
    traces.mkdir(parents=True, exist_ok=True)
    if attempts is not None:
        traces.joinpath("attempts.jsonl").write_text(
            "".join(json.dumps(a) + "\n" for a in attempts)
        )
    rounds = run_dir / shard / "rounds" / f"round_{rnd}"
    rounds.mkdir(parents=True, exist_ok=True)
    if verdicts is not None:
        rounds.joinpath("eval_results.json").write_text(json.dumps(verdicts))


def one_unit(tmp_path: Path, attempts, verdicts, *, run="runA", shard="shard_00", rnd=0) -> Unit:
    """The single ``Unit`` of a run holding exactly one (shard, round)."""
    run_dir = tmp_path / run
    write_round(run_dir, shard, rnd, attempts, verdicts)
    found, _ = units([str(run_dir)], [rnd])
    return found[0]
