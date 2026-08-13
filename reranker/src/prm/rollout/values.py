"""What the evals said -> V̂ per prefix: job D's measurement half (PLAN_v2 §6).

N1 is the whole point: every number here comes from the prefix's own rollouts. The v1
verdict for the completion the prefix was cut out of is one draw from a *different*
distribution -- the continuation the model actually took, kept because it was kept -- so
pooling it in would bias V̂ toward the observed path. Nothing in this module reads v1's parts.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from reranker.src.config import _resolve, load_config
from reranker.src.data.labels import load_baseline_times
from reranker.src.prm import corpus, targets
from reranker.src.prm.build import (
    NO_BASELINE,
    NO_FINISH_REASON,
    NO_RUNTIME,
    TRUNCATED,
    write_atomic,
)
from reranker.src.prm.rollout import prefixes, stage
from reranker.src.prm.targets import MEAN, MIN

VALUES = "values.jsonl"
VALUES_MANIFEST = "values_manifest.json"

NO_EVAL_ENTRY = corpus.NO_EVAL_ENTRY
# Applied in this order, and the order is what makes the ledger readable: a rollout that is
# both truncated and unevaluated is counted once, as truncated (§6). v1's names, from v1's
# build.py, so one campaign's drops can be read against the other's.
REASONS = (TRUNCATED, NO_FINISH_REASON, NO_EVAL_ENTRY, NO_BASELINE, NO_RUNTIME)
# Not in REASONS: these drop the *prefix*, so no row survives to carry them. They land on
# the campaign ledger instead, which is why value_for takes one.
TOO_FEW_ROLLOUTS = "too_few_rollouts"
NO_ROLLOUTS = "no_rollouts"
LEDGER = REASONS + (TOO_FEW_ROLLOUTS, NO_ROLLOUTS)


@dataclass(frozen=True)
class Eval:
    """One rollout's verdict, and whether the dedup had it share another rollout's eval."""

    verdict: dict | None
    shared: bool = False


@dataclass(frozen=True)
class Value:
    """One row of ``values.jsonl`` (PLAN_v2 §5).

    ``n_rollouts`` is the merge key with v1: a v1 row is this same measurement at
    ``n_rollouts = 1``. ``scores`` is kept per rollout so `label_mode`, `speedup_stat` and
    the speed knobs stay re-derivable from a built dataset without re-running an eval.
    """

    prefix_id: str
    K_requested: int
    n_rollouts: int
    n_dropped: dict
    n_dedup_shared: int
    n_compiled: int
    n_correct: int
    v_binary: float
    v_graded: float
    se_binary: float
    se_graded: float
    scores: list


def read_verdicts(runs_dir: str, run_name: str) -> dict[tuple[int, int], dict]:
    """Every eval shard's ``eval_results.json`` as one ``(problem_id, sample_id) -> entry`` table.

    One file per shard is not a preference: `add_to_eval_results_file` appends by
    load-rewrite of the whole file, so two jobs sharing one would drop each other's results.
    Job D is where they come back together.
    """
    # Anchored on the shard number, not left as the glob's prefix match: `prm_rollout_v1_s*`
    # also matches `prm_rollout_v1_smoke_s00` -- and this repo already names campaigns X and
    # X_smoke -- so the smoke run's verdicts would be read into the real campaign's table,
    # where both stage dense sample ids from 0 over the same level. The anchor also drops an
    # archived `..._s02.tar` beside them, which would otherwise read as an unevaluated shard.
    shard_dir = re.compile(rf"^{re.escape(run_name)}_s\d+$")
    run_dirs = sorted(
        d
        for d in glob.glob(os.path.join(runs_dir, f"{run_name}_s*"))
        if shard_dir.match(os.path.basename(d))
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"no eval run dirs matching {run_name}_s<NN> under {runs_dir} -- either the "
            "campaign was staged under a different eval_run_name, or nothing has been "
            "staged yet"
        )
    table: dict[tuple[int, int], dict] = {}
    for run_dir in run_dirs:
        path = os.path.join(run_dir, stage.EVAL_RESULTS)
        if not os.path.isfile(path):
            # An empty table would be indistinguishable from a shard whose every kernel
            # failed, and would drop that shard's prefixes as `no_eval_entry` instead.
            raise FileNotFoundError(
                f"{path} does not exist: shard {os.path.basename(run_dir)} has not been "
                "evaluated yet. Run job C for it before values.py, or its prefixes are "
                "silently dropped rather than measured"
            )
        for key, entry in corpus.verdicts(path).items():
            if key in table:
                raise ValueError(
                    f"(problem {key[0]}, sample {key[1]}) is graded by two eval shards, the "
                    f"second in {os.path.basename(run_dir)} -- the shard plan partitions the "
                    "problems, so two run dirs were pointed at overlapping subsets and "
                    "neither verdict can be taken as the sample's"
                )
            table[key] = entry
    return table


def join(rows, rollout_map: dict[str, dict], verdicts: dict[tuple[int, int], dict]):
    """``rollout_id -> Eval``: what the harness said about each rollout, via stage.py's map.

    The map is the only thing that knows a rollout's synthetic sample id, and deduped
    rollouts resolve through the kernel that was actually staged -- so a saved eval stays a
    measurement instead of becoming a lost one.
    """
    out: dict[str, Eval] = {}
    for r in rows:
        if r.rollout_id not in rollout_map:
            raise KeyError(
                f"{r.rollout_id} is in no eval run dir: {stage.ROLLOUT_MAP} does not place "
                "it, which means it was generated after the staging pass. Re-stage before "
                "scoring, or the prefix reports a K the campaign never bought"
            )
        placed = rollout_map[r.rollout_id]
        out[r.rollout_id] = Eval(
            verdicts.get((placed["problem_id"], placed["sample_id"])),
            bool(placed.get("shared", False)),
        )
    return out


def value_for(prefix, rows, joined, baselines, cfg, counts) -> Value | None:
    """Aggregate one prefix's rollouts into its measured value, or ``None`` if too few survive.

    ``counts`` is the campaign ledger and is updated for every drop, including the ones that
    happened inside a prefix this returns ``None`` for: those evals were paid for, and a
    ledger that forgot them would make the campaign look cheaper than it was.
    """
    baseline = baselines.get(prefix.level, {}).get(prefix.problem_id)
    dropped: Counter = Counter()
    scores, n_compiled, n_correct, shared = [], 0, 0, 0

    def drop(reason: str) -> None:
        dropped[reason] += 1
        counts[reason] += 1

    # Sorted, not in arrival order: `scores` is a stored column and job B returns its
    # rollouts bucketed by token budget, so the unsorted order is not stable across reruns.
    for r in sorted(rows, key=lambda r: r.rollout_id):
        if r.truncation == corpus.TRUNCATED:
            drop(TRUNCATED)
            continue
        # Not folded into the branch above: v1 tells the two apart because a rollout whose
        # finish reason never arrived is a harness gap, not a generation that ran out of room.
        if r.truncation != corpus.OK:
            drop(NO_FINISH_REASON)
            continue
        ev = joined[r.rollout_id]
        if ev.verdict is None:
            drop(NO_EVAL_ENTRY)
            continue
        compiled = bool(ev.verdict.get("compiled", False))
        target = targets.target_for(
            compiled=compiled,
            correct=bool(ev.verdict.get("correctness", False)),
            ours=_runtimes(ev.verdict),
            baseline=baseline,
            mode=cfg.label_mode,
            stat=cfg.speedup_stat,
            lo=cfg.speedup_lo,
            hi=cfg.speedup_hi,
            quant=cfg.speed_quant,
        )
        if target is None:
            # Only a correct kernel reaches here. v1's PRM-4 distinction, restated: a kernel
            # the harness never timed is not an absent baseline, and one entry for both
            # would hide an eval that ran and reported nothing.
            timed = targets.usable((baseline or {}).get(cfg.speedup_stat))
            drop(NO_RUNTIME if timed else NO_BASELINE)
            continue
        scores.append(target.value)
        n_compiled += compiled
        n_correct += target.label
        shared += ev.shared

    n = len(scores)
    if n < cfg.min_rollouts:
        # A V̂ from too few draws is a v1 label wearing a v2 schema, and mixing those is
        # exactly what this plan forbids -- so the prefix leaves the corpus, not just the list.
        counts[TOO_FEW_ROLLOUTS] += 1
        return None
    v_binary = n_correct / n
    v_graded = sum(scores) / n
    return Value(
        prefix_id=prefix.prefix_id,
        K_requested=prefix.K,
        n_rollouts=n,
        # Every reason, zeros included: a reader must not have to decide whether a missing
        # key is a zero or a reason this build never knew about.
        n_dropped={reason: dropped[reason] for reason in REASONS},
        n_dedup_shared=shared,
        n_compiled=n_compiled,
        n_correct=n_correct,
        v_binary=v_binary,
        v_graded=v_graded,
        se_binary=math.sqrt(v_binary * (1 - v_binary) / n),
        # One rollout has no measurable spread, and `statistics.stdev` raises on it rather
        # than returning 0. min_rollouts is what keeps such a prefix out of the corpus at
        # all; this branch only decides what the row says when it is deliberately let in.
        se_graded=statistics.stdev(scores) / math.sqrt(n) if n > 1 else 0.0,
        scores=scores,
    )


# --- the pass: prefixes + rollouts + every eval shard -> values.jsonl --------------------


def build_values(cfg) -> dict:
    """Score a whole campaign. CPU only, minutes long, and it reads no v1 label anywhere."""
    rollout_cfg = cfg.prm_rollout
    rollout_cfg.validate()
    out_dir = _resolve(rollout_cfg.out_dir)

    prefix_rows = stage.read_prefixes(os.path.join(out_dir, prefixes.PREFIXES))
    with open(os.path.join(out_dir, stage.ROLLOUT_MAP)) as f:
        rollout_map = json.load(f)
    parts = sorted(glob.glob(os.path.join(out_dir, stage.ROLLOUTS, "*.jsonl.gz")))
    rows = [r for part in parts for r in stage.read_rollouts(part)]
    verdicts = read_verdicts(_resolve(rollout_cfg.eval_runs_dir), rollout_cfg.eval_run_name)
    joined = join(rows, rollout_map, verdicts)
    baselines = load_baseline_times(_resolve(rollout_cfg.baseline_timing_json))

    by_prefix = defaultdict(list)
    for r in rows:
        by_prefix[r.prefix_id].append(r)

    counts: Counter = Counter()
    measured = []
    for prefix in prefix_rows:
        mine = by_prefix.get(prefix.prefix_id)
        if not mine:
            # Job B is resumable per unit, so a campaign scored before every unit has run
            # is a normal state -- counted, and legible against `prefixes` in the manifest.
            counts[NO_ROLLOUTS] += 1
            continue
        value = value_for(prefix, mine, joined, baselines, rollout_cfg, counts)
        if value is not None:
            measured.append(value)

    write_atomic(
        os.path.join(out_dir, VALUES),
        "".join(json.dumps(dataclasses.asdict(v)) + "\n" for v in measured),
    )
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": dataclasses.asdict(rollout_cfg),
        "parts": len(parts),
        # Read against each other: prefixes - values is what the ledger below has to explain.
        "prefixes": len(prefix_rows),
        "rollouts": len(rows),
        "evals": len(verdicts),
        "values": len(measured),
        "shared": sum(v.n_dedup_shared for v in measured),
        "dropped": {reason: counts[reason] for reason in LEDGER},
    }
    write_atomic(os.path.join(out_dir, VALUES_MANIFEST), json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> None:
    manifest = build_values(load_config(None if argv is None else list(argv)))
    fired = {r: n for r, n in manifest["dropped"].items() if n}
    print(
        f"Measured {manifest['values']} of {manifest['prefixes']} prefixes from "
        f"{manifest['rollouts']} rollouts against {manifest['evals']} evals"
    )
    print(f"  dropped {fired or 'nothing'}")


def _runtimes(verdict: dict) -> dict[str, float | None]:
    """The harness's two timings under the names `targets.speedups` grades them by.

    ``runtime`` is the mean and is what `build_dataset` grades, so v1 and v2 agree; the
    mapping is corpus._row's, restated rather than imported because that one reads a v1
    attempt record and this reads an eval entry.
    """
    stats = verdict.get("runtime_stats") or {}
    return {MEAN: _num(verdict.get("runtime")), MIN: _num(stats.get("min"))}


def _num(value) -> float | None:
    return None if value is None else float(value)


if __name__ == "__main__":
    main()
