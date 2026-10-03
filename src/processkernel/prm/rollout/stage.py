"""Rollouts -> kernels the eval harness reads, and the map job D joins back through (§6).

Staging is one campaign-wide pass, not a per-unit one: sample ids are dense per problem and the
dedup spans prefixes, and neither is decidable from a single unit's rows.
"""

from __future__ import annotations

import contextlib
import dataclasses
import glob
import gzip
import json
import os
import socket
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from processkernel.config import _resolve, load_config
from processkernel.orm.data.build_dataset import _staged_kernel_path
from processkernel.prm.data.build import write_atomic
from processkernel.prm.rollout import prefixes, rollout


ROLLOUTS = "rollouts"                # the dir of gzipped parts, one per unit
ROLLOUT_MAP = "rollout_map.json"
STAGE_MANIFEST = "stage_manifest.json"
EVAL_RESULTS = "eval_results.json"   # the harness's, in the run dir -- read here, never written


def homes(prefix_rows) -> dict[str, tuple[int, int]]:
    """``prefix_id -> (level, problem_id)``: what a rollout row is evaluated against.

    A rollout carries its `prefix_id` and nothing about the problem, which is deliberate -- the
    prefix is where that already lives, and copying it onto every one of K rows would be a
    second place for it to be wrong.

    Uniqueness is checked, not assumed: `prefix_id` carries the run *tag*, and two run names
    may map onto one tag, which nothing upstream refuses. A comprehension would keep the last
    of a colliding pair and file the other's rollouts against the survivor's problem.
    """
    where: dict[str, tuple[int, int]] = {}
    for p in prefix_rows:
        if p.prefix_id in where:
            raise ValueError(
                f"two prefixes share the id {p.prefix_id!r} -- prefix_id carries the run tag, "
                "so two run_tags entries pointing at one tag collide on it. Give each "
                "(run, level) its own tag, or one prefix's rollouts are graded against the "
                "other's problem"
            )
        where[p.prefix_id] = (p.level, p.problem_id)
    return where


def assign(rollouts, where: dict[str, tuple[int, int]], dedup: bool = True) -> list:
    """Give every rollout the ``(level, problem_id, sample_id)`` its eval will be filed under.

    The dedup key carries the problem, not just the sha. A kernel is evaluated against *its*
    problem's reference architecture, so identical bytes under two problems are two different
    measurements -- and `extract_code_block` returns "" for a rollout that emitted no fence, so
    every empty body in the campaign collides on one sha.

    Assignment walks the rollouts in id order rather than the order they arrive in: `generate`
    returns them bucketed by budget, and letting that reach the sample ids would restage the same
    kernel under a different id on a rerun, against an `eval_results.json` written under the old one.
    """
    placement: dict[str, dict | None] = {}
    owner: dict[tuple, dict] = {}
    used: Counter = Counter()
    for r in sorted(rollouts, key=lambda r: r.rollout_id):
        if r.rollout_id in placement:
            raise ValueError(
                f"two rollouts share the id {r.rollout_id!r} -- both would be staged under one "
                f"sample id and the other would be written by nothing. {ROLLOUTS} is globbed "
                "whole, so a copy of a part is read twice"
            )
        if r.prefix_id not in where:
            raise KeyError(
                f"{r.rollout_id} continues prefix {r.prefix_id!r}, which this campaign's "
                "prefixes.jsonl does not hold -- stage.py is being run over rollouts from a "
                "different job A build than the one it read"
            )
        level, problem = where[r.prefix_id]
        key = (level, problem, r.code_sha1)
        if dedup and key in owner:
            placement[r.rollout_id] = None
            continue
        placement[r.rollout_id] = owner[key] = {
            "level": level,
            "problem_id": problem,
            "sample_id": used[level, problem],
        }
        used[level, problem] += 1
    return [dataclasses.replace(r, staged_as=placement[r.rollout_id]) for r in rollouts]


def rollout_map(staged, where: dict[str, tuple[int, int]]) -> dict[str, dict]:
    """``rollout_id -> the eval that covers it``: job D's join, derived from the rows.

    Every rollout is in it, deduped ones included -- they resolve to the kernel that was
    actually staged, flagged ``shared`` so job D can count what the dedup saved. Dropping them
    instead would turn a saved eval into a lost measurement.

    Keyed through `where` rather than by sha alone, for the reason `assign` deduplicates by
    problem: an unfenced rollout's empty body collides across problems, and the owner picked
    by sha alone could be another problem's kernel.
    """
    owner = {
        (where[r.prefix_id], r.code_sha1): r.staged_as for r in staged if r.staged_as is not None
    }
    out = {}
    for r in staged:
        if r.staged_as is not None:
            out[r.rollout_id] = {**r.staged_as, "shared": False}
            continue
        placed = owner.get((where[r.prefix_id], r.code_sha1))
        if placed is None:
            raise ValueError(
                f"{r.rollout_id} was deduped against a kernel that is not in these rows -- the "
                "map has to be built over the whole campaign at once, because that is the scope "
                "the dedup ran at"
            )
        out[r.rollout_id] = {**placed, "shared": True}
    return out


@dataclass(frozen=True)
class Shard:
    """One eval job: its own run dir, its own slice of the problems, its own results file."""

    index: int
    run_name: str
    subset: tuple[int, int]     # EvalConfig.subset, inclusive at both ends
    problems: tuple[int, ...]
    num_samples_per_problem: int
    n_staged: int


def plan_shards(staged, cfg) -> list[Shard]:
    """Partition the staged kernels into the eval jobs that will grade them.

    Contiguous ranges, because `subset` is the only filter the harness actually honours:
    `problem_ids` is declared on `EvalConfig` and never read by its `main`, so a shard cannot
    hand over an arbitrary set of problems.

    One run dir per shard. `add_to_eval_results_file` appends by load-rewrite of the whole
    `eval_results.json`, so two jobs pointed at one file would each drop the other's results.
    """
    counts = _counts(staged)
    problems = sorted(counts)
    total = sum(counts.values())
    groups: list[list[int]] = []
    current: list[int] = []
    running = 0
    for problem in problems:
        current.append(problem)
        running += counts[problem]
        # No guard on the group count is needed to stay within eval_shards: every closed group
        # holds at least total/eval_shards, so once that many are closed the total is spent and
        # the flush below has nothing left to add.
        if running >= total / cfg.eval_shards:
            groups.append(current)
            current, running = [], 0
    if current:
        groups.append(current)
    return [
        Shard(
            index=i,
            run_name=f"{cfg.eval_run_name}_s{i:02d}",
            subset=(group[0], group[-1]),
            problems=tuple(group),
            # One number for the whole shard, and the harness walks range() of it: below the
            # busiest problem's count, that problem's tail is never evaluated at all.
            num_samples_per_problem=max(counts[p] for p in group),
            n_staged=sum(counts[p] for p in group),
        )
        for i, group in enumerate(groups)
    ]


@contextlib.contextmanager
def open_part(path: str):
    """The unit's part, open for writing and renamed into place only on a clean exit.

    Rename and not a plain write, for §9's reason: a part that exists means a unit that
    finished, so a job killed part-way through one must leave nothing a rerun would skip
    over. A context manager because job B writes a unit in batches -- it cannot hold one in
    memory (14,005 prefixes measured on a real unit) and must not publish it until the last
    batch is back.

    The temp name must be unique across hosts, not just within one: job B's array spans
    nodes and out_dir is shared scratch, so pid alone can collide between two nodes.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
    with gzip.open(tmp, "wt") as f:
        yield f
    os.replace(tmp, path)


def dump_rollouts(rollouts, f) -> None:
    """Append rows to an open part. One row per line, the shape `read_rollouts` parses."""
    for r in rollouts:
        f.write(json.dumps(dataclasses.asdict(r)) + "\n")


def write_rollouts(rollouts, path: str) -> str:
    """One unit's rows, gzipped, renamed into place. Returns the path."""
    with open_part(path) as f:
        dump_rollouts(rollouts, f)
    return path


def read_rollouts(path: str):
    with gzip.open(path, "rt") as f:
        return [rollout.Rollout(**json.loads(line)) for line in f]


def write_kernels(staged, shard: Shard, runs_dir: str) -> int:
    """This shard's kernels, under the names the harness fetches them by. Returns how many.

    Refuses a run dir that already holds kernels. A re-stage assigns sample ids over whatever
    rollouts exist at that moment, so the previous pass's files keep ids that now mean something
    else -- and its `eval_results.json`, keyed by `(problem_id, sample_id)`, would be read back
    through the new map. Clearing the dir is the operator's call, not this function's.

    The results file is refused too, and not only as a hint to clear it: with the kernels gone
    and it left behind, the harness skips every re-staged sample as already evaluated. That
    failure is silent, which the other one is not.
    """
    run_dir = os.path.join(runs_dir, shard.run_name)
    leftover = glob.glob(os.path.join(run_dir, "*_kernel.py"))
    if leftover or os.path.exists(os.path.join(run_dir, EVAL_RESULTS)):
        held = "staged kernels" if leftover else EVAL_RESULTS
        raise FileExistsError(
            f"{run_dir} already holds {held}. Sample ids are assigned per staging pass, "
            f"so these belong to a different one -- remove {shard.run_name} (kernels and "
            f"{EVAL_RESULTS} both) before staging again"
        )
    os.makedirs(run_dir, exist_ok=True)
    mine = set(shard.problems)
    n = 0
    for r in staged:
        if r.staged_as is None or r.staged_as["problem_id"] not in mine:
            continue
        write_atomic(_staged_kernel_path(run_dir, **r.staged_as), r.code)
        n += 1
    return n


# --- the pass: every unit's rollouts -> kernels on disk, once per campaign ---------------


def stage_campaign(cfg) -> dict:
    """Assign, write the kernels, and record what the eval jobs have to be launched with.

    One pass over the whole campaign, not one per unit: sample ids are dense *per problem* and
    the dedup spans prefixes, so a unit on its own cannot decide either. Generation stays
    per-unit and resumable; this runs once, after it.
    """
    rollout_cfg = cfg.prm_rollout
    rollout_cfg.validate()
    out_dir = _resolve(rollout_cfg.out_dir)

    prefix_rows = read_prefixes(os.path.join(out_dir, prefixes.PREFIXES))
    parts = sorted(glob.glob(os.path.join(out_dir, ROLLOUTS, "*.jsonl.gz")))
    if not parts:
        raise FileNotFoundError(
            f"no rollout parts under {os.path.join(out_dir, ROLLOUTS)}: nothing has been "
            "generated for this campaign yet, so there is nothing to stage"
        )

    by_part = {part: read_rollouts(part) for part in parts}
    where = homes(prefix_rows)
    staged = assign(
        [r for rows in by_part.values() for r in rows], where, rollout_cfg.dedup_by_code_sha1
    )
    shards = plan_shards(staged, rollout_cfg)

    runs_dir = _resolve(rollout_cfg.eval_runs_dir)
    for shard in shards:
        write_kernels(staged, shard, runs_dir)

    # Back into the rows before the map is written: the map is a projection of `staged_as`, and
    # a part left holding nulls would be the one file that cannot say where its rollouts landed.
    placed = {r.rollout_id: r for r in staged}
    for part, rows in by_part.items():
        write_rollouts([placed[r.rollout_id] for r in rows], part)

    manifest = _manifest(rollout_cfg, staged, shards, len(parts), prefix_rows)
    write_atomic(os.path.join(out_dir, ROLLOUT_MAP), json.dumps(rollout_map(staged, where)))
    write_atomic(os.path.join(out_dir, STAGE_MANIFEST), json.dumps(manifest, indent=2))
    return manifest


def unit_path(out_dir: str, unit: str) -> str:
    """Where one unit's rollouts live. Both jobs name the file through here, not by hand.

    The generation job writes it and this pass globs it back; two spellings would leave a unit
    generated and never staged, with a campaign smaller than the one that was paid for as the
    only symptom.
    """
    return os.path.join(_resolve(out_dir), ROLLOUTS, f"{unit}.jsonl.gz")


def read_prefixes(path: str) -> list:
    with open(path) as f:
        return [prefixes.Prefix(**json.loads(line)) for line in f]


def _manifest(rollout_cfg, staged, shards: list[Shard], n_parts: int, prefix_rows) -> dict:
    n_staged = sum(1 for r in staged if r.staged_as is not None)
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # So that "deduped: 0" can be told from dedup_by_code_sha1 off (§7's control arm).
        "config": dataclasses.asdict(rollout_cfg),
        "parts": n_parts,
        # Read against each other: the gap is a campaign smaller than the one job A planned.
        "prefixes": len(prefix_rows),
        "prefixes_with_rollouts": len({r.prefix_id for r in staged}),
        "rollouts": len(staged),
        "staged": n_staged,
        # A saving, not a bug (§12) -- but only legible if the pass writes down what it collapsed.
        "deduped": len(staged) - n_staged,
        "eval_runs_dir": _resolve(rollout_cfg.eval_runs_dir),
        # Passed straight through to eval_from_generations. Recorded rather than
        # re-typed into a launch script: that is how a campaign gets graded under trial counts
        # its own config never named.
        "eval": {
            "num_correct_trials": rollout_cfg.num_correct_trials,
            "num_perf_trials": rollout_cfg.num_perf_trials,
            "timeout": rollout_cfg.eval_timeout,
        },
        "shards": [
            {
                "index": s.index,
                "run_name": s.run_name,
                "level": _level(staged),
                "subset": list(s.subset),
                "problems": len(s.problems),
                "num_samples_per_problem": s.num_samples_per_problem,
                "n_staged": s.n_staged,
                # What the harness will actually walk: it takes range(num_samples_per_problem)
                # for every problem in the range, whether a kernel was staged there or not.
                # n_work - n_staged is the eval time the shard plan spends on gaps.
                "n_work": (s.subset[1] - s.subset[0] + 1) * s.num_samples_per_problem,
            }
            for s in shards
        ],
    }


def _level(staged) -> int:
    return next(r.staged_as["level"] for r in staged if r.staged_as is not None)


def main(argv=None) -> None:
    manifest = stage_campaign(load_config(None if argv is None else list(argv)))
    print(
        f"Staged {manifest['staged']} kernels from {manifest['rollouts']} rollouts "
        f"({manifest['deduped']} deduped) over {len(manifest['shards'])} shards"
    )
    for s in manifest["shards"]:
        print(
            f"  {s['run_name']}  problems {s['subset'][0]}-{s['subset'][1]}  "
            f"num_samples_per_problem {s['num_samples_per_problem']}  "
            f"staged {s['n_staged']} of {s['n_work']} slots"
        )


def _counts(staged) -> dict[int, int]:
    """How many sample ids each problem actually uses; deduped rollouts hold none."""
    levels = defaultdict(set)
    counts: Counter = Counter()
    for r in staged:
        if r.staged_as is None:
            continue
        levels[r.staged_as["level"]].add(r.staged_as["problem_id"])
        # max + 1, not a tally: it is the id range the harness walks that has to be covered.
        counts[r.staged_as["problem_id"]] = max(
            counts[r.staged_as["problem_id"]], r.staged_as["sample_id"] + 1
        )
    if len(levels) > 1:
        raise ValueError(
            f"this campaign spans levels {sorted(levels)}, and EvalConfig takes one level per "
            "run. Two levels in one run dir also collide in eval_results.json, which is keyed "
            "by problem_id alone -- give each level its own campaign out_dir and eval_run_name"
        )
    return dict(counts)


if __name__ == "__main__":
    main()
