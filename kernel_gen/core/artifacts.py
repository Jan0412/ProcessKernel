"""Everything the loop puts on disk, and the three contracts that constrain it.

**1. The final kernel goes flat, under the canonical name.** ``autotune/eval_run.py``
globs the run dir and ``triton_lint/scan.py`` scandirs it, both NON-recursively, and
the file stem is the primary key joining generation to eval to the sweep. So the
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
"""

from __future__ import annotations

import json
import os

from triton_lint.model import staged_kernel_filename

from ..gen_config import write_generation_config
from .model import Trajectory


def round_dir(out_dir: str, round_index: int) -> str:
    return os.path.join(out_dir, "rounds", f"round_{round_index}")


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
