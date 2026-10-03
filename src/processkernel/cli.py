"""``pk <command> [args]``: one command per pipeline step.

Each command runs its module as ``python -m`` would, so ``pk <command> --help`` shows that
module's own options.
"""

from __future__ import annotations

import runpy
import sys

COMMANDS = {
    # generation
    "stage-kernelbook": "processkernel.generation.kernelbook.stage",
    "generate": "processkernel.generation.generate",
    "lint": "processkernel.checker",
    # ORM
    "orm-data": "processkernel.orm.data.build_dataset",
    "orm-lists": "processkernel.orm.lists",
    "train-orm": "processkernel.orm.train",
    # PRM labels
    "build-prm": "processkernel.prm.data.build",
    "split-prm": "processkernel.prm.data.splits",
    "prm-stats": "processkernel.prm.data.stats",
    # PRM rollout campaign
    "prefixes": "processkernel.prm.rollout.prefixes",
    "rollout": "processkernel.prm.rollout.rollout",
    "orm-score": "processkernel.prm.rollout.orm_score",
    "stage-rollouts": "processkernel.prm.rollout.stage",
    "calibrate": "processkernel.prm.rollout.calibrate",
    "values": "processkernel.prm.rollout.values",
    "lists": "processkernel.prm.rollout.lists",
    "rollout-stats": "processkernel.prm.rollout.stats",
    # PRM training
    "train-prm": "processkernel.prm.train.train",
    "rank-eval": "processkernel.prm.train.rank_eval",
    # search
    "search": "processkernel.prm.search.run",
    "search-report": "processkernel.prm.search.report",
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    rows = "\n".join(f"  {name:<{width}}  python -m {mod}" for name, mod in COMMANDS.items())
    return f"usage: pk <command> [args]\n\ncommands:\n{rows}\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"pk: unknown command {command!r}\n\n{usage()}", file=sys.stderr)
        return 2
    sys.argv = [f"pk {command}", *rest]
    runpy.run_module(COMMANDS[command], run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
