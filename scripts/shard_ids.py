"""Print the ``--problems`` range for one task of a sharded array run.

    python -m scripts.shard_ids KernelBench/level6 7 32   ->   3811-4615

Invoked with ``-m``, from the repo root, like everything else here: nothing in this
project is pip-installed, so ``kernel_gen`` resolves off the current directory. Running
it as a plain path (``python scripts/shard_ids.py``) puts ``scripts/`` on ``sys.path``
instead and the import fails.

A thin CLI over :func:`kernel_gen.core.sources.shard_range`, which holds the logic (and
the tests) for why a sparse id space has to be cut by list position rather than by id.
Used by scripts/lintloop.sh in array mode.
"""

from __future__ import annotations

import sys

from kernel_gen.core.sources import shard_range

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    try:
        print(shard_range(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
    except ValueError as exc:
        raise SystemExit(f"shard_ids: {exc}") from exc
