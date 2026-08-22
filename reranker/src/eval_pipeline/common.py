"""Shared helpers for the reranker evaluation pipeline.

Single source of truth for how run dirs, baseline timings and reference
architectures are located, so the eval-table builder (Stage 1a) and the scorer
(Stage 1b) agree byte-for-byte with each other and with the training data build
(``reranker/src/data/build_dataset.py``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

# Staged kernel filename: level_{L}_problem_{P}_sample_{S}_kernel.py
FNAME_RE = re.compile(r"level_(\d+)_problem_(\d+)_sample_(\d+)_kernel\.py$")

# The reranker-independent columns describing one evaluated kernel. The scorer
# joins its (score_logit, score_sigmoid, ...) onto these via (run_name, kernel_file).
EVAL_TABLE_FIELDS = [
    # run_name is the LEAF (<run>__shard_00__round1) and is the join key; `generator` is
    # the run root, the unit a "problem" is grouped by when runs are not pooled.
    "run_name", "generator", "round",
    "level", "problem_id", "problem_name", "sample_id", "kernel_file",
    "compiled", "correct", "runtime_mean", "runtime_min", "runtime_std",
    "baseline_mean", "baseline_min",
]

# Reuse the exact baseline-timing loader the training build uses, so a kernel's
# speedup here == its speedup in the trained lists. Returns
# ``{level: {problem_id: {"mean", "min"}}}`` (or {} if the file is missing).
from reranker.src.data.labels import load_baseline_times as _load_baseline_times  # noqa: E402


def git_sha(cwd: str | os.PathLike) -> str | None:
    """Short git SHA of the checkout at ``cwd`` (None if not a repo)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


class RefArchIndex:
    """Resolve reference-architecture sources from ``KernelBench/level{L}/{pid}_*.py``.

    Local, folder-only (no ``datasets.load_dataset``), so custom levels 5/6 that
    only exist on disk work. Matches the resolver in ``reranker_eval_quality.ipynb``.
    """

    def __init__(self, kernelbench_dir: str | os.PathLike):
        self.kb_dir = Path(kernelbench_dir)
        self._index: dict[int, dict[int, Path]] = {}

    def _level_index(self, level: int) -> dict[int, Path]:
        if level not in self._index:
            lvl_dir = self.kb_dir / "KernelBench" / f"level{level}"
            idx: dict[int, Path] = {}
            if lvl_dir.is_dir():
                for p in lvl_dir.glob("*.py"):
                    m = re.match(r"(\d+)_", p.name)
                    if m:
                        idx[int(m.group(1))] = p
            else:
                print(f"[WARN] reference-arch dir not found: {lvl_dir}")
            self._index[level] = idx
            print(f"level {level}: indexed {len(idx)} reference archs from {lvl_dir}")
        return self._index[level]

    def path(self, level: int, problem_id: int) -> Path | None:
        return self._level_index(level).get(problem_id)

    def problem_name(self, level: int, problem_id: int) -> str | None:
        p = self.path(level, problem_id)
        return p.name if p is not None else None

    def source(self, level: int, problem_id: int) -> str | None:
        p = self.path(level, problem_id)
        return p.read_text() if p is not None else None


def iter_run_kernels(run_dir: str | os.PathLike):
    """Yield ``(kernel_path, level, problem_id, sample_id)`` for a run dir.

    Only files directly in the run dir matching the staged-kernel naming are
    returned (sorted), matching how the reranked-generation output and the
    training build stage kernels.
    """
    run_dir = Path(run_dir)
    for path in sorted(run_dir.glob("level_*_problem_*_sample_*_kernel.py")):
        m = FNAME_RE.search(path.name)
        if not m:
            continue
        yield path, int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_eval_results(run_dir: str | os.PathLike) -> dict | None:
    """Load ``eval_results.json`` for a run dir (None + warning if missing)."""
    p = Path(run_dir) / "eval_results.json"
    if not p.is_file():
        print(f"[WARN] no eval_results.json in {run_dir}, skipping run")
        return None
    return json.loads(p.read_text())


def run_name_of(run_dir: str | os.PathLike) -> str:
    return os.path.basename(str(run_dir).rstrip("/"))
