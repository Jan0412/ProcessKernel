"""Run-folder data layer.

Everything the analysis needs is derivable from a run folder plus the two
repo-level directories it points at -- no reranker, no built dataset:

    runs/<run_name>/
        generation_config.yaml   -> pseudo_level, model, run_name
        eval_results.json        -> {problem_id: [{sample_id, compiled, correctness,
                                     runtime, runtime_stats{...}, metadata{hardware}}]}
        level_{L}_problem_{P}_sample_{S}_kernel.py

    KernelBench/level{L}/{P}_*.py                  -> the reference PyTorch source
    timing/{GPU}/baseline_time_torch.json          -> eager baseline, keyed
                                                      level{L} -> "{P}_Name.py" -> {mean, ...}

``speedup = baseline.mean / sample.runtime``  (eval_results stores no speedup itself).
"""

from __future__ import annotations

import functools
import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass

from .core.naming import staged_kernel_filename

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELBENCH_DIR = os.path.join(REPO_ROOT, "KernelBench")
TIMING_DIR = os.path.join(REPO_ROOT, "timing")


@dataclass
class RunInfo:
    run_dir: str
    run_name: str
    level: int
    model: str | None = None
    backend: str | None = None
    num_samples: int | None = None


@dataclass
class SampleRef:
    """One generated kernel plus its measured outcome."""

    run_name: str
    level: int
    problem_id: int
    sample_id: int
    kernel_path: str
    compiled: bool
    correct: bool
    runtime: float | None  # ms
    runtime_min: float | None
    hardware: str | None
    error_name: str | None


def _read_yaml_scalars(path: str) -> dict[str, str]:
    """Minimal ``key: value`` reader -- generation_config.yaml is flat, and this keeps
    checker dependency-free (no PyYAML)."""
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].rstrip()
            if not line or line.startswith((" ", "-", "\t")) or ":" not in line:
                continue
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip().strip("'\"")
    return out


def _level_from_run_name(run_name: str) -> int | None:
    m = re.search(r"level(\d+)", run_name)
    return int(m.group(1)) if m else None


def load_run(run_dir: str) -> RunInfo:
    run_dir = run_dir.rstrip("/")
    run_name = os.path.basename(run_dir)
    cfg = _read_yaml_scalars(os.path.join(run_dir, "generation_config.yaml"))

    level_str = cfg.get("pseudo_level") or cfg.get("level")
    level = int(level_str) if level_str and level_str.isdigit() else _level_from_run_name(run_name)
    if level is None:
        raise ValueError(f"cannot determine level for run {run_dir!r}")

    num_samples = cfg.get("num_samples")
    return RunInfo(
        run_dir=run_dir,
        run_name=cfg.get("run_name") or run_name,
        level=level,
        model=cfg.get("model"),
        backend=cfg.get("backend"),
        num_samples=int(num_samples) if num_samples and num_samples.isdigit() else None,
    )


def load_eval_results(run_dir: str) -> dict:
    path = os.path.join(run_dir, "eval_results.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def iter_samples(run_dir: str, limit: int | None = None) -> Iterator[SampleRef]:
    """Yield every evaluated sample of a run, joined to its kernel file."""
    info = load_run(run_dir)
    results = load_eval_results(run_dir)

    count = 0
    for problem_id_str, samples in results.items():
        problem_id = int(problem_id_str)
        for entry in samples:
            sample_id = entry.get("sample_id")
            if sample_id is None:
                continue
            meta = entry.get("metadata") or {}
            stats = entry.get("runtime_stats") or {}
            yield SampleRef(
                run_name=info.run_name,
                level=info.level,
                problem_id=problem_id,
                sample_id=sample_id,
                kernel_path=os.path.join(
                    run_dir, staged_kernel_filename(info.level, problem_id, sample_id)
                ),
                compiled=bool(entry.get("compiled")),
                correct=bool(entry.get("correctness")),
                runtime=entry.get("runtime"),
                runtime_min=stats.get("min"),
                hardware=meta.get("hardware"),
                error_name=meta.get("runtime_error_name"),
            )
            count += 1
            if limit and count >= limit:
                return


# --------------------------------------------------------------------------
# Reference PyTorch source + eager baseline
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _level_index(level: int) -> dict[int, str]:
    """``{problem_id: filename}`` for ``KernelBench/level{L}/``."""
    directory = os.path.join(KERNELBENCH_DIR, f"level{level}")
    index: dict[int, str] = {}
    if not os.path.isdir(directory):
        return index
    for name in os.listdir(directory):
        head, _, _ = name.partition("_")
        if head.isdigit() and name.endswith(".py"):
            index[int(head)] = name
    return index


def reference_filename(level: int, problem_id: int) -> str | None:
    """``2`` at level 5 -> ``"2_CustomizeLayer.py"``."""
    return _level_index(level).get(problem_id)


def reference_path(level: int, problem_id: int) -> str | None:
    name = reference_filename(level, problem_id)
    if name is None:
        return None
    return os.path.join(KERNELBENCH_DIR, f"level{level}", name)


def reference_source(level: int, problem_id: int) -> str | None:
    path = reference_path(level, problem_id)
    if path is None or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _gpu_dir(hardware: str | None) -> str:
    """Map the hardware string in eval_results to a folder under ``timing/``."""
    if hardware and "A100" in hardware:
        return "A100"
    if hardware and "H100" in hardware:
        return "H100"
    return "A100"


@functools.lru_cache(maxsize=8)
def _baselines(gpu: str) -> dict:
    path = os.path.join(TIMING_DIR, gpu, "baseline_time_torch.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def baseline_time(level: int, problem_id: int, hardware: str | None = None) -> dict | None:
    """Eager PyTorch baseline stats ``{mean, std, min, max}`` for a problem."""
    name = reference_filename(level, problem_id)
    if name is None:
        return None
    return _baselines(_gpu_dir(hardware)).get(f"level{level}", {}).get(name)


def speedup(sample: SampleRef) -> float | None:
    """``baseline.mean / runtime``. ``None`` when the sample was wrong or unmeasured."""
    if not sample.correct or not sample.runtime:
        return None
    base = baseline_time(sample.level, sample.problem_id, sample.hardware)
    if not base or not base.get("mean"):
        return None
    return base["mean"] / sample.runtime
