"""Batch scanner.

A run folder holds 140k-175k files on GPFS, so: one ``os.scandir`` pass filtered by
the filename pattern (never a per-file ``stat``, never a shell glob), a process pool
over the parse work, and a per-file try/except so that no single malformed generation
can abort the run.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections.abc import Iterator

from .core.analyzer import Analyzer
from .core.model import FileReport
from .core.naming import parse_kernel_filename
from .lint import LintAnalyzer
from .runs import load_run


def iter_kernel_files(run_dir: str, limit: int | None = None) -> list[str]:
    """Every ``level_*_problem_*_sample_*_kernel.py`` in *run_dir*, sorted."""
    names = []
    with os.scandir(run_dir) as it:
        for entry in it:
            meta = parse_kernel_filename(entry.name)
            if meta is not None:
                names.append((meta, entry.name))
    names.sort()
    paths = [os.path.join(run_dir, name) for _, name in names]
    return paths[:limit] if limit else paths


_ONLY: set[str] | None = None
_RUN_NAME: str | None = None
_ANALYZER: Analyzer | None = None


def _init_worker(only: set[str] | None, run_name: str | None, analyzer: Analyzer) -> None:
    global _ONLY, _RUN_NAME, _ANALYZER
    _ONLY = only
    _RUN_NAME = run_name
    _ANALYZER = analyzer


def _work(path: str) -> str:
    try:
        report = _ANALYZER.analyze_path(path, only=_ONLY)
    except Exception as exc:  # noqa: BLE001 - a bad file must not kill the batch
        report = FileReport(path=path, parse_status="read_error")
        report.summary = {"notes": [f"{type(exc).__name__}: {exc}"]}
    report.run_name = _RUN_NAME
    return report.to_json()


def scan_run(
    run_dir: str,
    out_path: str,
    workers: int | None = None,
    limit: int | None = None,
    only: set[str] | None = None,
    analyzer: Analyzer | None = None,
) -> dict:
    analyzer = analyzer or LintAnalyzer()
    run_dir = run_dir.rstrip("/")
    try:
        run_name = load_run(run_dir).run_name
    except (OSError, ValueError):
        run_name = os.path.basename(run_dir)

    paths = iter_kernel_files(run_dir, limit)
    total = len(paths)
    workers = workers or min(os.cpu_count() or 4, 32)

    print(f"[checker] {run_name}: {total:,} kernel files, {workers} workers", file=sys.stderr)

    stats = {"total": total, "written": 0, "by_status": {}, "by_check": {}}

    with open(out_path, "w", encoding="utf-8") as out:
        for i, line in enumerate(_run_pool(paths, workers, only, run_name, analyzer), start=1):
            out.write(line + "\n")
            stats["written"] += 1
            _tally(stats, line)
            if i % 2000 == 0 or i == total:
                print(f"\r[checker] {i:,}/{total:,}", end="", file=sys.stderr, flush=True)

    print(file=sys.stderr)
    return stats


def _run_pool(
    paths: list[str],
    workers: int,
    only: set[str] | None,
    run_name: str | None,
    analyzer: Analyzer,
) -> Iterator[str]:
    if workers <= 1:
        _init_worker(only, run_name, analyzer)
        for path in paths:
            yield _work(path)
        return

    ctx = mp.get_context("fork")
    initargs = (only, run_name, analyzer)
    with ctx.Pool(workers, initializer=_init_worker, initargs=initargs) as pool:
        yield from pool.imap_unordered(_work, paths, chunksize=64)


def _tally(stats: dict, line: str) -> None:
    import json

    row = json.loads(line)
    status = row["parse_status"]
    stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
    for check_id in row["summary"].get("check_ids", []):
        stats["by_check"][check_id] = stats["by_check"].get(check_id, 0) + 1
