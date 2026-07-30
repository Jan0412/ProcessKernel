"""Join findings to measured outcomes.

The question this answers: *does each static check actually predict a wrong or slow
kernel?* We join the scan's findings back onto ``eval_results.json`` (correctness,
runtime) and the eager baseline (speedup), then report, per check, how the flagged
kernels differ from the unflagged ones.

Importable first, CLI second -- ``rows(...)`` returns dicts so a notebook can do
``pd.DataFrame(rows(run_dir, findings))``.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterator

from .core.analyzer import Analyzer
from .lint import LintAnalyzer
from .runs import iter_samples, speedup


def load_findings(path: str) -> dict[tuple[int, int, int], dict]:
    """``{(level, problem_id, sample_id): report}``."""
    index: dict[tuple[int, int, int], dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            key = (row.get("level"), row.get("problem_id"), row.get("sample_id"))
            if None not in key:
                index[key] = row  # type: ignore[index]
    return index


def rows(
    run_dir: str, findings_path: str, analyzer: Analyzer | None = None
) -> Iterator[dict]:
    """One row per evaluated sample: outcome + static findings.

    The check columns come from the analyzer's registry, so adding a check adds a column
    instead of silently going missing from every report."""
    index = load_findings(findings_path)
    check_ids = (analyzer or LintAnalyzer()).registry.check_ids

    for sample in iter_samples(run_dir):
        report = index.get((sample.level, sample.problem_id, sample.sample_id))
        if report is None:
            continue
        summary = report.get("summary", {})
        checks = set(summary.get("check_ids", []))

        row = {
            "run_name": sample.run_name,
            "level": sample.level,
            "problem_id": sample.problem_id,
            "sample_id": sample.sample_id,
            "compiled": sample.compiled,
            "correct": sample.correct,
            "runtime": sample.runtime,
            "speedup": speedup(sample),
            "parse_status": report.get("parse_status"),
            "n_kernels": summary.get("n_kernels"),
            "n_launches": summary.get("n_launches"),
            "launches_in_loop": summary.get("launches_in_loop"),
            "fallback_ops": summary.get("fallback_ops", []),
            "wasted_bytes": summary.get("wasted_bytes_lower_bound", 0),
            "n_fail": summary.get("n_fail", 0),
        }
        for check_id in check_ids:
            row[check_id] = check_id in checks
        yield row


def summarize(run_dir: str, findings_path: str, analyzer: Analyzer | None = None) -> dict:
    analyzer = analyzer or LintAnalyzer()
    data = list(rows(run_dir, findings_path, analyzer))
    if not data:
        return {"n": 0}

    checks = [k for k in data[0] if k in set(analyzer.registry.check_ids)]
    out: dict = {
        "n_samples": len(data),
        "n_correct": sum(1 for r in data if r["correct"]),
        "checks": {},
    }

    for check in checks:
        hit = [r for r in data if r[check]]
        miss = [r for r in data if not r[check]]
        if not hit:
            out["checks"][check] = {"n": 0, "rate": 0.0}
            continue
        out["checks"][check] = {
            "n": len(hit),
            "rate": len(hit) / len(data),
            "correct_rate_when_flagged": _mean([r["correct"] for r in hit]),
            "correct_rate_when_clean": _mean([r["correct"] for r in miss]),
            "median_speedup_when_flagged": _median([r["speedup"] for r in hit]),
            "median_speedup_when_clean": _median([r["speedup"] for r in miss]),
        }
    return out


def _mean(values: list) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _median(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 4) if vals else None


def print_summary(run_dir: str, findings_path: str) -> None:
    s = summarize(run_dir, findings_path)
    if not s.get("n_samples"):
        print("no rows joined -- did the scan cover this run?")
        return

    n = s["n_samples"]
    print(f"samples: {n:,}   correct: {s['n_correct']:,} ({s['n_correct'] / n:.1%})\n")
    header = f"{'check':6} {'flagged':>9} {'rate':>7} {'correct|flag':>13} {'correct|clean':>14} {'spd|flag':>9} {'spd|clean':>10}"
    print(header)
    print("-" * len(header))
    for check, stats in s["checks"].items():
        if not stats["n"]:
            continue
        print(
            f"{check:6} {stats['n']:>9,} {stats['rate']:>6.1%} "
            f"{_fmt(stats.get('correct_rate_when_flagged'), pct=True):>13} "
            f"{_fmt(stats.get('correct_rate_when_clean'), pct=True):>14} "
            f"{_fmt(stats.get('median_speedup_when_flagged')):>9} "
            f"{_fmt(stats.get('median_speedup_when_clean')):>10}"
        )


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}" if pct else f"{value:.2f}x"
