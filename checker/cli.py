"""Command line interface.

    python -m checker scan   runs/<run_name> --out findings.jsonl
    python -m checker report runs/<run_name> --findings findings.jsonl
    python -m checker check  path/to/kernel.py
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="checker")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="analyze every kernel file in a run folder")
    scan.add_argument("run_dir")
    scan.add_argument("--out", default="findings.jsonl")
    scan.add_argument("--workers", type=int, default=None)
    scan.add_argument("--limit", type=int, default=None, help="first N files (smoke test)")
    scan.add_argument("--checks", default=None, help="comma-separated check ids, e.g. F1.2,F1.4")

    rep = sub.add_parser("report", help="join findings with eval_results and summarize")
    rep.add_argument("run_dir")
    rep.add_argument("--findings", required=True)
    rep.add_argument("--out", default=None, help="also write joined rows as JSONL")

    one = sub.add_parser("check", help="analyze a single kernel file and print findings")
    one.add_argument("path")
    one.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "scan":
        from .scan import scan_run

        only = set(args.checks.split(",")) if args.checks else None
        stats = scan_run(args.run_dir, args.out, args.workers, args.limit, only)
        print(f"\nwrote {stats['written']:,} rows to {args.out}")
        print(f"parse status: {stats['by_status']}")
        if stats["by_check"]:
            print("files per check:")
            total = max(stats["written"], 1)
            for check_id, count in sorted(stats["by_check"].items()):
                print(f"  {check_id}  {count:>8,}  ({count / total:.1%})")
        return 0

    if args.command == "report":
        from .report import print_summary, rows

        print_summary(args.run_dir, args.findings)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                for row in rows(args.run_dir, args.findings):
                    fh.write(json.dumps(row) + "\n")
            print(f"\nwrote joined rows to {args.out}")
        return 0

    if args.command == "check":
        from . import analyze_file

        report = analyze_file(args.path)
        if args.json:
            print(report.to_json())
            return 0
        print(f"{report.path}  [{report.parse_status}]")
        print(f"  summary: {report.summary}")
        if not report.findings:
            print("  no findings")
        for finding in report.findings:
            print(f"\n  [{finding.severity.upper():4}] {finding.check_id}  {finding.message}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
