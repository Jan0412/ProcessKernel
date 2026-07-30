"""How many kernels the lint loop called clean cannot actually be loaded by Python.

This is the KGEN-14 headline measurement. `Review.clean` means "the linter had nothing to
say", which is not the same as "the evaluator can import this file and instantiate
ModelNew" -- so this replays the submission gate over every slot the loop declared clean
and tallies what it rejects, by check.

    python scripts/verify_submission_gate.py

Until `checker/submission` exists this fails with ModuleNotFoundError, which is the correct
answer before phase F.

Runs are deduplicated by directory name: the repo's `runs/` holds copies of several of the
traced runs, and counting a slot twice would inflate the headline.

**Cleanliness is recomputed, not read off the journal.** `lint_loop.jsonl` records what the
linter said on the day the run executed, and the linter has changed since -- BUG-25 moved
F1.2 from `info` to `fail` when the file has no model class, so thousands of slots the
journal calls clean are already rejected by today's linter and were never the target of
this gate. Trusting the recorded flag would credit this change with defects that a linter
fix already caught. So each shipped kernel is re-linted here and the gate is applied only
to the ones the *current* critic would still call clean.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from checker.core.feedback import StagedRenderer  # noqa: E402
from checker.lint import LintAnalyzer  # noqa: E402
from checker.submission import SubmissionAnalyzer  # noqa: E402

DEFAULT_ROOTS = (
    os.path.join(REPO_ROOT, "runs"),
    "/path/to/workdir/KernelBench/runs",
)


def clean_kernels(roots: tuple[str, ...]) -> list[str]:
    """The file each clean slot shipped, one per slot, deduplicated across roots."""
    seen: dict[str, str] = {}
    for root in roots:
        for journal in sorted(glob.glob(os.path.join(root, "*", "lint_loop.jsonl"))):
            run_dir = os.path.dirname(journal)
            run = os.path.basename(run_dir)
            if run in seen:
                continue
            seen[run] = run_dir

    paths = []
    for run, run_dir in sorted(seen.items()):
        for line in open(os.path.join(run_dir, "lint_loop.jsonl")):
            row = json.loads(line)
            if not row["clean"]:
                continue
            stem = (
                f"level_{row['level']}_problem_{row['problem_id']}"
                f"_sample_{row['sample_id']}_kernel.py"
            )
            path = os.path.join(run_dir, stem)
            if os.path.exists(path):
                paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", default=",".join(DEFAULT_ROOTS))
    parser.add_argument("--dump", default=None, help="write the rejected paths here")
    args = parser.parse_args()

    paths = clean_kernels(tuple(args.roots.split(",")))
    submission = SubmissionAnalyzer()
    lint = LintAnalyzer()
    renderer = StagedRenderer()

    tally: collections.Counter[str] = collections.Counter()
    rejected = []
    still_clean = 0
    for path in paths:
        source = open(path, encoding="utf-8", errors="replace").read()
        if renderer.render(lint.analyze_path(path)) is not None:
            continue  # today's linter already stops this slot; not this gate's business
        still_clean += 1

        report = submission.analyze(source, path)
        if not report.findings:
            continue
        # First-defect-wins, so the tally is disjoint and the columns add up.
        first = min(f.check_id for f in report.findings)
        tally[first] += 1
        rejected.append((first, path, report.findings[0].message))

    print(f"slots the journal recorded as clean : {len(paths)}")
    print(f"still lint-clean under today's linter: {still_clean}")
    print(f"REJECTED by the submission gate      : {len(rejected)} "
          f"({100 * len(rejected) / max(still_clean, 1):.2f}% of those)")
    for check_id, count in sorted(tally.items()):
        print(f"   {count:5d}  {check_id}")

    if args.dump:
        with open(args.dump, "w") as handle:
            for check_id, path, message in rejected:
                handle.write(json.dumps({"check": check_id, "path": path, "message": message}) + "\n")
        print(f"wrote {args.dump}", file=sys.stderr)


if __name__ == "__main__":
    main()
