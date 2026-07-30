"""Copy a stratified sample of real kernels into the submission test corpus.

The regression tests for this gate must run against kernels a model actually produced, not
kernels written to make the checks pass -- otherwise they prove only that the author
understood their own predicate. This pins the evidence: every distinct S1.0 message, a
sample of each other rejection, and a control group of confirmed-good kernels whose only
job is to stay accepted.

    python scripts/pin_submission_corpus.py

Run once; the result is committed. `manifest.json` records where each file came from and
what verdict it is pinned to.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from checker.core.feedback import StagedRenderer  # noqa: E402
from checker.lint import LintAnalyzer  # noqa: E402
from checker.submission import SubmissionAnalyzer  # noqa: E402

from verify_submission_gate import DEFAULT_ROOTS, clean_kernels  # noqa: E402

DEST = os.path.join(REPO_ROOT, "checker/tests/submission/real_kernels/data")

#: How many of each rejection to keep. S1.2 is small enough to take whole.
QUOTA = {"S1.0": 20, "S1.1": 20, "S1.2": 10, "S1.3": 20}
N_GOOD = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=DEST)
    args = parser.parse_args()

    submission, lint, renderer = SubmissionAnalyzer(), LintAnalyzer(), StagedRenderer()

    picked: dict[str, list] = collections.defaultdict(list)
    seen_reasons: set[str] = set()
    good: list = []

    for path in clean_kernels(DEFAULT_ROOTS):
        if renderer.render(lint.analyze_path(path)) is not None:
            continue
        source = open(path, encoding="utf-8", errors="replace").read()
        report = submission.analyze(source, path)

        if not report.findings:
            if len(good) < N_GOOD:
                good.append((path, source, None))
            continue

        check_id = min(f.check_id for f in report.findings)
        # Every distinct compiler message earns a slot even past the quota: they are the
        # constructs, and one uncovered construct is one untested branch.
        reason = report.findings[0].message.split("cannot be imported: ")[-1][:40]
        novel = check_id == "S1.0" and reason not in seen_reasons
        if len(picked[check_id]) < QUOTA[check_id] or novel:
            seen_reasons.add(reason)
            picked[check_id].append((path, source, check_id))

    os.makedirs(args.dest, exist_ok=True)
    for stale in os.listdir(args.dest):
        os.remove(os.path.join(args.dest, stale))

    manifest = []
    for check_id in sorted(picked):
        for i, (path, source, verdict) in enumerate(picked[check_id]):
            manifest.append(_write(args.dest, f"{check_id.replace('.', '_')}_{i:02d}", path, source, verdict))
    for i, (path, source, verdict) in enumerate(good):
        manifest.append(_write(args.dest, f"good_{i:02d}", path, source, verdict))

    with open(os.path.join(args.dest, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"{len(manifest)} kernels -> {args.dest}")
    for check_id in sorted(picked):
        print(f"  {len(picked[check_id]):3d}  {check_id}")
    print(f"  {len(good):3d}  accepted (the false-positive guard)")


def _write(dest: str, name: str, path: str, source: str, verdict: str | None) -> dict:
    filename = f"{name}.py"
    with open(os.path.join(dest, filename), "w") as handle:
        handle.write(source)
    run, _, stem = path.rpartition("/")
    return {
        "file": filename,
        "run": os.path.basename(run),
        "origin": stem,
        "expected": verdict,
    }


if __name__ == "__main__":
    main()
