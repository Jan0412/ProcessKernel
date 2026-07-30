"""The gate, run against kernels a model actually wrote.

Hand-written fixtures prove the author understood their own predicate. These are 109 real
generations copied verbatim out of the run dirs (see `manifest.json` for provenance), and
they are the reason to believe the bug was found rather than invented.

The accepted half matters more than the rejected half. A false positive marks a *working*
kernel dirty and spends a GPU round repairing nothing, which is a worse failure than the
one this package fixes.
"""

from __future__ import annotations

import json
import os

import pytest

from checker.submission import SubmissionAnalyzer

DATA = os.path.join(os.path.dirname(__file__), "data")


def load_manifest() -> list[dict]:
    with open(os.path.join(DATA, "manifest.json")) as handle:
        return json.load(handle)


MANIFEST = load_manifest()
REJECTED = [row for row in MANIFEST if row["expected"]]
ACCEPTED = [row for row in MANIFEST if row["expected"] is None]


def analyze(row: dict):
    with open(os.path.join(DATA, row["file"]), encoding="utf-8") as handle:
        return SubmissionAnalyzer().analyze(handle.read(), row["file"])


def test_the_corpus_is_stratified():
    """A corpus that drifts to one check stops testing the others."""
    by_check = {row["expected"] for row in REJECTED}
    assert by_check == {"S1.0", "S1.1", "S1.2", "S1.3"}
    assert len(ACCEPTED) >= 30


@pytest.mark.parametrize("row", REJECTED, ids=lambda r: r["file"])
def test_every_pinned_bad_kernel_is_rejected(row):
    report = analyze(row)

    assert report.findings, f"{row['origin']} should be rejected by {row['expected']}"
    assert min(f.check_id for f in report.findings) == row["expected"]
    assert all(f.severity == "fail" for f in report.findings)


@pytest.mark.parametrize("row", ACCEPTED, ids=lambda r: r["file"])
def test_every_pinned_good_kernel_is_accepted(row):
    """The false-positive guard. Each of these is a real kernel the lint loop called clean
    and the gate must agree is loadable."""
    assert analyze(row).findings == []


@pytest.mark.parametrize("row", ACCEPTED, ids=lambda r: r["file"])
def test_every_accepted_kernel_really_does_compile(row):
    """Independent of the checks: ask CPython directly, so a bug in S1.0 cannot make the
    accepted half pass by agreeing with itself."""
    with open(os.path.join(DATA, row["file"]), encoding="utf-8") as handle:
        compile(handle.read(), row["file"], "exec")
