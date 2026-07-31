"""How many rounds the mechanism table explains by nothing at all.

This is the KGEN-18 headline measurement. `report_lint` buckets each round under the
checks it attributes to that round, and it used to read `check_ids` -- the *linter's*
summary. A file that does not parse gives the linter no tree, so no lint check runs and
the round lands in no row, while still counting toward the denominator: the rows silently
stop summing to the total. Measured over the six real journals, that is 1139 of 3812
non-clean rounds, 29.9%, three times larger than the 3.7% gate-block rate the change was
aimed at.

    python scripts/verify_attribution.py

`compile()` runs the parser first, so every file that fails `ast.parse` also fails
`compile()` and fires S1.0 -- which is why routing the table through what the prompt
actually said collapses the gap to the handful of generations that contained no code at
all and therefore fire no check anywhere.

**The historical journals cannot be fixed by this change, and are not.** They predate
`shown_check_ids`, so they never recorded what was shown; `_shown_check_ids` falls back to
`check_ids` for them and reproduces their old numbers exactly. This script therefore
reports two things: the gap as those journals render today, and the gap that remains once
the same rounds are re-attributed the way a post-fix journal would record them -- which is
the bound the fix is claiming, not a measurement of an already-fixed run.

Runs are deduplicated by directory name, matching verify_submission_gate.py.
"""

from __future__ import annotations

import ast
import collections
import glob
import json
import os


def _rounds() -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for path in sorted(glob.glob("runs/*/lint_loop.jsonl")):
        name = os.path.basename(os.path.dirname(path))
        if name in seen:
            continue
        seen.add(name)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                out.extend(
                    e for e in json.loads(line)["rounds"] if "clean" in e
                )
    return out


def _gate_check(entry: dict) -> str | None:
    """Which submission check would have caught this round's file, if any.

    Judged from `parse_status`, which the journal does record. S1.0 wraps `compile()`, and
    `compile()` parses first, so anything the linter called a syntax error is something the
    gate rejects. An empty file is the opposite case -- it compiles fine, so S1.0 is right
    to stay silent, and S1.1 owns it because a module with no body binds no `ModelNew`
    (KGEN-22).
    """
    status = entry.get("parse_status")
    if status == "syntax_error":
        return "S1.0"
    if status == "empty":
        return "S1.1"
    return None


def main() -> None:
    rounds = _rounds()
    if not rounds:
        print("No runs/*/lint_loop.jsonl found -- nothing to verify.")
        return

    non_clean = [e for e in rounds if not e["clean"]]
    before = [e for e in non_clean if not e.get("check_ids")]
    after = [e for e in before if _gate_check(e) is None]

    by_status = collections.Counter(e.get("parse_status") for e in before)
    residual = collections.Counter(e.get("parse_status") for e in after)
    gained = collections.Counter(_gate_check(e) for e in before if _gate_check(e))

    n = len(non_clean)
    print(f"rounds journaled        : {len(rounds)}")
    print(f"did NOT go clean        : {n}")
    print("-" * 60)
    print(f"attributed to no check  : {len(before):>5}  ({len(before) / n:.2%})  <- today")
    for status, count in by_status.most_common():
        print(f"     parse_status={status:<13}: {count}")
    print("-" * 60)
    print(f"...once routed through what the prompt actually said:")
    print(f"attributed to no check  : {len(after):>5}  ({len(after) / n:.2%})  <- after")
    for status, count in residual.most_common():
        print(f"     parse_status={status:<13}: {count}")
    print("-" * 60)
    moved = len(before) - len(after)
    print(f"rounds that gain a row  : {moved}")
    for check_id, count in sorted(gained.items()):
        print(f"     -> {check_id}: {count}")

    # A syntax_error round that did not move means the S1.0 predicate is narrower than
    # compile(); an empty one means S1.1 is not owning the case its own analyzer assigns it.
    if after:
        raise SystemExit(
            f"FAIL: {len(after)} non-clean rounds still attributed to no check "
            f"({dict(residual)}); every one should be caught by the submission gate"
        )
    print("OK: every non-clean round is attributed to at least one check.")


if __name__ == "__main__":
    main()
