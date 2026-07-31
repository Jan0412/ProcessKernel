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


def _would_fire_s1_0(entry: dict) -> bool:
    """Whether the submission gate would have caught this round's file.

    Judged from `parse_status`, which the journal does record: S1.0 wraps `compile()`,
    and `compile()` parses first, so anything the linter called a syntax error is
    something the gate rejects. `empty` is the documented residual -- there is no code to
    reject, and nothing else fires either.
    """
    return entry.get("parse_status") == "syntax_error"


def main() -> None:
    rounds = _rounds()
    if not rounds:
        print("No runs/*/lint_loop.jsonl found -- nothing to verify.")
        return

    non_clean = [e for e in rounds if not e["clean"]]
    before = [e for e in non_clean if not e.get("check_ids")]
    after = [e for e in before if not _would_fire_s1_0(e)]

    by_status = collections.Counter(e.get("parse_status") for e in before)
    residual = collections.Counter(e.get("parse_status") for e in after)

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
    print(f"rounds that gain a row  : {moved}, all of them to S1.0")

    # If a syntax_error round did NOT move, the S1.0 predicate is narrower than compile().
    unmoved = [e for e in after if e.get("parse_status") == "syntax_error"]
    if unmoved:
        raise SystemExit(
            f"FAIL: {len(unmoved)} syntax_error rounds still attributed to nothing; "
            f"the S1.0 predicate is narrower than compile()"
        )
    for entry in after:
        if entry.get("parse_status") != "empty":
            raise SystemExit(
                f"FAIL: unexpected residual parse_status={entry.get('parse_status')!r}; "
                f"only 'empty' (a generation with no code at all) is a known residual"
            )
    print("OK: the only rounds left unattributed contained no code at all.")


if __name__ == "__main__":
    main()
