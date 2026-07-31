"""How many trajectories ship a different kernel because of the loadability rank term.

`Trajectory.final()` picks the first clean attempt, else the best by `_rank`. KGEN-14 adds
a term to that ordering, and it is the one change in the fix that can alter which file a
slot ships even when nothing else about the slot is broken. Guessing at how many is not
good enough, so this replays every multi-round trajectory in the run dirs under the old
ordering and the new one and reports the disagreements.

    python scripts/verify_rank_blast_radius.py

**Every move must be from an unloadable attempt to a loadable one.** A move in the other
direction means the term is in the wrong position in the tuple.

Cleanliness is held at whatever the journal recorded, so this isolates the ranking change
from the critic change -- the two land together but they are separate claims.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from checker.submission import SubmissionAnalyzer  # noqa: E402

from verify_submission_gate import DEFAULT_ROOTS  # noqa: E402


def old_rank(round_data: dict, loadable: bool) -> tuple:
    parses = round_data.get("parse_status") in (None, "ok", "partial")
    return (not parses, round_data.get("n_fail", 0), round_data.get("n_warn", 0),
            round_data["round"])


def new_rank(round_data: dict, loadable: bool) -> tuple:
    parses = round_data.get("parse_status") in (None, "ok", "partial")
    return (not parses, not loadable, round_data.get("n_fail", 0),
            round_data.get("n_warn", 0), round_data["round"])


def final(rounds: list[tuple[dict, bool]], rank) -> int:
    for data, _ in rounds:
        if data.get("clean"):
            return data["round"]
    return min(rounds, key=lambda r: rank(*r))[0]["round"]


def kernel_path(run_dir: str, row: dict, round_index: int, n_rounds: int) -> str:
    stem = (
        f"level_{row['level']}_problem_{row['problem_id']}"
        f"_sample_{row['sample_id']}_kernel.py"
    )
    per_round = os.path.join(run_dir, "rounds", f"round_{round_index}", stem)
    return per_round if os.path.exists(per_round) else os.path.join(run_dir, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", default=",".join(DEFAULT_ROOTS))
    args = parser.parse_args()

    analyzer = SubmissionAnalyzer()
    seen: dict[str, str] = {}
    for root in args.roots.split(","):
        if not os.path.isdir(root):
            continue
        for run in sorted(os.listdir(root)):
            run_dir = os.path.join(root, run)
            if run not in seen and os.path.exists(os.path.join(run_dir, "lint_loop.jsonl")):
                seen[run] = run_dir

    considered = moved = 0
    directions: collections.Counter[str] = collections.Counter()

    for run, run_dir in sorted(seen.items()):
        for line in open(os.path.join(run_dir, "lint_loop.jsonl")):
            row = json.loads(line)
            if len(row["rounds"]) < 2:
                continue  # one round: nothing to choose between
            considered += 1

            rounds = []
            for data in row["rounds"]:
                path = kernel_path(run_dir, row, data["round"], row["n_rounds"])
                try:
                    source = open(path, encoding="utf-8", errors="replace").read()
                except OSError:
                    rounds.append((data, True))
                    continue
                report = analyzer.analyze(source, path)
                rounds.append((data, not report.findings))

            before, after = final(rounds, old_rank), final(rounds, new_rank)
            if before == after:
                continue
            moved += 1
            was = next(ok for d, ok in rounds if d["round"] == before)
            now = next(ok for d, ok in rounds if d["round"] == after)
            directions[f"{'loadable' if was else 'unloadable'} -> "
                       f"{'loadable' if now else 'unloadable'}"] += 1

    print(f"multi-round trajectories : {considered}")
    print(f"ship a different attempt : {moved}")
    for direction, count in directions.most_common():
        print(f"   {count:5d}  {direction}")
    bad = sum(n for d, n in directions.items() if d.endswith("-> unloadable"))
    print("\nOK: every move is towards a loadable attempt" if bad == 0
          else f"\nFAIL: {bad} trajectories moved to an UNLOADABLE attempt")


if __name__ == "__main__":
    main()
