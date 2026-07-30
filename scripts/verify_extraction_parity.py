"""Hash what `extract_code_block` picks out of every captured completion.

Phase H switches the extractor's `_parses` probe from `ast.parse` to `compile`, which is
strictly stricter: a fenced block that parses but cannot be compiled stops outranking one
that can. The claim is that this moves a handful of attempts and regresses none, so this
records `sha256(extract_code_block(raw))` per attempt and the diff before/after is the
whole evidence.

    python scripts/verify_extraction_parity.py -o baseline.jsonl
    ...change _parses...
    python scripts/verify_extraction_parity.py -o new.jsonl && diff baseline.jsonl new.jsonl

Reads the traced runs' `attempts.jsonl`, which is the only place a raw completion survives
-- the run dirs keep the extracted kernel, not the text it came from.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from kernel_gen.core.text import extract_code_block  # noqa: E402

DEFAULT_TRACES = "/path/to/workdir/KernelBench/runs"


def records(traces_root: str):
    pattern = os.path.join(traces_root, "*", "traces", "round_*", "attempts.jsonl")
    for path in sorted(glob.glob(pattern)):
        run = path.split("/runs/")[-1].split("/traces/")[0]
        for line in open(path):
            record = json.loads(line)
            yield run, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", default=DEFAULT_TRACES)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    lines = []
    for run, record in records(args.traces_root):
        code = extract_code_block(record["raw"])
        lines.append(
            json.dumps(
                {
                    "run": run,
                    "round": record["round"],
                    "stem": record["stem"],
                    "sha256": hashlib.sha256(code.encode()).hexdigest(),
                    "len": len(code),
                }
            )
        )

    lines.sort()
    with open(args.out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"{len(lines)} attempts -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
