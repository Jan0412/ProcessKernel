"""Pre-flight a staged level dir before anything downstream trusts it.

    uv run --no-sync python -m scripts.check_level_dir KernelBench/level6

A staged dir is the single source of truth for the pipeline: lintloop.sh prompts from it
via --ref-dir, KernelBench's eval_from_generations.py scores against it, and the linter
reads its shapes. A
reference that is broken here is not a bad sample, it is a mislabelled problem -- every
sample paired with it is wrong, and nothing downstream can tell. So check the invariants
once, here, rather than discovering them 60 GPU-days into a run.

Checks, in the order a failure would bite:

  1. ids are unique          -- eval keys problems into a dict, so a duplicate id does not
                                error: one file silently wins and generation may have
                                prompted from the other.
  2. generation == eval      -- the file this repo's loader returns for an id is the same
                                file KernelBench's loader returns.
  3. every file parses       -- a SyntaxError reaches eval as a compile failure charged
                                to the model.
  4. the evaluator's ABI     -- eval.py does Model(*get_init_inputs()) then
                                forward(*get_inputs()); all three names must be defined.
                                Checked statically: executing ~17k arbitrary sources is
                                exactly what the sandboxed smoke test is for.
  5. the linter can see it   -- shape inference must yield input shapes, or every
                                shape-dependent F2 check silently loses its byte
                                estimates and the feedback degrades to vagueness.
  6. provenance              -- manifest.json covers the files on disk, and
                                conversion_stats.json exists (its absence means the dir
                                predates the OOM guards).

Exits non-zero if any hard check fails, so it can gate a swap in a script.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path

from kernel_gen.core.sources import index_level_dir

REQUIRED = ("Model", "get_inputs", "get_init_inputs")


def eval_index(ref_dir: Path) -> dict[int, str]:
    """The id -> filename map KernelBench's LocalKernelBenchDataset builds for this dir.

    Copied, not imported: that repo is a separate checkout with its own venv.
    """
    index: dict[int, str] = {}
    for name in sorted(os.listdir(ref_dir)):
        if not name.endswith(".py"):
            continue
        try:
            index[int(name.split("_")[0])] = name
        except (ValueError, IndexError):
            continue
    return index


def _defines(tree: ast.Module) -> set[str]:
    """Top-level names bound by a def or an assignment (``Model = GCN`` counts)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def check(ref_dir: str) -> int:
    from checker.shapes import shapes_from_source

    path = Path(ref_dir)
    print(f"checking {path}\n")

    try:
        index = index_level_dir(str(path))
    except ValueError as exc:
        print(f"FAIL  ids are not unique / dir unusable: {exc}")
        return 1
    print(f"  ids unique                 : OK ({len(index)} references)")

    eval_side = eval_index(path)
    disagree = [pid for pid, name in index.items() if eval_side.get(pid) != name]
    unseen = set(eval_side) - set(index)
    if disagree:
        print(f"FAIL  generation/eval disagree on {len(disagree)} ids, e.g. {disagree[:5]}")
        return 1
    if unseen:
        print(f"FAIL  eval would load {len(unseen)} ids generation never sees, "
              f"e.g. {sorted(unseen)[:5]}")
        return 1
    print(f"  generation == eval ref     : OK (all {len(index)} resolve identically)")

    unparsable, missing_abi, no_shapes = [], [], []
    for pid, name in sorted(index.items()):
        src = (path / name).read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            unparsable.append((name, f"line {exc.lineno}: {exc.msg}"))
            continue
        absent = [n for n in REQUIRED if n not in _defines(tree)]
        if absent:
            missing_abi.append((name, ",".join(absent)))
            continue
        try:
            if not shapes_from_source(src):
                no_shapes.append(name)
        except Exception:  # noqa: BLE001 - inference is best-effort by design
            no_shapes.append(name)

    hard = 0
    for label, bad in (("parse", unparsable), ("evaluator ABI", missing_abi)):
        if bad:
            hard = 1
            print(f"FAIL  {len(bad)} references fail the {label} check:")
            for name, why in bad[:5]:
                print(f"          {name}: {why}")
        else:
            print(f"  {label:<26} : OK")

    # Soft: inference legitimately gives up on some shapes (dynamic get_inputs). It
    # degrades F2 feedback for those rows rather than invalidating them.
    pct = 100 * len(no_shapes) / max(len(index), 1)
    verdict = "OK" if pct < 10 else "WARN"
    print(f"  linter shape inference     : {verdict} ({len(no_shapes)} of {len(index)}"
          f" = {pct:.1f}% yield no shapes)")

    manifest_path, stats_path = path / "manifest.json", path / "conversion_stats.json"
    if not stats_path.exists():
        hard = 1
        print("FAIL  conversion_stats.json missing -- this dir predates the OOM guards")
    else:
        stats = json.loads(stats_path.read_text())
        device, budget = stats.get("smoke_device"), stats.get("smoke_mem_gb") or 0
        # The budget is only ever applied when the smoke test ran ON CUDA -- see
        # kernelbook_convert._run_smoke_inproc, which gates set_per_process_memory_fraction
        # behind `on_cuda`. A CPU-staged dir still RECORDS smoke_mem_gb, and guard_oom
        # still counts rows, because that counter string-matches the rejection reason and
        # a host-RAM failure also says "out of memory". So the stats of a CPU run look
        # identical to a guarded one while nothing was enforced. Refuse it: this is the
        # same failure mode as staging on a card smaller than the budget, where the
        # fraction clamps to 1.0 and binds nothing.
        if not stats.get("smoke_test"):
            hard = 1
            print("FAIL  staged without --smoke-test: no reference was ever executed")
        elif device != "cuda" and not str(device).startswith("cuda"):
            hard = 1
            print(f"FAIL  smoke_device={device!r}: the memory budget is CUDA-only, so "
                  f"smoke_mem_gb={budget} was NOT enforced. Restage on an H100.")
        elif budget <= 0:
            hard = 1
            print("FAIL  smoke_mem_gb=0 disables the memory guard entirely")
        else:
            print(f"  guards                     : OK (mem caught {stats.get('guard_oom', 0)}"
                  f" rows, numel caught {stats.get('guard_blowup', 0)}; "
                  f"device={device}, smoke_mem_gb={budget})")

    if not manifest_path.exists():
        hard = 1
        print("FAIL  manifest.json missing")
    else:
        manifest = json.loads(manifest_path.read_text())
        uncovered = set(index) - {int(k) for k in manifest}
        if uncovered:
            hard = 1
            print(f"FAIL  manifest covers {len(manifest)} of {len(index)} files; "
                  f"{len(uncovered)} have no provenance (concurrent jobs clobbered it?)")
        else:
            scales = Counter(v.get("scale") for v in manifest.values())
            print(f"  manifest provenance        : OK ({len(manifest)} entries)")
            print(f"  scales                     : {dict(scales)}")

    print("\n" + ("FAILED" if hard else "PASSED"))
    return hard


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    if not os.path.isdir(sys.argv[1]):
        raise SystemExit(f"not a directory: {sys.argv[1]}")
    raise SystemExit(check(sys.argv[1]))
