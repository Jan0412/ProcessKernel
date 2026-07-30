"""Compare two staged level dirs -- what a restage actually changed.

    uv run --no-sync python -m scripts.diff_level_dirs KernelBench/level6 KernelBench/level6_new

A staged dir is the pipeline's source of truth, so replacing one is not a refresh, it is a
change of benchmark. Three things can move, and they have very different consequences:

  * **the row set** -- an id present in one dir and not the other. Every result keyed to
    that id in an existing run loses its reference. This is the change that invalidates
    prior numbers, so it is reported first and in full.
  * **the scale of a kept row** -- same problem, different input size. Results stay
    joinable but are no longer comparable: a kernel timed at 48**4 and one timed at 24**4
    are measuring different work.
  * **nothing** -- byte-identical. The useful majority, and worth counting so the two
    above can be read as a proportion.

Input shapes are read with the linter's own inference, so "scale changed" means what the
prompt and the F2 byte estimates would actually see, not what the manifest claims.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from kernel_gen.core.sources import index_level_dir


def _shapes(src: str):
    from checker.lint.shapes import shapes_from_source

    try:
        return tuple(tuple(s) for s, _ in shapes_from_source(src))
    except Exception:  # noqa: BLE001 - inference is best-effort
        return None


def _scales(path: Path) -> dict[int, str]:
    manifest = path / "manifest.json"
    if not manifest.exists():
        return {}
    return {int(k): v.get("scale", "?") for k, v in json.loads(manifest.read_text()).items()}


def diff(old_dir: str, new_dir: str) -> int:
    old_path, new_path = Path(old_dir), Path(new_dir)
    old, new = index_level_dir(old_dir), index_level_dir(new_dir)

    print(f"old: {old_dir}  ({len(old)} references)")
    print(f"new: {new_dir}  ({len(new)} references)\n")

    added, removed, kept = set(new) - set(old), set(old) - set(new), set(old) & set(new)

    print("== row set ==")
    print(f"  kept    : {len(kept)}")
    print(f"  added   : {len(added)}" + (f"  e.g. {sorted(added)[:8]}" if added else ""))
    print(f"  removed : {len(removed)}" + (f"  e.g. {sorted(removed)[:8]}" if removed else ""))
    if removed:
        print(f"  !! {len(removed)} ids vanish. Any eval_results.json keyed to them "
              f"loses its reference and those samples become unscoreable.")

    identical, shape_changed, other_changed = 0, [], 0
    for pid in sorted(kept):
        a = (old_path / old[pid]).read_text(encoding="utf-8", errors="replace")
        b = (new_path / new[pid]).read_text(encoding="utf-8", errors="replace")
        if a == b:
            identical += 1
            continue
        sa, sb = _shapes(a), _shapes(b)
        if sa != sb:
            shape_changed.append((pid, new[pid], sa, sb))
        else:
            other_changed += 1

    print("\n== kept rows ==")
    print(f"  byte-identical  : {identical}")
    print(f"  shapes CHANGED  : {len(shape_changed)}")
    print(f"  changed, same shapes : {other_changed}")

    if shape_changed:
        print("\n  the rows whose problem size moved (first 12):")
        for pid, name, sa, sb in shape_changed[:12]:
            print(f"    {pid:>6} {name[:34]:<34} {sa} -> {sb}")
        print("  !! kernels generated against the old size are not comparable to the new one.")

    old_scales, new_scales = _scales(old_path), _scales(new_path)
    if old_scales and new_scales:
        moved = Counter(
            (old_scales[p], new_scales[p])
            for p in kept
            if p in old_scales and p in new_scales and old_scales[p] != new_scales[p]
        )
        print(f"\n== manifest scale transitions ({len(old_scales)} vs {len(new_scales)} "
              f"entries; only ids present in BOTH manifests are comparable) ==")
        if moved:
            for (a, b), n in moved.most_common(10):
                print(f"    {a:<14} -> {b:<14} {n}")
        else:
            print("    none")

    for label, path in (("old", old_path), ("new", new_path)):
        stats = path / "conversion_stats.json"
        if stats.exists():
            s = json.loads(stats.read_text())
            print(f"\n  {label} guards: mem caught {s.get('guard_oom')}, "
                  f"numel caught {s.get('guard_blowup')}, device={s.get('smoke_device')}, "
                  f"budget={s.get('smoke_mem_gb')}GiB")
        else:
            print(f"\n  {label} guards: conversion_stats.json MISSING (predates the guards)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raise SystemExit(diff(sys.argv[1], sys.argv[2]))
