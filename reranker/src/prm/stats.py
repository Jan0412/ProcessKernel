"""Diagnostics over a built PRM dataset: §12's acceptance table, measured and checked.

    python -m reranker.src.prm.stats --config reranker/configs/prm_config.yaml

Read-only; exits non-zero when a check fails.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from kernel_gen.core.text import fence_spans
from reranker.src.config import RerankerConfig, _resolve, load_config
from reranker.src.prm import corpus
from reranker.src.prm.build import (
    CUTS,
    MANIFEST,
    PARTS,
    REASONS,
    ROW_KNOBS,
    ROWS,
    part_name,
    write_atomic,
)
from reranker.src.prm.splits import SPLITS

STATS = "stats.json"
# §3's claim under test is "0% of sibling pairs share >200 chars"; 256 is enough to refute it.
SIBLING_CAP = 256
SIBLING_OVER = 200
# The corpus holds 20 (§12, measured over all 285,149 attempts) and the rule there is
# "hundreds or thousands => the detector is wrong". §13 step 6 says "single digits", which
# contradicts that measurement; 100 is the threshold that catches a broken detector without
# failing on the corpus as it stands.
TRUNCATED_MAX = 100
# §12's stop-and-report gate: more than this and the length cap is distorting the corpus.
MAX_SHIFT_POINTS = 0.5


def unterminated(raw: str) -> bool:
    """A fence opened and never closed -- 99% of gpt-oss, and never a truncation signal."""
    # A closed block ends at its closing marker, which is strictly inside the text; only an
    # unterminated one is handed back running to the end (kernel_gen KGEN-2).
    spans = fence_spans(raw)
    return bool(spans) and spans[-1][1] == len(raw)


@dataclass
class Scan:
    """One pass over the parts. ``raw`` is never retained past the row it came from."""

    rows: int = 0
    correct: int = 0
    compiled: int = 0
    unterminated: int = 0
    cuts: list[int] = field(default_factory=list)
    kinds: Counter = field(default_factory=Counter)
    targets: Counter = field(default_factory=Counter)
    by_run: dict = field(default_factory=lambda: defaultdict(Counter))
    by_round: dict = field(default_factory=lambda: defaultdict(Counter))
    problems: set = field(default_factory=set)
    siblings: dict = field(default_factory=lambda: defaultdict(list))
    off_scale: list = field(default_factory=list)


def scan_parts(parts_dir: str) -> Scan:
    scan = Scan()
    parts = sorted(glob.glob(os.path.join(parts_dir, "*.jsonl")))
    if not parts:
        raise FileNotFoundError(f"no parts under {parts_dir}; run reranker.src.prm.build first")
    for part in parts:
        with open(part) as f:
            for line in f:
                _tally(json.loads(line), scan)
    return scan


def _tally(row: dict, s: Scan) -> None:
    label, target, correct = row["label"], row["target"], bool(row["correct"])
    s.rows += 1
    s.correct += correct
    s.compiled += bool(row["compiled"])
    s.targets[round(target, 2)] += 1
    # The graded scale is 0.0, or [0.5, 1.0] with nothing between (§5). A row off it means
    # target_for and this histogram disagree about what a label means.
    if not ((0.5 <= target <= 1.0) if label else target == 0.0):
        s.off_scale.append(row["stem"])
    s.cuts.append(len(row["cuts"]))
    s.kinds.update(row["cut_kinds"])
    open_fence = unterminated(row["raw"])
    s.unterminated += open_fence
    for group in (s.by_run[row["run_name"]], s.by_round[row["round"]]):
        group["rows"] += 1
        group["correct"] += correct
        group["unterminated"] += open_fence
    s.problems.add((row["level"], row["problem_id"]))
    # Siblings differ only in sample_id, and one shard holds every sample of its problems.
    s.siblings[(row["run_name"], row["level"], row["problem_id"], row["round"])].append(
        row["raw"][:SIBLING_CAP]
    )


def _lcp(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            return n
        n += 1
    return n


def sibling_prefixes(siblings: dict) -> dict:
    """§11's "no sibling pairs exist" fact, re-measured rather than left as folklore."""
    lens = [
        _lcp(texts[i], texts[j])
        for texts in siblings.values()
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    ]
    if not lens:
        return {"pairs": 0}
    over = sum(1 for n in lens if n > SIBLING_OVER)
    return {
        "pairs": len(lens),
        "median": statistics.median(lens),
        "max": max(lens),  # capped at SIBLING_CAP, which is all the >200 claim needs
        "over_200": over,
        "over_200_pct": _pct(over, len(lens)),
    }


def verdict_totals(units: Iterable[corpus.Unit], built: set[str]) -> tuple[int, int]:
    """Attempts carrying a verdict, and how many are correct.

    The parts hold only kept rows and the manifest keeps only dropped *stems*, so the
    base-rate gate has to read the eval side back to get a denominator over all attempts.
    """
    total = correct = 0
    for unit in units:
        if part_name(unit) not in built:
            continue  # a unit whose part never landed contributed no rows to compare against
        for entry in corpus.verdicts(unit.eval_path).values():
            total += 1
            correct += bool(entry.get("correctness", False))
    return total, correct


def _unit_key(part: str) -> tuple[str, int]:
    """``{run}__{shard}__round{R}.jsonl`` back into ``(run, round)``.

    From the right, so a run name containing ``__`` still parses; only a shard name
    containing one would break it, and they are all ``shard_NN``.
    """
    run, _shard, rnd = part.rsplit("__", 2)
    return run, int(rnd.removeprefix("round").removesuffix(".jsonl"))


def ledger(manifest: dict) -> dict:
    """Kept and dropped per run and per round, from the ``.meta`` records the manifest merged."""
    by_run: dict = defaultdict(Counter)
    by_round: dict = defaultdict(Counter)
    for part, rec in manifest.get("units", {}).items():
        run, rnd = _unit_key(part)
        by_run[run].update(rec["counts"])
        by_round[rnd].update(rec["counts"])
    return {
        "by_run": {k: dict(v) for k, v in sorted(by_run.items())},
        "by_round": {str(k): dict(v) for k, v in sorted(by_round.items())},
    }


def _pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


def _read_json(path: str, default=None):
    if not os.path.isfile(path):
        return default
    with open(path) as f:
        return json.load(f)


def report(cfg: RerankerConfig) -> dict:
    """Measure the built dataset and check it against §12; returns the report it writes."""
    prm = cfg.prm
    prm.validate()

    out_dir = _resolve(prm.out_dir)
    manifest = _read_json(os.path.join(out_dir, MANIFEST))
    if not manifest:
        raise FileNotFoundError(f"no {MANIFEST} in {out_dir}; run reranker.src.prm.build first")

    scan = scan_parts(os.path.join(out_dir, PARTS))
    counts = manifest["counts"]
    attempts = manifest["attempts"]
    built = set(manifest.get("units", {}))
    found, _ = corpus.units(prm.run_dirs, prm.rounds, max_shards=prm.max_shards)
    graded, graded_correct = verdict_totals(found, built)

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    # Finish reasons come out of the ledger, not a second walk of attempts.jsonl: every
    # attempt that is neither `truncated` nor `no_finish_reason` finished on "stop".
    length = counts.get("truncated", 0)
    unknown = counts.get("no_finish_reason", 0)
    finish = {"stop": attempts - length - unknown, "length": length, "unknown": unknown}

    band = sum(n for t, n in scan.targets.items() if 0 < t < 0.5)
    spikes = sorted(t for t in scan.targets if t > 0)
    cuts = sorted(scan.cuts)
    kept_pct, all_pct = _pct(scan.correct, scan.rows), _pct(graded_correct, graded)
    shift = kept_pct - all_pct
    dropped_n = graded - scan.rows

    splits = _read_json(os.path.join(out_dir, SPLITS), default={})
    keys = sorted(f"{lvl}:{pid}" for lvl, pid in scan.problems)
    missing = [k for k in keys if k not in splits]

    out = {
        "out_dir": out_dir,
        "created": manifest.get("created"),
        "git_sha": manifest.get("git_sha"),
        "git_dirty": manifest.get("git_dirty"),
        "baseline_sha1": manifest.get("baseline_sha1"),
        "config": {
            k: getattr(prm, k)
            for k in ("rounds", "max_length", "tokenizer", "label_mode", "speedup_stat",
                      "prose_lines_per_chunk", "code_steps_per_chunk", "min_frac")
        },
        "totals": {
            "parts": manifest.get("parts"),
            "attempts": attempts,
            "rows": scan.rows,
            "keep_pct": _pct(scan.rows, attempts),
            "cuts": sum(scan.cuts),
            "compiled_pct": _pct(scan.compiled, scan.rows),
        },
        "drops": {
            r: {"n": counts.get(r, 0), "pct": _pct(counts.get(r, 0), attempts)}
            for r in REASONS
            if counts.get(r, 0)
        },
        "ledger": ledger(manifest),
        "finish_reasons": finish,
        "targets": {
            "histogram": {str(t): n for t, n in sorted(scan.targets.items())},
            "zero_pct": _pct(scan.targets.get(0.0, 0), scan.rows),
            "empty_band": band,
            "spikes": len(spikes),
        },
        "cuts": {
            "mean": statistics.fmean(cuts) if cuts else 0.0,
            "median": statistics.median(cuts) if cuts else 0,
            "p5": cuts[int(0.05 * len(cuts))] if cuts else 0,
            "p95": cuts[int(0.95 * len(cuts))] if cuts else 0,
            "code": scan.kinds.get("code", 0),
            "prose": scan.kinds.get("prose", 0),
        },
        "bias": {
            "attempts_with_verdict": graded,
            "all_correct_pct": all_pct,
            "kept_correct_pct": kept_pct,
            "dropped_correct_pct": _pct(graded_correct - scan.correct, dropped_n),
            "shift_points": shift,
        },
        "fences": {
            "unterminated_pct": _pct(scan.unterminated, scan.rows),
            "by_run": {
                run: _pct(c["unterminated"], c["rows"]) for run, c in sorted(scan.by_run.items())
            },
        },
        "siblings": sibling_prefixes(scan.siblings),
        "splits": {
            "problems": len(scan.problems),
            "assigned": dict(Counter(splits.values())) if splits else {},
            "missing": missing[:20],
            "n_missing": len(missing),
        },
    }

    # Without this the header prints the knobs of the config passed in, which need not be
    # the ones the parts were built with -- a report that quietly describes the wrong dataset.
    was = manifest.get("config", {})
    moved = {k: (was.get(k), getattr(prm, k)) for k in ROW_KNOBS if was.get(k) != getattr(prm, k)}
    check(
        "report config matches the build's",
        not moved,
        f"row knobs moved since the build: {moved}" if moved else "row knobs agree",
    )
    check(
        "rows on disk match the manifest",
        scan.rows == counts.get(ROWS, 0),
        f"scanned {scan.rows}, manifest {counts.get(ROWS, 0)}",
    )
    check(
        "cuts on disk match the manifest",
        sum(scan.cuts) == counts.get(CUTS, 0),
        f"scanned {sum(scan.cuts)}, manifest {counts.get(CUTS, 0)}",
    )
    check(
        "provenance recorded",
        bool(manifest.get("git_sha")) and bool(manifest.get("baseline_sha1")),
        f"git_sha={manifest.get('git_sha')} baseline_sha1={manifest.get('baseline_sha1')}",
    )
    check(
        "one part per unit the config resolves",
        manifest.get("parts") == len(found),
        f"{manifest.get('parts')} parts, {len(found)} units found",
    )
    check(
        "every target on the graded scale",
        not scan.off_scale,
        f"{len(scan.off_scale)} off-scale rows" + (f", e.g. {scan.off_scale[0]}" if scan.off_scale else ""),
    )
    check(
        "nothing in the empty (0, 0.5) band",
        band == 0,
        f"{band} rows",
    )
    check(
        f"truncated under {TRUNCATED_MAX}",
        length <= TRUNCATED_MAX,
        f"{length} -- hundreds would mean the detector is reading the fence (20 in v5)",
    )
    check(
        f"base-rate shift under {MAX_SHIFT_POINTS} points",
        abs(shift) < MAX_SHIFT_POINTS,
        f"{all_pct:.3f}% over all attempts -> {kept_pct:.3f}% kept ({shift:+.3f} pt)",
    )
    unjoined = counts.get(corpus.NO_EVAL_ENTRY, 0)
    check(
        "every attempt reconciles with a verdict",
        graded == attempts - unjoined,
        f"{graded} verdicts, {attempts - unjoined} attempts joined",
    )
    # The check above is an identity and holds at any `no_eval_entry`. This one is the
    # precondition: an attempt with no verdict is one evaluation had not reached, and PLAN
    # §13 forbids building against a corpus whose evaluation is still in flight.
    check(
        "no attempt is missing its verdict",
        unjoined == 0,
        f"{unjoined} attempts have no eval entry"
        + (" -- evaluation was still in flight; this build is incomplete" if unjoined else ""),
    )
    check(
        "every problem has a split",
        bool(splits) and not missing,
        f"{len(missing)} unassigned of {len(scan.problems)}"
        + ("" if splits else f" -- no {SPLITS} yet"),
    )

    out["checks"] = checks
    write_atomic(os.path.join(out_dir, STATS), json.dumps(out, indent=2, default=str))
    return out


def render(out: dict) -> str:
    """The report as text; the same numbers ``stats.json`` carries."""
    t, b, c = out["totals"], out["bias"], out["cuts"]
    lines = [
        "=" * 72,
        f"PRM dataset  {out['out_dir']}",
        f"  built {out['created']}  git {str(out['git_sha'])[:12]}"
        f"{' DIRTY' if out['git_dirty'] else ''}  baseline {str(out['baseline_sha1'])[:12]}",
        f"  {out['config']}",
        "=" * 72,
        f"rows {t['rows']:,} of {t['attempts']:,} attempts ({t['keep_pct']:.2f}% kept)"
        f"   parts {t['parts']}   cuts {t['cuts']:,}",
        "",
        "drops",
    ]
    lines += [
        f"  {r:<16} {d['n']:>8,}  {d['pct']:6.3f}%" for r, d in out["drops"].items()
    ] or ["  (none)"]

    f = out["finish_reasons"]
    lines += [
        "",
        f"finish reasons   stop {f['stop']:,}   length {f['length']:,}   unknown {f['unknown']:,}",
        f"unterminated fence, KEPT rows   {out['fences']['unterminated_pct']:.1f}%"
        "   (a formatting artifact -- measures something else entirely)",
    ]
    lines += [f"    {run:<52} {pct:5.1f}%" for run, pct in out["fences"]["by_run"].items()]

    lines += [
        "",
        f"cuts per completion   mean {c['mean']:.1f}   median {c['median']}"
        f"   p5 {c['p5']}   p95 {c['p95']}   ({c['code']:,} code + {c['prose']:,} prose)",
        "",
        f"targets   {out['targets']['zero_pct']:.1f}% at 0.0"
        f"   {out['targets']['spikes']} spikes above"
        f"   {out['targets']['empty_band']} in the empty (0, 0.5) band",
    ]
    lines += [
        f"    {k:>5}  {n:>8,}" for k, n in out["targets"]["histogram"].items()
    ]

    lines += [
        "",
        f"bias   all attempts {b['all_correct_pct']:.3f}% correct"
        f" -> kept {b['kept_correct_pct']:.3f}%   shift {b['shift_points']:+.3f} pt",
        f"       dropped rows {b['dropped_correct_pct']:.3f}% correct"
        " (a rate gap here is expected; the shift is the gate)",
    ]

    s, sp = out["siblings"], out["splits"]
    if s.get("pairs"):
        lines.append(
            f"siblings   {s['pairs']:,} pairs   median shared prefix {s['median']} chars"
            f"   >{SIBLING_OVER}: {s['over_200']} ({s['over_200_pct']:.2f}%)"
        )
    lines.append(f"splits   {sp['problems']:,} problems   {sp['assigned']}")

    lines += ["", "checks", "-" * 72]
    for ck in out["checks"]:
        lines.append(f"  [{'PASS' if ck['ok'] else 'FAIL'}] {ck['check']:<44} {ck['detail']}")
    failed = [ck for ck in out["checks"] if not ck["ok"]]
    lines += ["-" * 72, f"{len(out['checks']) - len(failed)}/{len(out['checks'])} checks passed"]
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> None:
    out = report(load_config(None if argv is None else list(argv)))
    print(render(out))
    print(f"Written: {os.path.join(out['out_dir'], STATS)}")
    if any(not ck["ok"] for ck in out["checks"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
