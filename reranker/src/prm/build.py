"""Corpus -> labeled rows: one part file per unit, plus a manifest (PLAN §6, §9).

    python -m reranker.src.prm.build --config reranker/configs/prm_config.yaml
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from multiprocessing import Pool

from reranker.src.config import PROJECT_ROOT, PRMConfig, RerankerConfig, _resolve, load_config
from reranker.src.data.labels import load_baseline_times
from reranker.src.prm import corpus, targets
from reranker.src.prm.chunks import cut_points

PARTS, MANIFEST, META = "parts", "manifest.json", ".meta"

TRUNCATED = "truncated"
NO_FINISH_REASON = "no_finish_reason"
NO_BASELINE = "no_baseline"
NO_RUNTIME = "no_runtime"
TOKENIZE_FAILED = "tokenize_failed"
TOO_LONG = "too_long"
NO_CUTS = "no_cuts"
# Applied in this order so the counts are interpretable: `truncated` first because it is
# the only drop whose *label* is invalid rather than whose text is unusable (§6).
REASONS = (
    TRUNCATED,
    NO_FINISH_REASON,
    corpus.NO_EVAL_ENTRY,
    NO_BASELINE,
    NO_RUNTIME,
    TOKENIZE_FAILED,
    TOO_LONG,
    NO_CUTS,
)
ROWS, CUTS = "rows", "cuts"

# Knobs that do NOT change what a row contains -- which units to walk, how to run, how to
# split. Stated as the exclusion so a knob added later counts as row-affecting by default:
# then forgetting to classify it refuses a resume rather than silently permitting a mixed one.
_NOT_ROW_KNOBS = frozenset(
    {"run_dirs", "rounds", "max_shards", "out_dir", "num_workers", "split_ratios", "split_seed"}
)
ROW_KNOBS = tuple(
    sorted(f.name for f in dataclasses.fields(PRMConfig) if f.name not in _NOT_ROW_KNOBS)
)


@dataclass(frozen=True)
class Built:
    """What one worker hands back: its part, and the record published beside it."""

    part: str
    # {"counts", "dropped"} -- `dropped` maps a reason to its stems, so a retokenize can
    # redo just those attempts (§7). Also written to the part's `.meta`, from this dict.
    record: dict


def part_name(unit: corpus.Unit) -> str:
    """Both source runs contain a ``shard_00``, so the run name has to be in the file name."""
    return f"{unit.run_name}__{unit.shard}__round{unit.round}.jsonl"


def token_counter(name: str) -> Callable[[str], int]:
    """One tokenizer per worker -- loading a Qwen tokenizer per unit dominates the small ones."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])


def build_unit(
    unit: corpus.Unit,
    prm: PRMConfig,
    *,
    parts_dir: str,
    baselines: dict,
    count: Callable[[str], int],
) -> Built:
    """Write one unit's part, ``.idx`` and ``.meta``, counting every attempt that did not make it."""
    dropped: dict[str, list[str]] = {r: [] for r in REASONS}
    counts = Counter()
    path = os.path.join(parts_dir, part_name(unit))
    offsets: list[int] = []
    pos = 0

    # Written to .tmp and renamed, so an interrupted build leaves no half-row that
    # run_build's "the part exists, skip it" would then treat as finished.
    with open(path + ".tmp", "wb") as f:
        for labeled in corpus.iter_labeled(unit, counts):
            row = _row(labeled, prm, baselines, count, dropped)
            if row is None:
                continue
            # ensure_ascii spelled out because it is load-bearing, not a default: it is how
            # attempts.jsonl is written, and a lone surrogate read back from there cannot
            # encode as UTF-8 -- which aborts the whole pool rather than dropping one row.
            blob = (json.dumps(row, ensure_ascii=True) + "\n").encode()
            f.write(blob)
            offsets.append(pos)
            pos += len(blob)
            counts[CUTS] += len(row["cuts"])

    fired = {r: s for r, s in dropped.items() if s}
    counts[ROWS] = len(offsets)
    counts.update({r: len(s) for r, s in fired.items()})
    record = {"counts": dict(counts), "dropped": fired}

    write_atomic(path + ".idx", "".join(f"{o}\n" for o in offsets))
    # Beside the part, not gathered in the parent: a build killed at its walltime has
    # already published parts, and their counters have to survive with them.
    write_atomic(path + ".meta", json.dumps(record))
    # Both land first: the part's existence is what marks the unit done.
    os.replace(path + ".tmp", path)
    return Built(path, record)


def _row(
    labeled: corpus.Labeled,
    prm: PRMConfig,
    baselines: dict,
    count: Callable[[str], int],
    dropped: dict[str, list[str]],
) -> dict | None:
    """One output row, or ``None`` with the attempt's stem recorded under its drop reason."""

    def drop(reason: str) -> None:
        dropped[reason].append(labeled.stem)
        return None

    if labeled.truncation == corpus.TRUNCATED:
        return drop(TRUNCATED)
    if labeled.truncation != corpus.OK:
        return drop(NO_FINISH_REASON)

    baseline = baselines.get(labeled.level, {}).get(labeled.problem_id)
    target = targets.target_for(
        compiled=labeled.compiled,
        correct=labeled.correct,
        ours=labeled.runtimes,
        baseline=baseline,
        mode=prm.label_mode,
        stat=prm.speedup_stat,
        lo=prm.speedup_lo,
        hi=prm.speedup_hi,
        quant=prm.speed_quant,
    )
    if target is None:
        # Only a correct kernel gets here. Naming the two sides apart keeps the ledger
        # honest: a kernel the harness never timed is not an absent baseline (PRM-4).
        gradeable = targets.usable((baseline or {}).get(prm.speedup_stat))
        return drop(NO_RUNTIME if gradeable else NO_BASELINE)

    cuts = cut_points(
        labeled.raw,
        prose_lines=prm.prose_lines_per_chunk,
        code_steps=prm.code_steps_per_chunk,
    )
    if cuts is None:
        return drop(TOKENIZE_FAILED)

    n_prompt, n_raw = count(labeled.prompt), count(labeled.raw)
    if n_prompt + n_raw > prm.max_length:
        return drop(TOO_LONG)

    # A filter over cuts, not over samples (§6): survivors keep the depth cut_points gave
    # them, so `cut_index` is not 0..n-1 and has to be stored. A sample it empties is still
    # dropped -- a row with no cut has nothing to score. Measured 0 of 2,175 up to 0.3.
    kept = [c for c in cuts if c.char >= prm.min_frac * len(labeled.raw)]
    if not kept:
        return drop(NO_CUTS)

    return {
        "run_name": labeled.unit.run_name,
        "shard": labeled.unit.shard,
        "round": labeled.unit.round,
        "level": labeled.level,
        "problem_id": labeled.problem_id,
        "sample_id": labeled.sample_id,
        "stem": labeled.stem,
        "prompt": labeled.prompt,
        "raw": labeled.raw,
        "cuts": [c.char for c in kept],
        "cut_kinds": [c.kind for c in kept],
        "cut_index": [c.index for c in kept],
        "n_prompt_tokens": n_prompt,
        "n_raw_tokens": n_raw,
        "compiled": labeled.compiled,
        "correct": labeled.correct,
        # Kept so label_mode, speedup_stat, speedup_lo/hi and speed_quant stay re-derivable:
        # regrading is a pass over the parts, not a rebuild (§7).
        "speedup": target.speedup,
        "speedup_min": target.speedup_min,
        "target": target.value,
        "label": target.label,
    }


_STATE: dict = {}


def _init_worker(prm: PRMConfig, parts_dir: str, baselines: dict) -> None:
    _STATE.update(
        prm=prm, parts_dir=parts_dir, baselines=baselines, count=token_counter(prm.tokenizer)
    )


def _build_one(unit: corpus.Unit) -> Built:
    return build_unit(
        unit,
        _STATE["prm"],
        parts_dir=_STATE["parts_dir"],
        baselines=_STATE["baselines"],
        count=_STATE["count"],
    )


def _map(todo: list[corpus.Unit], prm: PRMConfig, parts_dir: str, baselines: dict) -> list[Built]:
    if not todo:
        return []  # a full rerun: no tokenizer to load, nothing to fork for
    args = (prm, parts_dir, baselines)
    # One unit is not worth a fork.
    if prm.num_workers == 1 or len(todo) == 1:
        _init_worker(*args)
        return [_build_one(u) for u in todo]
    with Pool(min(prm.num_workers, len(todo)), initializer=_init_worker, initargs=args) as pool:
        return list(pool.imap_unordered(_build_one, todo))


def run_build(cfg: RerankerConfig) -> str:
    """Build every unit that has no part yet; returns the manifest path."""
    # First, before out_dir exists or the pool forks: an unvalidated knob would otherwise
    # first raise inside a worker with part files already on disk (PLAN §13 step 5).
    prm = cfg.prm
    prm.validate()

    found, skipped = corpus.units(prm.run_dirs, prm.rounds, max_shards=prm.max_shards)
    out_dir = _resolve(prm.out_dir)
    parts_dir = os.path.join(out_dir, PARTS)
    todo = [u for u in found if not os.path.isfile(os.path.join(parts_dir, part_name(u)))]

    baseline = _resolve(prm.baseline_timing_json)
    baseline_sha1 = _sha1(baseline)
    if len(todo) < len(found):
        _check_resume(_load_manifest(out_dir), prm, baseline_sha1, out_dir, len(found) - len(todo))
    os.makedirs(parts_dir, exist_ok=True)

    def stamp(built: int) -> dict:
        return _manifest(prm, baseline, baseline_sha1, _unit_records(parts_dir), skipped, built)

    path = os.path.join(out_dir, MANIFEST)
    # Stamped before the pool as well as after it: a build killed at its walltime leaves
    # parts behind, and the next run's _check_resume needs a config to compare them to.
    write_atomic(path, json.dumps(stamp(0), indent=2))
    _map(todo, prm, parts_dir, load_baseline_times(baseline))
    manifest = stamp(len(todo))
    write_atomic(path, json.dumps(manifest, indent=2))

    counts = manifest["counts"]
    ledger = {r: counts[r] for r in REASONS if r in counts}
    print(f"Parts: {len(todo)} built, {manifest['units_reused']} reused -> {parts_dir}")
    print(
        f"  rows {counts.get(ROWS, 0)}, cuts {counts.get(CUTS, 0)}, "
        f"attempts {manifest['attempts']}"
    )
    print(f"  dropped {ledger}")
    return path


def _manifest(prm, baseline, baseline_sha1, units, skipped, built) -> dict:
    """The dataset as it stands on disk -- not what this invocation happened to do."""
    total: Counter = Counter()
    for name in sorted(units):
        total.update(units[name]["counts"])
    # Fixed key order, and stems in part order, so two identical builds agree byte for byte.
    counts = {k: total[k] for k in (ROWS, CUTS, *REASONS) if k in total}
    dropped = {
        r: [s for n in sorted(units) for s in units[n]["dropped"].get(r, [])] for r in REASONS
    }
    # Without `dirty` the sha is not provenance: the prm package can be uncommitted while a
    # build runs, and HEAD then names a commit that does not contain the builder.
    dirty = _git("status", "--porcelain")
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "config": dataclasses.asdict(prm),
        "baseline_timing_json": baseline,
        "baseline_sha1": baseline_sha1,
        "parts": len(units),
        "units_built": built,
        "units_reused": len(units) - built,
        "skipped_units": [list(s) for s in skipped],
        "attempts": counts.get(ROWS, 0) + sum(counts.get(r, 0) for r in REASONS),
        "counts": counts,
        "dropped_stems": {r: v for r, v in dropped.items() if v},
        "units": {n: units[n] for n in sorted(units)},
    }


def _unit_records(parts_dir: str) -> dict[str, dict]:
    """Every part's counters, read from the ``.meta`` its own worker published with it."""
    out = {}
    for meta in sorted(glob.glob(os.path.join(parts_dir, f"*.jsonl{META}"))):
        part = meta[: -len(META)]
        if os.path.isfile(part):  # a sidecar whose part never landed describes nothing
            with open(meta) as f:
                out[os.path.basename(part)] = json.load(f)
    return out


def _load_manifest(out_dir: str) -> dict:
    path = os.path.join(out_dir, MANIFEST)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _check_resume(was: dict, prm: PRMConfig, baseline_sha1: str, out_dir: str, n: int) -> None:
    """A part is reused on its name alone, so the knobs behind it must not have moved."""
    if not was:
        # Every build stamps a manifest before writing a part, so parts without one were
        # not made here -- copied in, or left by a version that stamped only at the end.
        raise ValueError(
            f"{out_dir} holds {n} parts but no {MANIFEST} to check them against; what "
            "built them cannot be verified -- build into a new out_dir, or delete this one"
        )
    old = was.get("config", {})
    moved = {k: (old.get(k), getattr(prm, k)) for k in ROW_KNOBS if old.get(k) != getattr(prm, k)}
    # The path can stay put while the file behind it is re-measured.
    if was.get("baseline_sha1") != baseline_sha1:
        moved["baseline_sha1"] = (was.get("baseline_sha1"), baseline_sha1)
    if moved:
        raise ValueError(
            f"{out_dir} holds parts built with different knobs {moved}; reusing them would "
            "mix two labelings in one dataset -- build into a new out_dir, or delete this one"
        )


def write_atomic(path: str, text: str) -> None:
    """Rename into place, so a reader never sees a partly written file. Shared with splits."""
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    """``None`` off a checkout, or if git hangs -- provenance is not worth aborting a build for."""
    try:
        done = subprocess.run(
            ["git", "-C", PROJECT_ROOT, *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def main(argv: Iterable[str] | None = None) -> None:
    run_build(load_config(None if argv is None else list(argv)))


if __name__ == "__main__":
    main()
