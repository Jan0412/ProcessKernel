"""Job B2: score every rollout and every anchor with the ORM. One code path for both."""
from __future__ import annotations

import dataclasses
import functools
import glob
import json
import os
import sys
from collections.abc import Iterator

from kernel_gen.core.text import extract_code_block
from reranker.src.config import _resolve, load_config
from reranker.src.encoding import SequenceEncoder
from reranker.src.prm import build
from reranker.src.prm.rollout import prefixes as prefixes_mod, rollout as rollout_mod, stage

# A directory, not a single file, despite the name: one part per job-B unit plus
# `_anchors.jsonl`, so a killed job resumes by skipping whatever already landed. Task 4
# and Task 6 glob os.path.join(out_dir, ORM_SCORES, "*.jsonl") to read every part.
ORM_SCORES = "orm_scores.jsonl"
ANCHORS_UNIT = "_anchors"
SCORE_META = ".meta"   # sidecar beside a score part -- mirrors rollout.py's UNIT_META


@dataclasses.dataclass(frozen=True)
class Item:
    kind: str            # rollout | anchor
    id: str
    level: int
    problem_id: int
    code: str
    code_sha1: str
    n_code_tokens: int    # filled at encode time; 0 on construction


@dataclasses.dataclass(frozen=True)
class OrmScore:
    kind: str
    id: str
    level: int
    problem_id: int
    code_sha1: str
    orm_score: float          # the raw logit -- never a sigmoid, never rescaled
    orm_checkpoint_sha: str
    n_code_tokens: int


def encoding_module_path() -> str:
    """Path to the PRM's own encoding.py -- the format this module must never use."""
    return prefixes_mod.__file__.replace("prefixes.py", "encoding.py")


def head_type(checkpoint_dir: str) -> str:
    """Read off reranker_head.json. Never guessed from config.json (PLAN_v3)."""
    with open(os.path.join(checkpoint_dir, "reranker_head.json")) as f:
        return json.load(f)["head_type"]


def anchor_item(row: dict) -> Item:
    """One v1 row -> an anchor Item. Anchors legitimately re-extract; rollouts never do."""
    # run__shard__rN__stem: the same stem recurs every round and across runs, so a bare
    # stem silently overwrites one round's score with another's (PLAN_v3 §5).
    ident = f"{row['run_name']}__{row['shard']}__r{row['round']}__{row['stem']}"
    code = extract_code_block(row["raw"])
    return Item("anchor", ident, row["level"], row["problem_id"], code,
                build._text_sha1(code), 0)


def iter_rollout_items_from_rows(rows, prefix_loc: dict) -> Iterator[Item]:
    """Rows -> Items, keyed to their prefix's (level, problem_id).

    Never re-extracts: job B stored `code` at generation time (KGEN-20).
    """
    for r in rows:
        level, pid = prefix_loc[r["prefix_id"]]
        yield Item("rollout", r["rollout_id"], level, pid, r["code"], r["code_sha1"], 0)


def iter_rollout_items(out_dir: str, prefixes) -> Iterator[Item]:
    """Every rollout unit currently on disk under `out_dir` -- job B may still be writing more.

    `prefixes` is the campaign's Prefix rows (as read by `stage.read_prefixes`), used only to
    resolve each rollout's (level, problem_id) via `stage.homes`.
    """
    where = stage.homes(prefixes)
    for part in sorted(glob.glob(os.path.join(out_dir, stage.ROLLOUTS, "*.jsonl.gz"))):
        rows = (dataclasses.asdict(r) for r in stage.read_rollouts(part))
        yield from iter_rollout_items_from_rows(rows, where)


def iter_anchor_items(cfg) -> Iterator[Item]:
    """Already-evaluated v1 rows this campaign anchors against -- re-extracted from `raw`.

    Restricted to the runs this campaign reads (`run_tags`) and to `anchor_rounds`: later
    lintloop rounds hurt kernel quality (see prm_rollout_l6_r0.yaml), so a campaign anchors
    only on the rounds it trusts.
    """
    conf = cfg.prm_rollout
    if not conf.use_anchors:
        return
    for part in sorted(glob.glob(_resolve(conf.parts_glob))):
        with open(part) as f:
            for line in f:
                row = json.loads(line)
                if row["run_name"] not in conf.run_tags:
                    continue
                if row["round"] not in conf.anchor_rounds:
                    continue
                yield anchor_item(row)


@functools.lru_cache(maxsize=None)
def _ref_src(kb_dir: str, level: int, pid: int) -> str:
    """The reference architecture's source. Raises rather than scoring against nothing."""
    matches = sorted(glob.glob(os.path.join(kb_dir, f"level{level}", f"{pid}_*.py")))
    if not matches:
        raise FileNotFoundError(
            f"no KernelBench/level{level}/{pid}_*.py under {kb_dir} -- an unresolvable "
            "reference is a corpus error, not a zero score"
        )
    with open(matches[0]) as f:
        return f.read()


def _kernelbench_dir(cfg) -> str:
    """Where the reference architectures live -- same resolution as build_dataset.py's kb_base.

    Underscored so it cannot be mistaken for `score_items`'s `kb_dir` *parameter* -- the two
    names would otherwise collide, one a function and the other a path string.
    """
    return os.path.join(_resolve(cfg.data.kernelbench_dir), "KernelBench")


def score_items(items, scorer, encoder, kb_dir, batch_size, ckpt_sha: str) -> Iterator[OrmScore]:
    """Dedup by (level, problem_id, code_sha1) -- code_sha1 ALONE is not a safe key:
    extract_code_block returns "" for any rollout with no code fence, so every empty body
    in the campaign would collide onto one sha and inherit whichever problem's reference
    happened to be scored first. stage.py's own staging dedup keys on this same triple for
    the identical reason. One forward pass per unique (level, problem_id, code_sha1);
    every item still gets its own output row.

    `ckpt_sha` is required, not defaulted: a caller that forgets it gets a TypeError at
    the call site, not a silent `orm_checkpoint_sha: null` on 1.1M written rows.
    """
    items = list(items)
    first: dict[tuple, Item] = {}
    for it in items:
        first.setdefault((it.level, it.problem_id, it.code_sha1), it)
    uniq = list(first.values())
    scores: dict[tuple, float] = {}
    for i in range(0, len(uniq), batch_size):
        chunk = uniq[i:i + batch_size]
        encoded = [encoder.encode(_ref_src(kb_dir, c.level, c.problem_id), c.code) for c in chunk]
        for c, e, s in zip(chunk, encoded, scorer(encoded)):
            key = (c.level, c.problem_id, c.code_sha1)
            scores[key] = s
            first[key] = dataclasses.replace(c, n_code_tokens=len(e))
    for it in items:
        key = (it.level, it.problem_id, it.code_sha1)
        yield OrmScore(it.kind, it.id, it.level, it.problem_id, it.code_sha1,
                       scores[key], ckpt_sha, first[key].n_code_tokens)


# --- the one part that needs a GPU --------------------------------------------------------


def _pad_right(encoded: list[list[int]], pad_id: int) -> tuple[list[list[int]], list[list[int]]]:
    """Right-padded ids and attention mask.

    Matches how the checkpoint was trained (dataset.py's `pad_sequences`, used by
    `RerankerCollator`) and is batch-invariant: measured on 96 real anchors under 3
    reshuffled batchings, right-padding scored bit-identically (max|delta| 0.0) while
    left-padding drifted up to 0.293 logit. The seq-cls head pools the last non-pad token
    either way (HF locates it off `input_ids != pad_token_id`), so this side is the one
    that has to match training, and training was right-padded.
    """
    n = max(len(x) for x in encoded)
    ids = [list(x) + [pad_id] * (n - len(x)) for x in encoded]
    att = [[1] * len(x) + [0] * (n - len(x)) for x in encoded]
    return ids, att


def load_scorer(cfg):
    """Loads exactly as reranker/src/eval.py::_load_model does -- head from JSON, never config.

    `cfg` here is the prm_rollout section (its orm_* fields), not the whole RerankerConfig.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt = cfg.orm_checkpoint
    assert head_type(ckpt) == "seq_cls", "only the seq_cls head is wired here"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=1, dtype=torch.bfloat16).cuda().eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    pad = tok.pad_token_id or 0

    def scorer(encoded):
        ids, att = _pad_right(encoded, pad)
        with torch.no_grad():
            return model(
                input_ids=torch.tensor(ids).cuda(), attention_mask=torch.tensor(att).cuda()
            ).logits[:, 0].float().cpu().tolist()

    return scorer, SequenceEncoder(tok, cfg.orm_max_length, cfg.orm_reserve_ref_tokens)


# --- the resumable driver: whatever job B has produced -> orm_scores/*.jsonl --------------


def rollout_units(out_dir: str) -> list[str]:
    """The campaign's full unit index space, off prefixes.jsonl -- stable no matter how many
    job-B units have actually landed at call time.

    Not a glob of disk: job B is still producing units while this job runs, a QoS cap runs
    a big array in waves, and a later wave's glob would see a longer, re-sorted list and
    index onto different units than an earlier wave did (duplicated GPU work, missed
    units). Mirrors rollout.py's own `unit_names()`, which draws its index space from the
    same file for the same reason.
    """
    prefix_rows = stage.read_prefixes(os.path.join(out_dir, prefixes_mod.PREFIXES))
    return sorted(rollout_mod.units(prefix_rows))


def _unit_ready(out_dir: str, unit: str) -> bool:
    """Has job B actually landed this unit's part yet? Checked by exact final path, so a
    mid-write `.tmp.<host>.<pid>` file (stage.open_part) is never mistaken for a landed one.
    """
    return os.path.isfile(stage.unit_path(out_dir, unit))


def unit_score_path(out_dir: str, unit: str) -> str:
    return os.path.join(out_dir, ORM_SCORES, f"{unit}.jsonl")


def _is_fresh(out_path: str, source_sha1: str) -> bool:
    """Has this part already been scored from exactly this source? False on any doubt:
    missing part, missing sidecar, zero rows, or a source_sha1 that no longer matches.
    """
    meta_path = out_path + SCORE_META
    if not os.path.isfile(out_path) or not os.path.isfile(meta_path):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("rows", 0) > 0 and meta.get("source_sha1") == source_sha1


def _write_scores(path: str, scores, source_sha1: str) -> bool:
    """The part plus its resume sidecar. Writes nothing and returns False if `scores` is
    empty: an empty part would satisfy a bare `os.path.exists` forever, sealing resume on
    a misconfigured run_tags/anchor_rounds with Task 4 fitting a curve on nothing.
    """
    rows = list(scores)
    if not rows:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    build.write_atomic(path, "".join(json.dumps(dataclasses.asdict(s)) + "\n" for s in rows))
    build.write_atomic(
        path + SCORE_META, json.dumps({"rows": len(rows), "source_sha1": source_sha1})
    )
    return True


def score_unit(conf, unit: str, scorer, encoder, kb: str, ckpt_sha: str) -> str | None:
    """This unit's rollouts -> its part under ORM_SCORES.

    Resumed off the job-B unit's own `.meta` sidecar content, not bare existence of the
    score part: a regenerated unit samples fresh at T=0.6 (rollout.py) and gets a new
    sidecar, so its old score would otherwise reference rollout_ids that no longer exist.
    Returns None (writes nothing) if the unit has no rollouts to score.
    """
    out_dir = _resolve(conf.out_dir)
    out_path = unit_score_path(out_dir, unit)
    unit_meta = stage.unit_path(out_dir, unit) + rollout_mod.UNIT_META
    if not os.path.isfile(unit_meta):
        # Degrades rather than raising, mirroring rollout.py's own _metas(): a landed part
        # with no sidecar (job B always writes one, so this means something else copied a
        # part in without it) must not take down the whole array task over one unit.
        print(f"{unit}: no {rollout_mod.UNIT_META} sidecar, skipping (cannot verify freshness)")
        return None
    source_sha1 = build._sha1(unit_meta)
    if _is_fresh(out_path, source_sha1):
        return out_path
    prefix_rows = stage.read_prefixes(os.path.join(out_dir, prefixes_mod.PREFIXES))
    where = stage.homes(prefix_rows)
    rows = (dataclasses.asdict(r) for r in stage.read_rollouts(stage.unit_path(out_dir, unit)))
    items = iter_rollout_items_from_rows(rows, where)
    scores = score_items(items, scorer, encoder, kb, conf.orm_batch_size, ckpt_sha)
    return out_path if _write_scores(out_path, scores, source_sha1) else None


def score_anchors(cfg, scorer, encoder, kb: str, ckpt_sha: str) -> str | None:
    """Every anchor -> its own part under ORM_SCORES.

    Resumed off a hash of (run_tags, anchor_rounds), not bare existence: changing either
    knob changes which v1 rows are anchors, so the old part must not be silently reused.
    Returns None if anchors are off, or if the current selection scores zero rows.
    """
    conf = cfg.prm_rollout
    if not conf.use_anchors:
        return None
    out_path = unit_score_path(_resolve(conf.out_dir), ANCHORS_UNIT)
    source_sha1 = build._text_sha1(json.dumps(
        {"run_tags": conf.run_tags, "anchor_rounds": sorted(conf.anchor_rounds)}, sort_keys=True
    ))
    if _is_fresh(out_path, source_sha1):
        return out_path
    scores = score_items(iter_anchor_items(cfg), scorer, encoder, kb, conf.orm_batch_size, ckpt_sha)
    return out_path if _write_scores(out_path, scores, source_sha1) else None


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    anchors_only = "--anchors" in argv
    if anchors_only:
        argv.remove("--anchors")
    if "--units" in argv:
        argv.remove("--units")
        out_dir = _resolve(load_config(argv).prm_rollout.out_dir)
        print(len(rollout_units(out_dir)))
        return

    cfg = load_config(argv)
    conf = cfg.prm_rollout
    conf.validate()
    # The gate is the checkpoint, NOT label_source. A `measured` campaign scores its anchors
    # too: level 1 is the only corpus whose labels are evaluated, so it is the only place the
    # imputation curve can be checked against measured truth (prm_rollout_l1.yaml carries the
    # orm_* keys for exactly that sweep). Without this, calibrate on level 1 dies with "job B2
    # has not scored the anchors yet" and the method has no validation at all.
    if not conf.orm_checkpoint:
        raise ValueError(
            "orm_score needs prm_rollout.orm_checkpoint -- there is no model to score with"
        )
    out_dir = _resolve(conf.out_dir)

    # Readiness is decided BEFORE the model ever loads. Job B is still landing units, the
    # QoS caps this account at 16 GPUs across every job, and a task with nothing to do must
    # not spend one of them loading the ORM only to immediately exit.
    unit, units = None, None
    if not anchors_only:
        units = rollout_units(out_dir)
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task is not None:
            idx = int(task)
            if idx >= len(units):
                print(f"task {task}: only {len(units)} campaign units, nothing to do")
                return
            unit = units[idx]
            if not _unit_ready(out_dir, unit):
                print(f"task {task}: unit {unit} has no rollout part yet, nothing to do")
                return
        elif not any(_unit_ready(out_dir, u) for u in units):
            print("no rollout units are ready yet, nothing to do")
            return

    ckpt_sha = build._sha1(os.path.join(conf.orm_checkpoint, "model.safetensors"))
    scorer, encoder = load_scorer(conf)
    kb = _kernelbench_dir(cfg)

    if anchors_only:
        path = score_anchors(cfg, scorer, encoder, kb, ckpt_sha)
        print(f"anchors -> {path or 'empty, not written'}")
        return

    if unit is not None:
        path = score_unit(conf, unit, scorer, encoder, kb, ckpt_sha)
        print(f"task {os.environ['SLURM_ARRAY_TASK_ID']}: unit {unit} -> "
              f"{path or 'empty, not written'}")
        return

    for u in units:
        if _unit_ready(out_dir, u):
            score_unit(conf, u, scorer, encoder, kb, ckpt_sha)


if __name__ == "__main__":
    main()
