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
from reranker.src.prm.rollout import prefixes as prefixes_mod, stage

# A directory, not a single file, despite the name: one part per job-B unit plus
# `_anchors.jsonl`, so a killed job resumes by skipping whatever already landed. Task 4
# and Task 6 glob os.path.join(out_dir, ORM_SCORES, "*.jsonl") to read every part.
ORM_SCORES = "orm_scores.jsonl"
ANCHORS_UNIT = "_anchors"


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


# Set once in main() from the checkpoint's model.safetensors; score_items reads it as a
# global so its own signature stays (items, scorer, encoder, kb_dir, batch_size).
_CKPT_SHA: str | None = None


def score_items(items, scorer, encoder, kb_dir, batch_size) -> Iterator[OrmScore]:
    """Dedup by code_sha1, encode with the ORM's own format, one forward pass per kernel."""
    items = list(items)
    first: dict[str, Item] = {}
    for it in items:
        first.setdefault(it.code_sha1, it)
    uniq = list(first.values())
    scores: dict[str, float] = {}
    for i in range(0, len(uniq), batch_size):
        chunk = uniq[i:i + batch_size]
        encoded = [encoder.encode(_ref_src(kb_dir, c.level, c.problem_id), c.code) for c in chunk]
        for c, e, s in zip(chunk, encoded, scorer(encoded)):
            scores[c.code_sha1] = s
            first[c.code_sha1] = dataclasses.replace(c, n_code_tokens=len(e))
    for it in items:
        yield OrmScore(it.kind, it.id, it.level, it.problem_id, it.code_sha1,
                       scores[it.code_sha1], _CKPT_SHA, first[it.code_sha1].n_code_tokens)


# --- the one part that needs a GPU --------------------------------------------------------


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
        # Left-padded: the seq-cls head scores the last non-pad token, so right-padding
        # would read a pad token there and score garbage (verified against the checkpoint).
        n = max(len(x) for x in encoded)
        ids = torch.tensor([[pad] * (n - len(x)) + list(x) for x in encoded]).cuda()
        att = torch.tensor([[0] * (n - len(x)) + [1] * len(x) for x in encoded]).cuda()
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=att).logits[:, 0].float().cpu().tolist()

    return scorer, SequenceEncoder(tok, cfg.orm_max_length, cfg.orm_reserve_ref_tokens)


# --- the resumable driver: whatever job B has produced -> orm_scores/*.jsonl --------------


def rollout_units(out_dir: str) -> list[str]:
    """Job-B units that have landed on disk, sorted -- the array-task index space here.

    Off disk, not off prefixes.jsonl: job B can still be producing units while this job
    runs, and a unit not yet generated has nothing to score.
    """
    parts = sorted(glob.glob(os.path.join(out_dir, stage.ROLLOUTS, "*.jsonl.gz")))
    return [os.path.basename(p)[: -len(".jsonl.gz")] for p in parts]


def unit_score_path(out_dir: str, unit: str) -> str:
    return os.path.join(out_dir, ORM_SCORES, f"{unit}.jsonl")


def _write_scores(path: str, scores) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    build.write_atomic(path, "".join(json.dumps(dataclasses.asdict(s)) + "\n" for s in scores))


def score_unit(conf, unit: str, scorer, encoder, kb: str) -> str:
    """This unit's rollouts -> its part under ORM_SCORES. Skips if already scored (resume)."""
    out_dir = _resolve(conf.out_dir)
    out_path = unit_score_path(out_dir, unit)
    if os.path.exists(out_path):
        return out_path
    prefix_rows = stage.read_prefixes(os.path.join(out_dir, prefixes_mod.PREFIXES))
    where = stage.homes(prefix_rows)
    rows = (dataclasses.asdict(r) for r in stage.read_rollouts(stage.unit_path(out_dir, unit)))
    items = iter_rollout_items_from_rows(rows, where)
    scores = score_items(items, scorer, encoder, kb, conf.orm_batch_size)
    _write_scores(out_path, scores)
    return out_path


def score_anchors(cfg, scorer, encoder, kb: str) -> str | None:
    """Every anchor -> its own part under ORM_SCORES. None if anchors are off or already done."""
    conf = cfg.prm_rollout
    out_path = unit_score_path(_resolve(conf.out_dir), ANCHORS_UNIT)
    if not conf.use_anchors or os.path.exists(out_path):
        return None
    scores = score_items(iter_anchor_items(cfg), scorer, encoder, kb, conf.orm_batch_size)
    _write_scores(out_path, scores)
    return out_path


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
    if conf.label_source != "imputed":
        raise ValueError(
            f"orm_score scores against an ORM only under label_source=imputed, got "
            f"{conf.label_source!r}"
        )

    global _CKPT_SHA
    _CKPT_SHA = build._sha1(os.path.join(conf.orm_checkpoint, "model.safetensors"))
    scorer, encoder = load_scorer(conf)
    kb = _kernelbench_dir(cfg)
    out_dir = _resolve(conf.out_dir)

    if anchors_only:
        path = score_anchors(cfg, scorer, encoder, kb)
        print(f"anchors -> {path or 'already scored'}")
        return

    units = rollout_units(out_dir)
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is None:
        for u in units:
            score_unit(conf, u, scorer, encoder, kb)
        return
    idx = int(task)
    if idx >= len(units):
        print(f"task {task}: only {len(units)} rollout units, nothing to do")
        return
    path = score_unit(conf, units[idx], scorer, encoder, kb)
    print(f"task {task}: unit {units[idx]} -> {path}")


if __name__ == "__main__":
    main()
