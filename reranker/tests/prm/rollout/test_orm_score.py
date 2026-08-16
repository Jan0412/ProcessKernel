import dataclasses
import json
import os
import pathlib

from reranker.src.config import PRMRolloutConfig, RerankerConfig
from reranker.src.encoding import SequenceEncoder
from reranker.src.prm import build
from reranker.src.prm.rollout import orm_score
from reranker.src.prm.rollout import prefixes as prefixes_mod
from reranker.src.prm.rollout import rollout as rollout_mod
from reranker.src.prm.rollout import stage


class _StubTokenizer:
    """One id per character -- deterministic, no real vocab needed."""

    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 1000 for c in text]


def _enc() -> SequenceEncoder:
    return SequenceEncoder(_StubTokenizer(), max_length=6144, reserve_ref_tokens=1024)


def _kb(tmp_path) -> str:
    """A KernelBench root with one problem: level6/37_x.py.

    Idempotent (exist_ok=True): several tests call this twice against the same tmp_path
    (e.g. once before and once after fixing a config), and it must not collide with itself.
    """
    level_dir = tmp_path / "KernelBench" / "level6"
    level_dir.mkdir(parents=True, exist_ok=True)
    (level_dir / "37_x.py").write_text("class Model:\n    pass\n")
    return str(tmp_path / "KernelBench")


def _kb2(tmp_path) -> str:
    """A KernelBench root with two problems, whose refs have different (countable)
    lengths -- lets a test tell "scored against problem 37's ref" apart from
    "scored against problem 12's ref" even when the two items' code is identical."""
    level_dir = tmp_path / "KernelBench" / "level6"
    level_dir.mkdir(parents=True, exist_ok=True)
    (level_dir / "37_x.py").write_text("A" * 5)
    (level_dir / "12_y.py").write_text("B" * 50)
    return str(tmp_path / "KernelBench")


def _prefix(prefix_id="p1", *, run_name="a_run", shard="shard_00", round=0, level=6,
            problem_id=37, **over):
    fields = dict(
        prefix_id=prefix_id, source="cut", run_name=run_name, run_tag="ar", shard=shard,
        round=round, level=level, problem_id=problem_id, sample_id=0,
        stem=f"level_{level}_problem_{problem_id}_sample_0_kernel",
        cut_char=10, cut_index=5, cut_kind="code", n_cuts_total=20, rel_depth=0.25,
        list_key=f"ar:{level}:{problem_id}:{round}:5", split="train", selection="random",
        selection_score=None, K=2, min_rollouts=1,
    )
    fields.update(over)
    return prefixes_mod.Prefix(**fields)


def _roll(rid, prefix_id, code="import torch\n", sha="a" * 40, **over):
    fields = dict(
        rollout_id=rid, prefix_id=prefix_id, j=0, continuation="...", code=code,
        code_sha1=sha, n_prefix_tokens=1, n_gen_tokens=1,
        finish_reason={"plan": None, "code": "stop"}, truncation="ok",
    )
    fields.update(over)
    return rollout_mod.Rollout(**fields)


def _write_prefix_file(out_dir, rows) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, prefixes_mod.PREFIXES)
    with open(path, "w") as f:
        for p in rows:
            f.write(json.dumps(dataclasses.asdict(p)) + "\n")


def _rollout_conf(out_dir, **over) -> PRMRolloutConfig:
    conf = PRMRolloutConfig(out_dir=out_dir, run_tags={"a_run": "ar"}, orm_batch_size=8)
    for k, v in over.items():
        setattr(conf, k, v)
    return conf


class StubScorer:
    """Returns len(code) as the logit, and counts forward passes."""
    def __init__(self): self.calls = 0
    def __call__(self, encoded):
        self.calls += len(encoded)
        return [float(len(e)) for e in encoded]


def test_rollouts_use_the_stored_code_and_never_re_extract(tmp_path):
    # KGEN-20: re-extraction drifts on 2.78% of completions, so the stored code is the truth.
    row = {"rollout_id": "r1", "prefix_id": "p1", "code": "STORED",
           "code_sha1": "aaa", "truncation": "ok"}
    items = list(orm_score.iter_rollout_items_from_rows([row], {"p1": (6, 37)}))
    assert [i.code for i in items] == ["STORED"]


def test_anchors_re_extract_from_raw():
    row = {"level": 6, "problem_id": 37, "sample_id": 1, "round": 0, "shard": "shard_00",
           "run_name": "run", "stem": "s", "raw": "prose\n```python\nCODE\n```\n"}
    item = orm_score.anchor_item(row)
    assert item.code == "CODE"
    assert item.kind == "anchor"


def test_anchor_id_carries_run_shard_and_round():
    # A bare stem collides across rounds and across runs (PLAN_v3 §5).
    base = {"level": 6, "problem_id": 37, "sample_id": 1, "shard": "shard_00",
            "run_name": "run", "stem": "s", "raw": "```python\nX\n```"}
    a = orm_score.anchor_item({**base, "round": 0}).id
    b = orm_score.anchor_item({**base, "round": 1}).id
    assert a != b and "r0" in a and "r1" in b and "shard_00" in a


def test_dedup_key_is_level_problem_and_code_sha1_not_sha1_alone(tmp_path):
    """Critical 2: code_sha1 alone is not a safe dedup key. extract_code_block("") collides
    every unfenced rollout onto one sha, so two DIFFERENT problems sharing that sha must
    each be scored against their OWN reference, not whichever came first."""
    kb = _kb2(tmp_path)
    enc = _enc()
    expect_37 = len(enc.encode("A" * 5, "SAME"))
    expect_12 = len(enc.encode("B" * 50, "SAME"))
    assert expect_37 != expect_12   # the fixture must actually distinguish the two refs

    items = [
        orm_score.Item("rollout", "r0", 6, 37, "SAME", "sha1", 4),
        orm_score.Item("rollout", "r1", 6, 37, "SAME", "sha1", 4),   # same problem+sha as r0
        orm_score.Item("rollout", "r2", 6, 12, "SAME", "sha1", 4),   # same sha, OTHER problem
    ]
    scorer = StubScorer()
    out = {o.id: o for o in orm_score.score_items(items, scorer, enc, kb, 8, "ckpt-x")}
    assert scorer.calls == 2                          # one forward pass per (level, pid, sha)
    assert out["r0"].orm_score == out["r1"].orm_score   # r0/r1 really did share one group
    assert out["r0"].n_code_tokens == expect_37          # encoded against problem 37's ref
    assert out["r2"].n_code_tokens == expect_12          # encoded against problem 12's OWN ref


def test_score_is_the_raw_logit_not_a_probability(tmp_path):
    items = [orm_score.Item("rollout", "r1", 6, 37, "x" * 9, "sha", 3)]
    out = list(orm_score.score_items(items, lambda e: [-2.5], _enc(), _kb(tmp_path), 8, "ckpt-x"))
    assert out[0].orm_score == -2.5     # a sigmoid would be 0.076
    assert out[0].orm_checkpoint_sha == "ckpt-x"   # Important 3: never silently null


def test_scorer_pads_on_the_right():
    # Critical 1: measured on GPU (96 real anchors, 3 batch orderings) -- left-padding
    # drifted up to 0.293 logit under a reshuffled batch, right-padding was bit-exact and
    # matches training (dataset.py's pad_sequences / RerankerCollator).
    ids, att = orm_score._pad_right([[1, 2, 3], [4, 5]], pad_id=0)
    assert ids == [[1, 2, 3], [4, 5, 0]]
    assert att == [[1, 1, 1], [1, 1, 0]]


def test_head_type_comes_from_reranker_head_json(tmp_path):
    (tmp_path / "reranker_head.json").write_text('{"head_type": "seq_cls", "yes_id": -1, "no_id": -1}')
    assert orm_score.head_type(str(tmp_path)) == "seq_cls"


def test_prm_encoding_is_not_used_here():
    # The PRM scores a half-written generation; the ORM scores a finished kernel. Crossing
    # them is silent, so both directions are asserted. Checked as an instantiation
    # ("SequenceEncoder(") rather than a bare substring: the PRM's own encoding.py *names*
    # SequenceEncoder in its docstring, to explain why it is deliberately not used there.
    src = pathlib.Path(orm_score.__file__).read_text()
    assert "SequenceEncoder(" in src
    prm = pathlib.Path(orm_score.encoding_module_path()).read_text()
    assert "SequenceEncoder(" not in prm


def test_iter_rollout_items_reads_units_from_disk(tmp_path):
    """iter_rollout_items globs whatever job B has written -- no re-extraction, no prefixes.jsonl."""
    p = prefixes_mod.Prefix(
        prefix_id="p1", source="cut", run_name="a_run", run_tag="ar", shard="shard_00",
        round=0, level=6, problem_id=37, sample_id=0, stem="s0", cut_char=10, cut_index=5,
        cut_kind="code", n_cuts_total=20, rel_depth=0.25, list_key="ar:6:37:0:5",
        split="train", selection="random", selection_score=None, K=2, min_rollouts=1,
    )
    r = rollout_mod.Rollout(
        rollout_id="p1__j00", prefix_id="p1", j=0, continuation="...", code="STORED",
        code_sha1="abc", n_prefix_tokens=1, n_gen_tokens=1,
        finish_reason={"plan": None, "code": "stop"}, truncation="ok",
    )
    stage.write_rollouts([r], os.path.join(str(tmp_path), stage.ROLLOUTS, "unit1.jsonl.gz"))

    items = list(orm_score.iter_rollout_items(str(tmp_path), [p]))
    assert [(i.code, i.level, i.problem_id) for i in items] == [("STORED", 6, 37)]


def test_iter_anchor_items_filters_by_run_and_round(tmp_path):
    """Anchors are restricted to run_tags and anchor_rounds -- other rows are not anchors."""
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    rows = [
        {"run_name": "run", "shard": "shard_00", "round": 0, "level": 6, "problem_id": 37,
         "sample_id": 0, "stem": "s0", "raw": "```python\nA\n```"},
        {"run_name": "run", "shard": "shard_00", "round": 1, "level": 6, "problem_id": 37,
         "sample_id": 1, "stem": "s1", "raw": "```python\nB\n```"},
        {"run_name": "other", "shard": "shard_00", "round": 0, "level": 6, "problem_id": 37,
         "sample_id": 2, "stem": "s2", "raw": "```python\nC\n```"},
    ]
    (parts_dir / "part.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    cfg = RerankerConfig()
    cfg.prm_rollout.parts_glob = str(parts_dir / "*.jsonl")
    cfg.prm_rollout.run_tags = {"run": "tag"}
    cfg.prm_rollout.anchor_rounds = [0]
    cfg.prm_rollout.use_anchors = True

    items = list(orm_score.iter_anchor_items(cfg))
    assert [i.code for i in items] == ["A"]


# --- the driver: rollout_units, resume, and the tmp-file/empty-part edge cases -----------


def test_tmp_rollout_file_is_not_discovered_as_landed(tmp_path):
    """A unit mid-write (stage.open_part's `.tmp.<host>.<pid>` file, before the atomic
    rename) must not be treated as ready to score."""
    out_dir = str(tmp_path)
    p = _prefix(prefix_id="p1", run_name="a", shard="shard_00", round=0)
    _write_prefix_file(out_dir, [p])
    unit = rollout_mod.unit_name(p)
    real = stage.unit_path(out_dir, unit)
    os.makedirs(os.path.dirname(real), exist_ok=True)
    open(real + ".tmp.somehost.123", "w").close()   # mid-write; final path not there yet

    assert orm_score.rollout_units(out_dir) == [unit]     # index space is from prefixes.jsonl
    assert orm_score._unit_ready(out_dir, unit) is False    # but not ready to score


def test_array_index_space_is_stable_as_job_b_lands_more_units(tmp_path):
    """Finding 6: the unit at a given index must not change as job B lands more units --
    a live glob of disk would re-sort/lengthen wave to wave under a QoS-capped array and
    misindex SLURM_ARRAY_TASK_ID onto a different unit each wave."""
    out_dir = str(tmp_path)
    p1 = _prefix(prefix_id="p1", run_name="a", shard="shard_00", round=0, problem_id=1)
    p2 = _prefix(prefix_id="p2", run_name="b", shard="shard_00", round=0, problem_id=2)
    _write_prefix_file(out_dir, [p1, p2])

    before = orm_score.rollout_units(out_dir)      # neither unit has landed yet
    assert len(before) == 2

    u1 = rollout_mod.unit_name(p1)
    stage.write_rollouts([_roll("p1__j00", "p1")], stage.unit_path(out_dir, u1))

    after = orm_score.rollout_units(out_dir)
    assert after == before   # same index space -- unaffected by what has landed since


def _ids_in(path) -> set:
    with open(path) as f:
        return {json.loads(line)["id"] for line in f}


def test_resume_is_a_noop_and_regeneration_forces_rescore(tmp_path):
    """Finding 4: a second score_unit call does no new scoring; a job-B unit regenerated
    (new .meta sidecar, T=0.6 makes new kernels -- rollout.py) IS re-scored, not silently
    reused -- and the part's CONTENTS actually change, not just the call count."""
    out_dir = str(tmp_path)
    p = _prefix(prefix_id="p1", run_name="a", shard="shard_00", round=0, problem_id=37)
    _write_prefix_file(out_dir, [p])
    unit = rollout_mod.unit_name(p)
    part = stage.unit_path(out_dir, unit)
    stage.write_rollouts([_roll("p1__j00", "p1", code="V1", sha="v1sha")], part)
    build.write_atomic(part + rollout_mod.UNIT_META, json.dumps({"seconds": 1}))

    conf = _rollout_conf(out_dir)
    scorer = StubScorer()
    kb = _kb(tmp_path)

    path1 = orm_score.score_unit(conf, unit, scorer, _enc(), kb, "ckpt")
    assert scorer.calls == 1
    assert _ids_in(path1) == {"p1__j00"}
    path2 = orm_score.score_unit(conf, unit, scorer, _enc(), kb, "ckpt")
    assert path2 == path1 and scorer.calls == 1   # no-op: resumed, not rescored

    # Job B regenerates the unit: a DIFFERENT set of rollouts (new sample at j01 too), new
    # sidecar. Not just new code under the old id -- the part's rollout ids themselves change.
    stage.write_rollouts(
        [_roll("p1__j00", "p1", code="V2", sha="v2sha"),
         _roll("p1__j01", "p1", code="V3", sha="v3sha")],
        part,
    )
    build.write_atomic(part + rollout_mod.UNIT_META, json.dumps({"seconds": 2}))

    path3 = orm_score.score_unit(conf, unit, scorer, _enc(), kb, "ckpt")
    assert scorer.calls == 3   # rescored (1 from before + 2 newly-encoded items), not skipped
    assert _ids_in(path3) == {"p1__j00", "p1__j01"}   # the part's contents actually replaced


def test_unit_ready_is_true_once_the_final_part_has_landed(tmp_path):
    """The True branch of _unit_ready, not just the False one: an _unit_ready that always
    returns False would pass every other test here while silently scoring nothing."""
    out_dir = str(tmp_path)
    p = _prefix(prefix_id="p1", run_name="a", shard="shard_00", round=0)
    unit = rollout_mod.unit_name(p)
    assert orm_score._unit_ready(out_dir, unit) is False   # nothing written yet

    stage.write_rollouts([_roll("p1__j00", "p1")], stage.unit_path(out_dir, unit))
    assert orm_score._unit_ready(out_dir, unit) is True


def test_missing_unit_meta_sidecar_is_skipped_not_fatal(tmp_path):
    """Minor 3: a landed part with no .meta sidecar must not take down the whole array
    task -- rollout.py's own _metas() tolerates exactly this the same way."""
    out_dir = str(tmp_path)
    p = _prefix(prefix_id="p1", run_name="a", shard="shard_00", round=0)
    _write_prefix_file(out_dir, [p])
    unit = rollout_mod.unit_name(p)
    stage.write_rollouts([_roll("p1__j00", "p1")], stage.unit_path(out_dir, unit))
    # no .meta sidecar written

    conf = _rollout_conf(out_dir)
    result = orm_score.score_unit(conf, unit, StubScorer(), _enc(), _kb(tmp_path), "ckpt")
    assert result is None


def test_empty_anchors_part_does_not_seal_resume(tmp_path):
    """Finding 5: a misconfigured run_tags/anchor_rounds scores zero anchors; that must
    not look 'done' once the config is fixed and real anchors exist."""
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    (parts_dir / "part.jsonl").write_text(json.dumps(
        {"run_name": "run", "shard": "shard_00", "round": 0, "level": 6, "problem_id": 37,
         "sample_id": 0, "stem": "s0", "raw": "```python\nA\n```"}) + "\n")

    cfg = RerankerConfig()
    cfg.prm_rollout.out_dir = str(tmp_path)
    cfg.prm_rollout.parts_glob = str(parts_dir / "*.jsonl")
    cfg.prm_rollout.run_tags = {"WRONG_RUN": "tag"}   # matches nothing
    cfg.prm_rollout.anchor_rounds = [0]
    cfg.prm_rollout.use_anchors = True

    scorer = StubScorer()
    result = orm_score.score_anchors(cfg, scorer, _enc(), _kb(tmp_path), "ckpt")
    assert result is None
    assert not os.path.exists(orm_score.unit_score_path(str(tmp_path), orm_score.ANCHORS_UNIT))

    cfg.prm_rollout.run_tags = {"run": "tag"}   # fixed
    result2 = orm_score.score_anchors(cfg, scorer, _enc(), _kb(tmp_path), "ckpt")
    assert result2 is not None
    assert scorer.calls == 1


def test_score_anchors_reruns_when_run_tags_change(tmp_path):
    """Finding 4 (anchors half): two different anchor selections must not share one cached
    part just because both are non-empty."""
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    rows = [
        {"run_name": "run1", "shard": "shard_00", "round": 0, "level": 6, "problem_id": 37,
         "sample_id": 0, "stem": "s0", "raw": "```python\nA\n```"},
        {"run_name": "run2", "shard": "shard_00", "round": 0, "level": 6, "problem_id": 37,
         "sample_id": 1, "stem": "s1", "raw": "```python\nB\n```"},
    ]
    (parts_dir / "part.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    cfg = RerankerConfig()
    cfg.prm_rollout.out_dir = str(tmp_path)
    cfg.prm_rollout.parts_glob = str(parts_dir / "*.jsonl")
    cfg.prm_rollout.anchor_rounds = [0]
    cfg.prm_rollout.use_anchors = True

    cfg.prm_rollout.run_tags = {"run1": "tag"}
    scorer = StubScorer()
    orm_score.score_anchors(cfg, scorer, _enc(), _kb(tmp_path), "ckpt")
    assert scorer.calls == 1

    cfg.prm_rollout.run_tags = {"run2": "tag"}   # different selection, same part path
    orm_score.score_anchors(cfg, scorer, _enc(), _kb(tmp_path), "ckpt")
    assert scorer.calls == 2   # rescored, not skipped as "already done"


def test_score_anchors_is_a_noop_when_config_is_unchanged(tmp_path):
    """The other half of finding 4/review-round-2: an UNCHANGED run_tags/anchor_rounds
    must be a no-op on the second call, not just a changed one forcing a rescore."""
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    (parts_dir / "part.jsonl").write_text(json.dumps(
        {"run_name": "run", "shard": "shard_00", "round": 0, "level": 6, "problem_id": 37,
         "sample_id": 0, "stem": "s0", "raw": "```python\nA\n```"}) + "\n")

    cfg = RerankerConfig()
    cfg.prm_rollout.out_dir = str(tmp_path)
    cfg.prm_rollout.parts_glob = str(parts_dir / "*.jsonl")
    cfg.prm_rollout.run_tags = {"run": "tag"}
    cfg.prm_rollout.anchor_rounds = [0]
    cfg.prm_rollout.use_anchors = True

    scorer = StubScorer()
    path1 = orm_score.score_anchors(cfg, scorer, _enc(), _kb(tmp_path), "ckpt")
    assert scorer.calls == 1

    path2 = orm_score.score_anchors(cfg, scorer, _enc(), _kb(tmp_path), "ckpt")
    assert path2 == path1
    assert scorer.calls == 1   # unchanged config -- no-op, not rescored


# --- main(): the array-task GPU-load ordering, end to end --------------------------------


def test_main_selects_unit_writes_its_part_and_skips_gpu_when_not_ready(tmp_path, monkeypatch):
    """Important 1/2: main() must pick the right unit for SLURM_ARRAY_TASK_ID, actually
    write that unit's part -- and, for a task whose unit has not landed, must return
    without ever calling load_scorer (no GPU load for nothing to do)."""
    out_dir = str(tmp_path)
    p1 = _prefix(prefix_id="p1", run_name="a", shard="shard_00", round=0, problem_id=37)
    p2 = _prefix(prefix_id="p2", run_name="b", shard="shard_00", round=0, problem_id=37)
    _write_prefix_file(out_dir, [p1, p2])
    units_sorted = sorted([rollout_mod.unit_name(p1), rollout_mod.unit_name(p2)])
    landed_prefix = p1 if rollout_mod.unit_name(p1) == units_sorted[0] else p2
    landed, missing = units_sorted[0], units_sorted[1]

    part = stage.unit_path(out_dir, landed)
    stage.write_rollouts([_roll(f"{landed_prefix.prefix_id}__j00", landed_prefix.prefix_id)], part)
    build.write_atomic(part + rollout_mod.UNIT_META, json.dumps({"seconds": 1}))

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.safetensors").write_bytes(b"fake-checkpoint-bytes")

    conf = _rollout_conf(
        out_dir, run_tags={"a": "ar", "b": "br"}, orm_checkpoint=str(ckpt_dir),
        label_source="imputed", baseline_timing_json=__file__,
    )
    cfg = RerankerConfig(prm_rollout=conf)

    monkeypatch.setattr(orm_score, "load_config", lambda argv: cfg)
    monkeypatch.setattr(orm_score, "_kernelbench_dir", lambda cfg: _kb(tmp_path))
    calls = {"n": 0}

    def fake_load_scorer(c):
        calls["n"] += 1
        return StubScorer(), _enc()

    monkeypatch.setattr(orm_score, "load_scorer", fake_load_scorer)

    # Task index -> the LANDED unit: scores it, loads the (fake) model once.
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(units_sorted.index(landed)))
    orm_score.main([])
    assert calls["n"] == 1
    assert os.path.exists(orm_score.unit_score_path(out_dir, landed))
    assert _ids_in(orm_score.unit_score_path(out_dir, landed)) == {f"{landed_prefix.prefix_id}__j00"}

    # Task index -> the unit that has NOT landed: must exit without loading the model again.
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(units_sorted.index(missing)))
    orm_score.main([])
    assert calls["n"] == 1   # unchanged -- the second call never touched the GPU
