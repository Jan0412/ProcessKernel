import json
import os
import pathlib

from reranker.src.config import RerankerConfig
from reranker.src.encoding import SequenceEncoder
from reranker.src.prm.rollout import orm_score


class _StubTokenizer:
    """One id per character -- deterministic, no real vocab needed."""

    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 1000 for c in text]


def _enc() -> SequenceEncoder:
    return SequenceEncoder(_StubTokenizer(), max_length=6144, reserve_ref_tokens=1024)


def _kb(tmp_path) -> str:
    level_dir = tmp_path / "KernelBench" / "level6"
    level_dir.mkdir(parents=True)
    (level_dir / "37_x.py").write_text("class Model:\n    pass\n")
    return str(tmp_path / "KernelBench")


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


def test_identical_code_sha1_is_one_forward_pass(tmp_path):
    items = [orm_score.Item("rollout", f"r{i}", 6, 37, "SAME", "sha1", 4) for i in range(3)]
    scorer = StubScorer()
    out = list(orm_score.score_items(items, scorer, _enc(), _kb(tmp_path), batch_size=8))
    assert scorer.calls == 1
    assert len({o.orm_score for o in out}) == 1 and len(out) == 3


def test_score_is_the_raw_logit_not_a_probability(tmp_path):
    items = [orm_score.Item("rollout", "r1", 6, 37, "x" * 9, "sha", 3)]
    out = list(orm_score.score_items(items, lambda e: [-2.5], _enc(), _kb(tmp_path), 8))
    assert out[0].orm_score == -2.5     # a sigmoid would be 0.076


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
    from reranker.src.prm.rollout import prefixes as prefixes_mod, rollout, stage

    p = prefixes_mod.Prefix(
        prefix_id="p1", source="cut", run_name="a_run", run_tag="ar", shard="shard_00",
        round=0, level=6, problem_id=37, sample_id=0, stem="s0", cut_char=10, cut_index=5,
        cut_kind="code", n_cuts_total=20, rel_depth=0.25, list_key="ar:6:37:0:5",
        split="train", selection="random", selection_score=None, K=2, min_rollouts=1,
    )
    r = rollout.Rollout(
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
