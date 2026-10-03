"""``run``: the artifacts a search leaves behind, which is what the whole feature is judged by."""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest
import torch

from processkernel.generation.core.model import Problem
from processkernel.config import PRMSearchConfig, RerankerConfig
from processkernel.prm.search import run
from processkernel.prm.search.candidate import Candidate, pkey
from processkernel.prm.search.search import Result

PROBLEM = Problem(level=6, problem_id=12, name="12_Thing.py", ref_arch_src="")
KERNEL = "## Plan\nx\n```python\nimport torch\nclass ModelNew: pass\n```\n"


def cand(cid, text=KERNEL, stop="eos") -> Candidate:
    return Candidate(problem=PROBLEM, head="H", prompt="P", text=text, cid=cid,
                     done=True, stop=stop, spent=42, scores=[0.2, 0.7], cut_chars=[10, 20])


def result_with(*cands) -> Result:
    r = Result()
    for c in cands:
        r.pool[pkey(PROBLEM)].append(c)
    r.steps.append({"step": 0, "rows": [
        {"cid": c.cid, "parent": c.parent, "pkey": pkey(PROBLEM),
         "score": c.scores[-1], "kept": c is cands[0], "tokens": c.spent,
         "cut_char": c.cut_chars[-1], "reached_target": True, "done": True, "stop": c.stop}
        for c in cands
    ]})
    return r


def cfg_for(tmp_path, **over) -> RerankerConfig:
    cfg = RerankerConfig()
    cfg.prm_search = PRMSearchConfig(prm_checkpoint="/c", out_dir=str(tmp_path), **over)
    return cfg


def test_the_pool_is_written_under_kernelbench_naming(tmp_path):
    cfg = cfg_for(tmp_path)
    run.write_artifacts(result_with(cand("a"), cand("b")), {"a": 1.5, "b": 0.5}, cfg,
                        str(tmp_path))
    names = sorted(os.listdir(tmp_path / "pool"))
    assert names == ["level_6_problem_12_sample_0_kernel.py",
                     "level_6_problem_12_sample_1_kernel.py"]


def test_the_pool_holds_extracted_code_not_the_raw_completion(tmp_path):
    cfg = cfg_for(tmp_path)
    run.write_artifacts(result_with(cand("a")), {"a": 1.5}, cfg, str(tmp_path))
    body = (tmp_path / "pool" / "level_6_problem_12_sample_0_kernel.py").read_text()
    assert body.startswith("import torch")
    assert "## Plan" not in body


def test_best_holds_the_highest_orm_score(tmp_path):
    cfg = cfg_for(tmp_path)
    run.write_artifacts(result_with(cand("a"), cand("b")), {"a": 0.5, "b": 9.0}, cfg,
                        str(tmp_path))
    tree = json.loads((tmp_path / "trees" / "problem_6_12.json").read_text())
    assert tree["picked"]["cid"] == "b"
    assert os.path.isfile(tmp_path / "best" / "level_6_problem_12_sample_0_kernel.py")


def test_the_tree_joins_every_pool_entry_to_its_sample_id(tmp_path):
    cfg = cfg_for(tmp_path)
    run.write_artifacts(result_with(cand("a"), cand("b")), {"a": 1.0, "b": 2.0}, cfg,
                        str(tmp_path))
    tree = json.loads((tmp_path / "trees" / "problem_6_12.json").read_text())
    assert {e["cid"]: e["sample_id"] for e in tree["pool"]} == {"a": 0, "b": 1}
    assert all("orm_score" in e for e in tree["pool"])


def test_the_tree_records_every_step_score_and_whether_it_survived(tmp_path):
    cfg = cfg_for(tmp_path)
    run.write_artifacts(result_with(cand("a"), cand("b")), {"a": 1.0, "b": 2.0}, cfg,
                        str(tmp_path))
    tree = json.loads((tmp_path / "trees" / "problem_6_12.json").read_text())
    rows = tree["steps"][0]["candidates"]
    assert {r["cid"] for r in rows} == {"a", "b"}
    assert [r["kept"] for r in rows if r["cid"] == "a"] == [True]
    assert [r["kept"] for r in rows if r["cid"] == "b"] == [False]


def test_a_candidate_with_no_extractable_kernel_is_skipped_but_counted(tmp_path):
    cfg = cfg_for(tmp_path)
    manifest = run.write_artifacts(
        result_with(cand("a"), cand("b", text="## Plan\nnever wrote code\n",
                                    stop="eos_in_plan")),
        {"a": 1.0, "b": 2.0}, cfg, str(tmp_path),
    )
    assert manifest["n_pool"] == 1
    assert manifest["n_no_code"] == 1
    tree = json.loads((tmp_path / "trees" / "problem_6_12.json").read_text())
    assert tree["picked"]["cid"] == "a"


def test_write_pool_false_still_writes_best_and_the_tree(tmp_path):
    cfg = cfg_for(tmp_path, write_pool=False)
    run.write_artifacts(result_with(cand("a")), {"a": 1.0}, cfg, str(tmp_path))
    assert not os.path.isdir(tmp_path / "pool")
    assert os.path.isfile(tmp_path / "best" / "level_6_problem_12_sample_0_kernel.py")
    assert os.path.isfile(tmp_path / "trees" / "problem_6_12.json")


def test_the_manifest_records_the_knobs_the_run_was_shaped_by(tmp_path):
    cfg = cfg_for(tmp_path, beam_width=3, expand=5)
    manifest = run.write_artifacts(result_with(cand("a")), {"a": 1.0}, cfg, str(tmp_path))
    assert manifest["config"]["beam_width"] == 3
    assert manifest["config"]["expand"] == 5
    assert manifest["config"]["advance"] == "tokens"
    assert json.loads((tmp_path / "search_manifest.json").read_text()) == manifest


def test_the_prm_is_batched_by_prm_batch_size_not_the_trainers_eval_batch():
    """rank_eval.load_scorer reads cfg.train.per_device_eval_batch_size, which belongs to the
    trainer. The search has to reach it through its own knob or the field is decoration."""
    cfg = RerankerConfig()
    cfg.prm_search = PRMSearchConfig(prm_checkpoint="/c", prm_batch_size=3)
    cfg.train.per_device_eval_batch_size = 64
    seen = {}

    def fake_load_scorer(passed_cfg, checkpoint):
        seen["batch"] = passed_cfg.train.per_device_eval_batch_size
        seen["checkpoint"] = checkpoint
        return (lambda ids: [0.0] * len(ids)), _Tok()

    run.rank_eval.load_scorer, original = fake_load_scorer, run.rank_eval.load_scorer
    try:
        run._prm(cfg)
    finally:
        run.rank_eval.load_scorer = original
    assert seen == {"batch": 3, "checkpoint": "/c"}
    # and the caller's own config is untouched -- the swap is on a copy
    assert cfg.train.per_device_eval_batch_size == 64


def test_the_backend_is_built_with_the_searchs_utilization_not_vllms_default():
    """vLLM's 0.92 default would leave the PRM and ORM ~6.4 GB of 80 to share."""
    import processkernel.prm.rollout.rollout as R

    rollout_conf = RerankerConfig().prm_rollout
    seen = {}

    class FakeVLLM:
        def __init__(self, model_id, **kw):
            seen.update(kw, model_id=model_id)

    import processkernel.generation.core.backend as B
    B.VLLMBackend, original = FakeVLLM, B.VLLMBackend
    try:
        R._backend(rollout_conf, 0.85)
        assert seen["gpu_memory_utilization"] == 0.85
        seen.clear()
        # unset stays vLLM's own default, so the rollout path is unchanged
        R._backend(rollout_conf)
        assert "gpu_memory_utilization" not in seen
    finally:
        B.VLLMBackend = original


class _Tok:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


def test_the_orm_scores_against_the_problems_own_reference_never_a_second_copy_on_disk(
        monkeypatch):
    """orm_score's disk lookup is a second source that can disagree with the prompt -- and
    does: data.kernelbench_dir resolves to a tree holding no level 1 or 2, which killed
    jobs 2474471-2 after a full hour of generation each."""
    problem = Problem(level=1, problem_id=7, name="7_X.py", ref_arch_src="REF-OFF-THE-PROBLEM")
    c = Candidate(problem=problem, head="H", prompt="P", text=KERNEL, cid="a",
                  done=True, stop="eos", spent=1, scores=[0.1], cut_chars=[1])
    r = Result()
    r.pool[pkey(problem)].append(c)

    seen = []

    class Enc:
        def encode(self, ref, code):
            seen.append(ref)
            return [1, 2, 3]

    def no_disk(*a, **k):
        raise AssertionError("the ORM re-read the reference from disk")

    monkeypatch.setattr(run.orm_score, "load_scorer",
                        lambda conf: (lambda chunk: [0.0] * len(chunk), Enc()))
    monkeypatch.setattr(run.orm_score, "_ref_src", no_disk)
    monkeypatch.setattr(run.orm_score, "_kernelbench_dir", no_disk)

    cfg = RerankerConfig()
    cfg.prm_search = PRMSearchConfig(prm_checkpoint="/c", orm_checkpoint="/o")
    scores = run._orm_scores(cfg, r)

    assert seen == ["REF-OFF-THE-PROBLEM"]
    assert set(scores) == {"a"}


def test_stage_pool_returns_the_sample_id_order_it_wrote(tmp_path):
    cfg = cfg_for(tmp_path)
    staged = run.stage_pool(result_with(cand("a"), cand("b")), cfg, str(tmp_path))
    assert [c.cid for c, _ in staged[pkey(PROBLEM)]] == ["a", "b"]
    assert sorted(os.listdir(tmp_path / "pool")) == [
        "level_6_problem_12_sample_0_kernel.py",
        "level_6_problem_12_sample_1_kernel.py",
    ]


def test_stage_pool_keeps_every_finished_text_including_codeless_ones(tmp_path):
    cfg = cfg_for(tmp_path)
    run.stage_pool(result_with(cand("a"), cand("b", text="prose only")), cfg, str(tmp_path))
    rows = json.load(open(tmp_path / "texts" / "problem_6_12.json"))
    assert [(r["cid"], r["sample_id"], r["text"]) for r in rows] == [
        ("a", 0, KERNEL), ("b", None, "prose only")]
    assert rows[0]["stop"] == "eos" and rows[0]["tokens"] == 42


@pytest.mark.skipif(not torch.cuda.is_available(), reason="_free_backend polls cuda:0")
def test_main_writes_the_pool_even_when_the_orm_dies(tmp_path, monkeypatch):
    """The ORM is the last step and can still fail after hours of generation -- it loads its
    own model onto a card vLLM has just filled. Jobs 2474471-2 raised in _orm_scores and lost
    an hour of generation each because nothing had reached disk yet. The pool is the artifact
    the run exists to produce; the ORM only picks a winner out of it."""
    cfg = cfg_for(tmp_path)
    result = result_with(cand("a"), cand("b"))

    def die(*a, **k):
        raise RuntimeError("ORM died after the search finished")

    monkeypatch.setattr(run, "load_config", lambda argv: cfg)
    monkeypatch.setattr(run, "load_problems", lambda *a, **k: [PROBLEM])
    monkeypatch.setattr(run, "_prm", lambda c: (None, None))
    monkeypatch.setattr(run.rollout, "_backend", lambda *a, **k: object())
    monkeypatch.setattr(run.rollout, "gen_counter", lambda c: len)
    monkeypatch.setattr(run.S, "search", lambda *a, **k: result)
    monkeypatch.setattr(run, "_orm_scores", die)

    with pytest.raises(RuntimeError):
        run.main([])

    assert sorted(os.listdir(tmp_path / "pool")) == [
        "level_6_problem_12_sample_0_kernel.py",
        "level_6_problem_12_sample_1_kernel.py",
    ]


def test_main_writes_the_pool_before_it_frees_the_gpu_and_runs_the_orm(tmp_path, monkeypatch):
    """The same failure on a machine without CUDA: only the GPU teardown is faked."""
    cfg = cfg_for(tmp_path)
    order = []

    def die(*a, **k):
        order.append(("orm", sorted(os.listdir(tmp_path / "pool"))))
        raise RuntimeError("ORM died after the search finished")

    monkeypatch.setattr(run, "load_config", lambda argv: cfg)
    monkeypatch.setattr(run, "load_problems", lambda *a, **k: [PROBLEM])
    monkeypatch.setattr(run, "_prm", lambda c: (None, None))
    monkeypatch.setattr(run.rollout, "_backend", lambda *a, **k: "backend")
    monkeypatch.setattr(run.rollout, "gen_counter", lambda c: len)
    monkeypatch.setattr(run.S, "search", lambda *a, **k: result_with(cand("a"), cand("b")))
    monkeypatch.setattr(run, "_free_backend", lambda b: order.append(("free", b)))
    monkeypatch.setattr(run, "_orm_scores", die)

    with pytest.raises(RuntimeError):
        run.main([])

    pool = ["level_6_problem_12_sample_0_kernel.py", "level_6_problem_12_sample_1_kernel.py"]
    assert order == [("free", "backend"), ("orm", pool)]


def engine(calls, fail=False):
    def shutdown():
        calls.append("shutdown")
        if fail:
            raise RuntimeError("engine gone")

    return SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(
        engine_core=SimpleNamespace(shutdown=shutdown))))


def gib(*free):
    it = iter(free)
    return lambda device: (next(it) * 2**30, 100 * 2**30)


@pytest.mark.parametrize("fail", [False, True])
def test_free_backend_shuts_vllm_down_and_waits_until_the_card_is_free(monkeypatch, capsys, fail):
    calls, sleeps = [], []
    monkeypatch.setattr(torch.cuda, "mem_get_info", gib(10, 50, 80, 90))
    monkeypatch.setattr(time, "sleep", sleeps.append)
    run._free_backend(engine(calls, fail))
    out = capsys.readouterr().out
    assert calls == ["shutdown"] and len(sleeps) == 2
    assert "80.0 of 100.0 GiB free" in out
    assert ("engine shutdown raised" in out) is fail


def test_free_backend_gives_up_after_two_minutes_rather_than_hang(monkeypatch, capsys):
    sleeps = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", gib(*[10] * 60))
    monkeypatch.setattr(time, "sleep", sleeps.append)
    run._free_backend(engine([]))
    assert sleeps == [2] * 60
    assert "10.0 of 100.0 GiB free" in capsys.readouterr().out
