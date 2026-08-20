"""``run``: the artifacts a search leaves behind, which is what the whole feature is judged by."""

from __future__ import annotations

import json
import os

from kernel_gen.core.model import Problem
from reranker.src.config import PRMSearchConfig, RerankerConfig
from reranker.src.prm.search import run
from reranker.src.prm.search.candidate import Candidate, pkey
from reranker.src.prm.search.search import Result

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
