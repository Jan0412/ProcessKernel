"""``processkernel.generation.generate``: the driver and the run dir it leaves behind.

The layout is what every downstream reader depends on: kernels flat (eval and the PRM
and ORM data builders), traces under ``traces/`` (the PRM corpus), and a journal
``--skip-existing`` resumes from.
"""

from __future__ import annotations

import os
import shlex

import pytest

from processkernel.generation import generate as gen
from processkernel.generation.core import artifacts
from processkernel.generation.core.backend import FakeBackend
from processkernel.generation.core.engine import generate
from processkernel.generation.core.model import Problem
from processkernel.generation.core.sampling import SamplingSpec

GOOD = "```python\nimport torch\n```"
PROBLEMS = [Problem(level=1, problem_id=i, name=f"{i}_P.py", ref_arch_src="ref") for i in (3, 4)]


def _trajectories(trace: bool):
    spec = SamplingSpec(think_temperature=None, trace_topk=4 if trace else None)
    slots = [(p, s) for p in PROBLEMS for s in range(2)]
    return generate(FakeBackend(default=GOOD), slots, lambda p: f"solve {p.problem_id}", spec)


def test_the_run_dir_holds_kernels_traces_and_the_journal_and_nothing_else(tmp_path):
    out = str(tmp_path)
    n = gen.write_outputs(out, _trajectories(trace=True), trace=True, window=4, vocab_size=None)

    assert n == 4
    flat = sorted(f for f in os.listdir(out) if f.endswith("_kernel.py"))
    assert flat == [f"level_1_problem_{p}_sample_{s}_kernel.py" for p in (3, 4) for s in (0, 1)]
    assert sorted(os.listdir(out)) == sorted([*flat, "generation.jsonl", "traces"])
    records = artifacts.read_jsonl(os.path.join(artifacts.trace_dir(out), "attempts.jsonl"))
    assert [r["stem"] for r in records] == [f[: -len(".py")] for f in flat]
    assert gen.load_done_slots(out) == {(p, s) for p in (3, 4) for s in (0, 1)}


def test_without_trace_no_traces_dir_is_written(tmp_path):
    out = str(tmp_path)
    gen.write_outputs(out, _trajectories(trace=False), trace=False, window=4, vocab_size=None)
    assert not os.path.exists(os.path.join(out, "traces"))
    assert os.path.exists(gen.journal_path(out))


def test_the_journal_is_named_for_generation_not_for_a_loop(tmp_path):
    assert os.path.basename(gen.journal_path(str(tmp_path))) == "generation.jsonl"
    assert gen.load_done_slots(str(tmp_path)) == set()  # nothing written yet


def test_the_lint_loop_options_are_gone():
    dests = {a.dest for a in gen.build_parser()._actions}
    assert not dests & {"rounds", "feedback_policy", "lint_checks", "max_findings",
                        "include_hardware", "backend", "option"}


def test_default_output_dir_names_the_model_and_the_level():
    args = gen.build_parser().parse_args(["--model", "org/Model-7B", "--level", "6",
                                          "--dataset", "kernelbook"])
    assert gen.default_output_dir(args).endswith(os.path.join("runs", "Model-7B_kb6_triton"))


def test_thinking_and_a_plan_prefill_together_are_rejected():
    with pytest.raises(SystemExit, match="enable-thinking"):
        gen.main(["--model", "m", "--level", "1", "--problems", "0",
                  "--enable-thinking", "--think-temperature", "1.0", "--dry-run"])


def test_dry_run_prints_the_paper_prompt_and_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(gen, "load_problems", lambda *a, **k: PROBLEMS[:1])
    monkeypatch.setattr(gen, "build_base_prompt", lambda p, deltas: "ONE-SHOT PROMPT")
    out = str(tmp_path / "run")
    gen.main(["--model", "m", "--level", "1", "--problems", "3", "--dry-run",
              "--output-dir", out])

    printed = capsys.readouterr().out
    assert "ONE-SHOT PROMPT" in printed and gen.SYSTEM_PROMPT in printed
    assert not os.path.exists(out)


@pytest.mark.parametrize(("flag", "deltas"), [
    ([], frozenset()),
    (["--prompt-deltas", "contract,precision"], frozenset({"contract", "precision"})),
])
def test_prompt_deltas_reach_the_prompt_only_when_asked_for(tmp_path, capsys, monkeypatch,
                                                           flag, deltas):
    seen = []
    monkeypatch.setattr(gen, "load_problems", lambda *a, **k: PROBLEMS[:1])
    monkeypatch.setattr(gen, "build_base_prompt", lambda p, d: seen.append(d) or "PROMPT")
    gen.main(["--model", "m", "--level", "1", "--problems", "3", "--dry-run",
              "--output-dir", str(tmp_path / "run"), *flag])
    assert seen == [deltas]


def test_an_unknown_prompt_delta_is_rejected_before_anything_runs():
    with pytest.raises(SystemExit, match="--prompt-deltas: unknown prompt delta"):
        gen.main(["--model", "m", "--level", "1", "--problems", "0", "--dry-run",
                  "--prompt-deltas", "pitfalls"])


@pytest.mark.parametrize(("rel", "runs", "name"), [
    ("my work/runs/m_kb7/shard_03", "my work/runs", "m_kb7/shard_03"),
    ("out/m_level1_triton", "out", "m_level1_triton"),
])
def test_the_grading_hint_runs_the_patched_grader_on_this_run(tmp_path, capsys, rel, runs, name):
    gen.report(8, str(tmp_path / rel), 7, 4)
    line = next(l for l in capsys.readouterr().out.splitlines() if "eval_from_generations" in l)
    cmd = shlex.split(line)
    assert cmd[:4] == ["uv", "run", "python", "scripts/eval_from_generations.py"]
    assert dict(a.split("=", 1) for a in cmd[4:]) == {
        "run_name": name, "runs_dir": str(tmp_path / runs), "dataset_src": "local",
        "level": "7", "backend": "triton", "num_samples_per_problem": "4",
    }
