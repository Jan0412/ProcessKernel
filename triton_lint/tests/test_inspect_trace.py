"""The trace reader, exercised on a run the real arm produced.

This is the round-trip proof: arrays written by one process, read back by another with
nothing shared but the files, and joined to the kernel source through the linter's line
numbers. If this passes, the captured data supports later credit assignment; if the
format ever drifts, the reader is what notices.
"""

from __future__ import annotations

import os

import pytest

from kernel_gen import inspect_trace
from kernel_gen.core import artifacts
from kernel_gen.core.backend import FakeBackend
from kernel_gen.core.model import Attempt, Problem, Review, Trajectory
from kernel_gen.core.sampling import CODE_FENCE, SamplingSpec, generate_batch_traced

PLAN = "launch the kernel, do not fall back to torch\n"
CODE = "\nimport torch\n\n\nclass ModelNew:\n    pass\n```\n"
PROBLEM = Problem(level=1, problem_id=19, name="19_ReLU.py", ref_arch_src="ref")
FINDING = {
    "check_id": "F1.2",
    "severity": "fail",
    "message": "Kernel `relu_kernel` is defined but never launched.",
    "data": {"lineno": 5, "kernel": "relu_kernel"},
}


@pytest.fixture
def run_dir(tmp_path):
    """A run dir holding one round of two traced attempts, written by the real writer."""
    out = str(tmp_path)
    backend = FakeBackend(rules=[(PLAN, CODE)], default=PLAN + CODE_FENCE + CODE)
    spec = SamplingSpec(think_temperature=1.0, temperature=0.3, trace_topk=8)

    trajectories = []
    for sample_id, completion in enumerate(
        generate_batch_traced(backend, ["solve", "solve again"], spec)
    ):
        traj = Trajectory(problem=PROBLEM, sample_id=sample_id)
        traj.attempts.append(
            Attempt(
                round=0,
                raw=completion.text,
                code="import torch",
                review=Review(text="F1.2", clean=False, data={"n_fail": 1},
                              findings=[FINDING]),
                trace=completion.trace,
            )
        )
        trajectories.append(traj)

    artifacts.write_trace_config(
        out,
        {"model": "fake", "logprobs_mode": "raw_logprobs", "trace_topk": 8,
         "vocab_size": backend.vocab_size},
    )
    artifacts.write_traces(out, trajectories, 0, window=8, vocab_size=backend.vocab_size)
    return out


def test_the_reader_finds_every_traced_attempt(run_dir):
    records = inspect_trace.load_records(run_dir, 0)
    assert len(records) == 2
    assert all(r["trace"] for r in records)


def test_the_run_level_config_comes_back_intact(run_dir):
    config = inspect_trace.load_trace_config(run_dir)
    assert config["logprobs_mode"] == "raw_logprobs"
    assert config["vocab_size"] == FakeBackend().vocab_size


def test_a_missing_trace_config_is_an_empty_dict_not_a_crash(tmp_path):
    assert inspect_trace.load_trace_config(str(tmp_path)) == {}


def test_inspecting_an_attempt_prints_the_seam_the_findings_and_the_frame(run_dir, capsys):
    record = inspect_trace.load_records(run_dir, 0)[0]
    inspect_trace.inspect(run_dir, record, 0, n_tokens=5)
    out = capsys.readouterr().out

    assert "logprobs_mode raw_logprobs" in out  # the setting that would silently corrupt
    assert "plan tokens" in out and "code tokens" in out
    assert "plan finished on 'stop'" in out
    assert "F1.2" in out and "line 5" in out  # the finding, joined by its line number
    assert "deepconf" in out and "tail" in out


def test_the_summary_reports_how_many_plans_were_cut_off(run_dir, capsys):
    inspect_trace.summarize_round(inspect_trace.load_records(run_dir, 0), 0)
    out = capsys.readouterr().out

    assert "2 attempts, 2 traced" in out
    assert "plans truncated  : 0/2" in out


def test_the_reader_needs_nothing_but_the_files(run_dir):
    # No shared state with the writer: the npz path is resolved from the jsonl record,
    # and the record's stem is the same key the kernels and eval use.
    record = inspect_trace.load_records(run_dir, 0)[0]
    path = os.path.join(artifacts.trace_dir(run_dir, 0), record["trace"]["file"])

    assert os.path.exists(path)
    assert record["trace"]["file"] == f"{record['stem']}.npz"
