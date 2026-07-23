"""What the trace writer puts on disk, and what it must not disturb.

The run dir has three documented contracts (see ``artifacts.py``), all of them about
non-recursive globs over a directory that already holds 140k-175k files. Tracing adds
roughly one file per kernel, so most of this file is about staying out of their way --
and the rest is about the two things this pipeline has been silently discarding since it
was written: the plan prose, and the line numbers the linter already knows.
"""

from __future__ import annotations

import json
import os

from kernel_gen.core import artifacts
from kernel_gen.core.backend import FakeBackend
from kernel_gen.core.model import Attempt, Problem, Review, Trajectory
from kernel_gen.core.sampling import CODE_FENCE, PLAN_PREFIX, SamplingSpec, generate_batch_traced
from kernel_gen.core.trace import read_trace
from triton_lint.model import parse_kernel_filename

PLAN = "fuse the elementwise ops\n"
CODE = "\nimport torch\n```\n"
PROBLEM = Problem(level=1, problem_id=7, name="7_Add.py", ref_arch_src="ref")
STEM = "level_1_problem_7_sample_0_kernel"


def _slot(out_dir: str, *, trace: bool = True, findings: list | None = None) -> Trajectory:
    """One slot with one real traced attempt, run through the real sampler."""
    backend = FakeBackend(rules=[(PLAN, CODE)], default=PLAN + CODE_FENCE + CODE)
    spec = SamplingSpec(think_temperature=1.0, temperature=0.3, trace_topk=8 if trace else None)
    completion = generate_batch_traced(backend, ["solve"], spec)[0]

    review = Review(
        text="F1.2: never launched",
        clean=not findings,
        data={"parse_status": "ok", "n_fail": len(findings or []), "n_warn": 0,
              "check_ids": [f["check_id"] for f in findings or []]},
        findings=findings or [],
    )
    traj = Trajectory(problem=PROBLEM, sample_id=0)
    traj.attempts.append(
        Attempt(round=0, raw=completion.text, code="import torch",
                review=review, trace=completion.trace)
    )
    return traj


def _records(out_dir: str, round_index: int = 0) -> list[dict]:
    path = os.path.join(artifacts.trace_dir(out_dir, round_index), "attempts.jsonl")
    return artifacts.read_jsonl(path)


# ------------------------------------------------------------------ where it lands


def test_traces_go_under_traces_and_nowhere_the_globs_look(tmp_path):
    out = str(tmp_path)
    artifacts.write_attempts(out, [_slot(out)], 0)
    artifacts.write_traces(out, [_slot(out)], 0)

    # Contract 1: eval_run globs the run dir and scan.py scandirs it, both
    # non-recursively and both anchored on _kernel.py. Nothing new may appear flat.
    assert [f for f in os.listdir(out) if not os.path.isdir(os.path.join(out, f))] == []
    assert sorted(os.listdir(out)) == ["rounds", "traces"]
    # Contract 2: rounds/round_0 is itself a run dir; it gets no extra files either.
    assert os.listdir(artifacts.round_dir(out, 0)) == [f"{STEM}.py"]


def test_an_npz_is_invisible_to_the_kernel_filename_parser():
    # Belt and braces on the same contract: even if a .npz ever landed flat, the scanner
    # would not mistake it for a kernel.
    assert parse_kernel_filename(f"{STEM}.npz") is None
    assert parse_kernel_filename(f"{STEM}.py") is not None


def test_each_round_gets_its_own_trace_directory(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)], 0)
    artifacts.write_traces(out, [_slot(out)], 1)

    assert os.path.isdir(artifacts.trace_dir(out, 0))
    assert os.path.isdir(artifacts.trace_dir(out, 1))


# -------------------------------------------------------------- what it preserves


def test_the_plan_prose_is_persisted_for_the_first_time(tmp_path):
    # write_attempts writes attempt.code; the `## Plan` half has never reached disk.
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)], 0)

    raw = _records(out)[0]["raw"]
    assert raw.startswith(PLAN_PREFIX)
    assert PLAN in raw
    assert CODE_FENCE in raw


def test_findings_are_persisted_with_their_line_numbers(tmp_path):
    out = str(tmp_path)
    finding = {"check_id": "F1.2", "severity": "fail", "message": "never launched",
               "data": {"lineno": 8, "kernel": "add_kernel"}}
    artifacts.write_traces(out, [_slot(out, findings=[finding])], 0)

    record = _records(out)[0]
    assert record["findings"][0]["data"]["lineno"] == 8
    assert record["clean"] is False


def test_the_record_carries_the_seam_offsets_and_finish_reasons(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)], 0)

    trace_meta = _records(out)[0]["trace"]
    assert trace_meta["file"] == f"{STEM}.npz"
    assert trace_meta["n_plan_tokens"] > 0
    assert trace_meta["plan_finish_reason"] == "stop"
    assert trace_meta["plan_temperature"] == 1.0 and trace_meta["code_temperature"] == 0.3


def test_the_record_carries_deepconf_summary_statistics(tmp_path):
    # The reason the jsonl exists separately from the npz: it must be enough on its own
    # to decide which traces are worth opening.
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)], 0, window=4, vocab_size=FakeBackend().vocab_size)

    confidence = _records(out)[0]["confidence"]
    assert {"c_least", "c_bottom10", "c_tail", "n_tokens"} <= set(confidence)
    assert "mean_self_cert" in confidence  # only present because vocab_size was given
    assert confidence["c_least"] <= confidence["mean_deepconf_c"]


def test_self_certainty_is_absent_rather_than_wrong_without_a_vocab_size(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)], 0)
    assert "mean_self_cert" not in _records(out)[0]["confidence"]


# ------------------------------------------------------------------- the round trip


def test_the_npz_round_trips_and_joins_to_its_record_by_stem(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)], 0)
    record = _records(out)[0]

    trace = read_trace(os.path.join(artifacts.trace_dir(out, 0), record["trace"]["file"]))
    assert len(trace) == record["confidence"]["n_tokens"]
    assert record["stem"] == STEM  # the same key the kernel, eval and reranker use
    # And the offsets still cut the persisted raw text into its two halves.
    meta = record["trace"]
    assert record["raw"][meta["plan_char_start"] : meta["plan_char_end"]] == PLAN


def test_a_second_round_appends_rather_than_truncating(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)], 0)
    artifacts.write_traces(out, [_slot(out)], 0)
    assert len(_records(out)) == 2


# ---------------------------------------------------------------------- degradation


def test_an_attempt_with_no_trace_still_gets_its_prose_and_findings(tmp_path):
    # A trace that would not assemble must not cost us the raw text too, and must not
    # leave the journal disagreeing with the kernels on disk.
    out = str(tmp_path)
    traj = _slot(out)
    traj.attempts[0].trace = None
    written = artifacts.write_traces(out, [traj], 0)

    record = _records(out)[0]
    assert written == 0
    assert record["trace"] is None and record["confidence"] == {}
    assert record["raw"].startswith(PLAN_PREFIX)
    assert not os.path.exists(os.path.join(artifacts.trace_dir(out, 0), f"{STEM}.npz"))


def test_a_slot_with_no_attempt_this_round_is_skipped(tmp_path):
    out = str(tmp_path)
    assert artifacts.write_traces(out, [Trajectory(problem=PROBLEM, sample_id=0)], 0) == 0
    assert _records(out) == []


# ------------------------------------------------------------------- the run config


def test_the_run_level_facts_are_written_once_not_per_record(tmp_path):
    out = str(tmp_path)
    path = artifacts.write_trace_config(
        out, {"model": "Qwen/Qwen3.6-27B", "logprobs_mode": "raw_logprobs",
              "trace_topk": 20, "vocab_size": 151936}
    )
    config = json.loads(open(path).read())

    assert config["logprobs_mode"] == "raw_logprobs"
    assert config["vocab_size"] == 151936
    assert os.path.basename(path) == "trace_config.json"
    artifacts.write_traces(out, [_slot(out)], 0)
    assert "model" not in _records(out)[0]
