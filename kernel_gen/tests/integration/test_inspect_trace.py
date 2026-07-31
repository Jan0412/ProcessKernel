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
from kernel_gen.core.text import extract_code_block

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
                # What `engine.py` actually stores: the extraction of THIS raw, not a
                # placeholder. The join under test resolves linenos against it, so a
                # fixture disconnected from `raw` would make every assertion below vacuous.
                code=extract_code_block(completion.text),
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


def test_a_finding_resolves_to_the_source_line_the_linter_actually_meant(run_dir, capsys):
    # The join this whole module exists to demonstrate, and the half the assertion above
    # cannot see: it pins that the finding is PRINTED with its number, not that the source
    # shown next to it is the right line.
    #
    # linenos are 1-based into the code the critic was handed -- `record["code"]`, stored
    # verbatim at write time. Slicing raw at code_char_start instead keeps the newline
    # after the fence, the closing fence and any trailing prose, so every line comes out
    # shifted -- on real traces, 479 of 480 findings landed on the wrong line and several
    # onto reasoning prose rather than code (KGEN-10).
    record = inspect_trace.load_records(run_dir, 0)[0]
    expected = record["code"].splitlines()[FINDING["data"]["lineno"] - 1].strip()
    assert expected == "pass"  # the fixture's line 5, stated so the test is readable

    inspect_trace.inspect(run_dir, record, 0, n_tokens=5)
    finding_line = next(
        line for line in capsys.readouterr().out.splitlines() if "F1.2" in line and "line 5" in line
    )
    assert finding_line.rstrip().endswith(f"| {expected}")


def test_the_line_join_uses_the_stored_code_not_a_fresh_extraction(run_dir, capsys):
    """KGEN-20: the join must survive a change to ``extract_code_block``.

    The reader used to re-derive the code by re-running the extractor over ``raw``, so a
    trace captured under an older ranking silently re-joined against different text under
    a newer one -- no error, no warning. Measured over 10,510 real records, 292 (2.78%)
    had already drifted that way, printing the wrong line for 119 findings and a blank one
    for 20 more.

    Simulated here by storing code that a fresh extraction demonstrably would NOT produce.
    That divergence is what six real extractor changes (KGEN-9, 11, 14, 15, 19, 21) each
    manufactured for the records written before them.
    """
    marker = "return marker_the_extractor_cannot_reach(x)"
    drifted = f"import torch\n\n\nclass ModelNew:\n    {marker}\n"

    records = inspect_trace.load_records(run_dir, 0)
    record = dict(records[0], code=drifted)
    # The scenario is only meaningful if the two really disagree.
    assert marker not in extract_code_block(record["raw"])

    inspect_trace.inspect(run_dir, record, 0, n_tokens=5)
    finding_line = next(
        line for line in capsys.readouterr().out.splitlines() if "F1.2" in line and "line 5" in line
    )
    assert finding_line.rstrip().endswith(f"| {marker}")


def test_a_record_written_before_the_code_field_does_not_crash(run_dir, capsys):
    # Pre-KGEN-20 journals carry `n_chars_code` and no `code`. Re-extracting for them would
    # reintroduce the very drift this fix removes, so the reader says so and skips the
    # lookup rather than printing a line it cannot vouch for.
    record = inspect_trace.load_records(run_dir, 0)[0]
    record.pop("code")

    inspect_trace.inspect(run_dir, record, 0, n_tokens=5)
    out = capsys.readouterr().out

    assert "no stored code" in out
    assert "F1.2" in out and "line 5" in out  # the finding itself still reports


def test_the_reader_no_longer_re_extracts_the_code(run_dir):
    # The meta-guard on the fix: the drift is impossible only while there is no second
    # extraction path left to drift. A re-introduced fallback would pass every test above.
    assert not hasattr(inspect_trace, "extract_code_block")


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
