"""``core/artifacts.py``: the on-disk contracts, and what the trace writer adds.

Two concerns, one module, so one file.

*The run-dir contracts* (bottom section): the ``generation_config.yaml`` coupling -- the
writer is PyYAML, the reader a 12-line hand-rolled scanner, and the two agree only by
convention, the level key being the part that bites -- and where kernel files are
allowed to land (the eval glob is non-recursive).

*The trace writer* (top sections): the run dir already holds 140k-175k files, so the
traces must stay out of the non-recursive globs' way; and they persist what the kernel
file drops -- the plan prose and the exact prompt.
"""

from __future__ import annotations

import json
import os

from processkernel.generation.core import artifacts
from processkernel.generation.core.artifacts import write_config, write_kernels
from processkernel.generation.core.backend import FakeBackend
from processkernel.generation.core.model import Attempt, Problem, Trajectory
from processkernel.generation.core.sampling import CODE_FENCE, PLAN_PREFIX, SamplingSpec, generate_batch_traced
from processkernel.generation.core.trace import read_trace
from processkernel.checker.core.naming import parse_kernel_filename
from processkernel.checker.runs import load_run

PLAN = "fuse the elementwise ops\n"
CODE = "\nimport torch\n```\n"
PROBLEM = Problem(level=1, problem_id=7, name="7_Add.py", ref_arch_src="ref")
STEM = "level_1_problem_7_sample_0_kernel"

BENCH_CONFIG = {
    "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "level": 1,
    "num_samples": 10,
    "backend": "triton",
    "run_name": "test_run",
    "problems": "1,5,10",
}


def _cfg_traj(problem_id: int, sample_id: int, *attempts: Attempt) -> Trajectory:
    problem = Problem(level=1, problem_id=problem_id, name="x.py", ref_arch_src="")
    return Trajectory(problem=problem, sample_id=sample_id, attempts=list(attempts))


def _cfg_attempt(code: str) -> Attempt:
    return Attempt(raw=code, code=code)


def _slot(out_dir: str, *, trace: bool = True) -> Trajectory:
    """One slot with one real traced attempt, run through the real sampler."""
    backend = FakeBackend(rules=[(PLAN, CODE)], default=PLAN + CODE_FENCE + CODE)
    spec = SamplingSpec(think_temperature=1.0, temperature=0.3, trace_topk=8 if trace else None)
    completion = generate_batch_traced(backend, ["solve"], spec)[0]

    traj = Trajectory(problem=PROBLEM, sample_id=0)
    traj.attempts.append(
        Attempt(raw=completion.text, code="import torch", trace=completion.trace)
    )
    return traj


def _records(out_dir: str) -> list[dict]:
    path = os.path.join(artifacts.trace_dir(out_dir), "attempts.jsonl")
    return artifacts.read_jsonl(path)


# ------------------------------------------------------------------ where it lands


def test_traces_go_under_traces_and_nowhere_the_globs_look(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])

    # Contract 1: eval_run globs the run dir and scan.py scandirs it, both
    # non-recursively and both anchored on _kernel.py. Nothing new may appear flat.
    assert [f for f in os.listdir(out) if not os.path.isdir(os.path.join(out, f))] == []
    assert os.listdir(out) == ["traces"]


def test_an_npz_is_invisible_to_the_kernel_filename_parser():
    # Belt and braces on the same contract: even if a .npz ever landed flat, the scanner
    # would not mistake it for a kernel.
    assert parse_kernel_filename(f"{STEM}.npz") is None
    assert parse_kernel_filename(f"{STEM}.py") is not None


def test_traces_land_where_the_prm_corpus_reads_them(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])

    assert artifacts.layout(out) == (out, os.path.join(out, "traces"))
    assert os.path.isfile(os.path.join(artifacts.layout(out)[1], "attempts.jsonl"))


def test_a_lint_loop_run_is_read_from_its_first_generation(tmp_path):
    # The paper's runs: the run root holds a later, repaired kernel.
    run = tmp_path / "run"
    (run / "rounds" / "round_0").mkdir(parents=True)
    (run / "traces" / "round_0").mkdir(parents=True)
    assert artifacts.layout(str(run)) == (
        str(run / "rounds" / "round_0"), str(run / "traces" / "round_0"))


# -------------------------------------------------------------- what it preserves


def test_the_plan_prose_is_persisted_for_the_first_time(tmp_path):
    # The kernel file holds only attempt.code; the `## Plan` half lives only here.
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])

    raw = _records(out)[0]["raw"]
    assert raw.startswith(PLAN_PREFIX)
    assert PLAN in raw
    assert CODE_FENCE in raw


def test_the_graded_code_is_persisted_verbatim_not_as_a_length(tmp_path):
    # KGEN-20. The record used to keep `n_chars_code` and leave the reader to re-derive the
    # code by re-running the extractor over `raw` -- which silently returns something else
    # once the extractor's ranking changes, as it has six times (KGEN-9, 11, 14, 15, 19, 21).
    # A lineno is 1-based into THIS string, so THIS string is what has to survive.
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])

    record = _records(out)[0]
    assert record["code"] == "import torch"  # _slot's attempt.code, byte for byte
    assert "n_chars_code" not in record  # len(record["code"]) recovers it


def test_the_record_carries_the_system_prompt_and_the_user_prompt(tmp_path):
    # The two fields that make a line a self-contained training example: the constant
    # system message and the exact user turn the model saw. Without them the conversation
    # is only reconstructable by replaying the prompt builders on a pinned dataset.
    out = str(tmp_path)
    traj = _slot(out)
    traj.attempts[0].prompt = "solve problem 7"
    artifacts.write_traces(out, [traj], system_prompt="You write custom kernels ...")

    record = _records(out)[0]
    assert record["system_prompt"] == "You write custom kernels ..."
    assert record["prompt"] == "solve problem 7"


def test_the_record_carries_no_linter_verdict(tmp_path):
    # The linter is a final check run on its own, never part of generation.
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])
    assert not {"clean", "feedback", "findings"} & set(_records(out)[0])


def test_the_record_carries_the_seam_offsets_and_finish_reasons(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])

    trace_meta = _records(out)[0]["trace"]
    assert trace_meta["file"] == f"{STEM}.npz"
    assert trace_meta["n_plan_tokens"] > 0
    assert trace_meta["plan_finish_reason"] == "stop"
    assert trace_meta["plan_temperature"] == 1.0 and trace_meta["code_temperature"] == 0.3


def test_the_record_carries_deepconf_summary_statistics(tmp_path):
    # The reason the jsonl exists separately from the npz: it must be enough on its own
    # to decide which traces are worth opening.
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)], window=4, vocab_size=FakeBackend().vocab_size)

    confidence = _records(out)[0]["confidence"]
    assert {"c_least", "c_bottom10", "c_tail", "n_tokens"} <= set(confidence)
    assert "mean_self_cert" in confidence  # only present because vocab_size was given
    assert confidence["c_least"] <= confidence["mean_deepconf_c"]


def test_self_certainty_is_absent_rather_than_wrong_without_a_vocab_size(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])
    assert "mean_self_cert" not in _records(out)[0]["confidence"]


# ------------------------------------------------------------------- the round trip


def test_the_npz_round_trips_and_joins_to_its_record_by_stem(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])
    record = _records(out)[0]

    trace = read_trace(os.path.join(artifacts.trace_dir(out), record["trace"]["file"]))
    assert len(trace) == record["confidence"]["n_tokens"]
    assert record["stem"] == STEM  # the same key the kernel, eval and reranker use
    # And the offsets still cut the persisted raw text into its two halves.
    meta = record["trace"]
    assert record["raw"][meta["plan_char_start"] : meta["plan_char_end"]] == PLAN


def test_a_second_write_appends_rather_than_truncating(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])
    artifacts.write_traces(out, [_slot(out)])
    assert len(_records(out)) == 2


# ------------------------------------------------------------ the name eval resolves


def test_eval_run_name_is_the_path_below_runs_not_the_basename():
    # eval takes RUN_NAME relative to runs/ and resolves runs/$RUN_NAME. A sharded array
    # run lives one level deeper, so basename would name a directory that does not exist
    # -- for every task of the array.
    assert artifacts.eval_run_name("/x/KernelBench/runs/model_kb6_lintloop") == "model_kb6_lintloop"
    assert (
        artifacts.eval_run_name("/x/KernelBench/runs/model_kb6_lintloop/shard_07")
        == "model_kb6_lintloop/shard_07"
    )


def test_eval_run_name_normalises_trailing_slashes_and_dot_segments():
    assert artifacts.eval_run_name("/x/runs/a_run/") == "a_run"
    assert artifacts.eval_run_name("/x/runs/a_run/shard_00/") == "a_run/shard_00"
    assert artifacts.eval_run_name("/x/runs/a_run/./shard_00") == "a_run/shard_00"


def test_eval_run_name_takes_the_last_runs_component():
    # The repo nests one (KernelBench/runs/…), and a checkout could itself sit under a
    # directory called runs. The innermost one is the one eval is relative to.
    assert artifacts.eval_run_name("/runs/checkout/runs/a_run/shard_01") == "a_run/shard_01"


def test_eval_run_name_falls_back_to_the_basename_without_a_runs_component():
    # An ad-hoc --output-dir must still print something usable rather than raising.
    assert artifacts.eval_run_name("/tmp/scratch/my_run") == "my_run"
    assert artifacts.eval_run_name("/tmp/scratch/runs") == "runs"


# ------------------------------------------------------- contract 4: prune on resume
#
# write_traces appends and write_trace overwrites (the two tests above pin exactly that),
# so "one record per stem, describing its own arrays" is not a property of the
# writers. prune_traces is what makes it hold across a resume.


def test_prune_traces_drops_the_records_and_arrays_of_slots_being_regenerated(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])
    assert os.path.exists(os.path.join(artifacts.trace_dir(out), f"{STEM}.npz"))

    assert artifacts.prune_traces(out, {STEM}) == 1

    assert _records(out) == []
    assert not os.path.exists(os.path.join(artifacts.trace_dir(out), f"{STEM}.npz"))


def test_prune_traces_leaves_every_other_slot_untouched(tmp_path):
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])

    assert artifacts.prune_traces(out, {"level_1_problem_999_sample_0_kernel"}) == 0

    record = _records(out)[0]
    assert record["stem"] == STEM
    assert os.path.exists(os.path.join(artifacts.trace_dir(out), record["trace"]["file"]))


def test_prune_traces_removes_arrays_no_record_ever_mentioned(tmp_path):
    # write_traces writes each .npz inside its loop and appends the journal once at the
    # end, so a crash between the two leaves orphan arrays. They belong to a slot that is
    # about to be regenerated, so they must not survive under the name the new record
    # will claim.
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])
    os.unlink(os.path.join(artifacts.trace_dir(out), "attempts.jsonl"))

    artifacts.prune_traces(out, {STEM})

    assert not os.path.exists(os.path.join(artifacts.trace_dir(out), f"{STEM}.npz"))


def test_prune_traces_drops_a_record_that_never_had_a_trace(tmp_path):
    # An attempt whose trace would not assemble still gets a record (with trace: null) so
    # the journal does not disagree with the kernels. That record is just as stale after a
    # regeneration as a traced one, and there is no array name on it to follow.
    out = str(tmp_path)
    traj = _slot(out)
    traj.attempts[0].trace = None  # as above: a trace that would not assemble
    artifacts.write_traces(out, [traj])
    assert _records(out)[0]["trace"] is None

    assert artifacts.prune_traces(out, {STEM}) == 1
    assert _records(out) == []


def test_prune_traces_leaves_the_trace_config_and_other_files_alone(tmp_path):
    # traces/ also holds trace_config.json, and could hold anything a reader dropped there.
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])
    artifacts.write_trace_config(out, {"model": "m"})
    os.mkdir(os.path.join(out, "traces", "notes"))

    assert artifacts.prune_traces(out, {STEM}) == 1

    assert os.path.exists(os.path.join(out, "traces", "trace_config.json"))
    assert os.path.isdir(os.path.join(out, "traces", "notes"))


def test_prune_traces_is_a_no_op_on_a_fresh_run(tmp_path):
    # The common case: no traces/ at all. Must not raise and must not create anything.
    out = str(tmp_path)
    assert artifacts.prune_traces(out, {STEM}) == 0
    assert artifacts.prune_traces(out, set()) == 0
    assert not os.path.exists(os.path.join(out, "traces"))


def test_prune_traces_leaves_no_temp_file_behind(tmp_path):
    # The journal is rewritten through a temp file so a crash cannot truncate it; the
    # temp must not survive into a directory something else scandirs.
    out = str(tmp_path)
    artifacts.write_traces(out, [_slot(out)])
    artifacts.prune_traces(out, {STEM})

    assert [f for f in os.listdir(artifacts.trace_dir(out)) if f.endswith(".tmp")] == []


# ---------------------------------------------------------------------- degradation


def test_an_attempt_with_no_trace_still_gets_its_prose(tmp_path):
    # A trace that would not assemble must not cost us the raw text too, and must not
    # leave the journal disagreeing with the kernels on disk.
    out = str(tmp_path)
    traj = _slot(out)
    traj.attempts[0].trace = None
    written = artifacts.write_traces(out, [traj])

    record = _records(out)[0]
    assert written == 0
    assert record["trace"] is None and record["confidence"] == {}
    assert record["raw"].startswith(PLAN_PREFIX)
    assert not os.path.exists(os.path.join(artifacts.trace_dir(out), f"{STEM}.npz"))


def test_a_slot_with_no_attempt_is_skipped(tmp_path):
    out = str(tmp_path)
    assert artifacts.write_traces(out, [Trajectory(problem=PROBLEM, sample_id=0)]) == 0
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
    artifacts.write_traces(out, [_slot(out)])
    assert "model" not in _records(out)[0]


# ================= the run-dir contracts (ex-test_artifacts.py) =================
#
# The generation_config.yaml coupling and where kernel files land. Independent of
# tracing; the sharpest edges in the repo (the level key, the non-recursive eval glob).


def test_kernelbench_config_round_trips_and_carries_no_pseudo_level(tmp_path):
    write_config(str(tmp_path), BENCH_CONFIG, dataset="kernelbench")

    info = load_run(str(tmp_path))
    assert info.level == 1
    assert info.model == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert info.num_samples == 10

    raw = (tmp_path / "generation_config.yaml").read_text()
    assert "pseudo_level" not in raw


def test_kernelbook_config_round_trips_as_pseudo_level_and_carries_no_level(tmp_path):
    write_config(str(tmp_path), {**BENCH_CONFIG, "level": 5}, dataset="kernelbook")

    assert load_run(str(tmp_path)).level == 5

    raw = (tmp_path / "generation_config.yaml").read_text()
    assert "pseudo_level: 5" in raw
    # `runs.py` resolves `pseudo_level or level`. Emitting both would make a
    # KernelBench run at level 1 report level 5 and break every filename lookup.
    assert not any(line.startswith("level:") for line in raw.splitlines())


def test_comma_separated_flags_survive_the_flat_yaml_reader(tmp_path):
    # A nargs="+" flag would serialize as a block list, whose lines start with "-",
    # and runs.py's scanner drops those. This is why --problems is a string.
    write_config(str(tmp_path), BENCH_CONFIG, dataset="kernelbench")
    raw = (tmp_path / "generation_config.yaml").read_text()
    assert "problems: 1,5,10" in raw


def test_the_config_goes_to_the_run_dir_and_nowhere_else(tmp_path):
    write_config(str(tmp_path), BENCH_CONFIG, dataset="kernelbench")

    assert os.listdir(tmp_path) == ["generation_config.yaml"]
    assert load_run(str(tmp_path)).level == 1


def test_the_kernel_goes_flat_and_nowhere_else(tmp_path):
    trajs = [_cfg_traj(19, 0, _cfg_attempt("kernel"))]

    write_kernels(str(tmp_path), trajs)

    # eval_run.py globs the run dir non-recursively: it must see exactly one kernel.
    flat = sorted(f for f in os.listdir(tmp_path) if f.endswith("_kernel.py"))
    assert flat == ["level_1_problem_19_sample_0_kernel.py"]
    assert (tmp_path / flat[0]).read_text() == "kernel"
    assert os.listdir(tmp_path) == flat


def test_a_slot_with_empty_code_is_still_written(tmp_path):
    # "N samples per problem" is a contract; dropping the slots whose code could not be
    # extracted would bias pass@k and the reranker's lists toward the easy problems.
    trajs = [_cfg_traj(19, 0, _cfg_attempt("")), _cfg_traj(19, 1, _cfg_attempt("ok"))]
    assert write_kernels(str(tmp_path), trajs) == 2
    assert (tmp_path / "level_1_problem_19_sample_0_kernel.py").read_text() == ""


# -- the degradation paths -------------------------------------------------


def test_a_slot_with_no_attempt_at_all_warns_and_writes_nothing(tmp_path, capsys):
    # No attempt -- nothing to ship. It must say so rather than write an empty file
    # that eval would score as a failure indistinguishable from a bad kernel.
    trajs = [_cfg_traj(19, 0)]  # no attempts
    assert write_kernels(str(tmp_path), trajs) == 0
    assert "no attempt at all" in capsys.readouterr().out
    assert not [f for f in os.listdir(tmp_path) if f.endswith("_kernel.py")]


def test_read_jsonl_of_a_missing_file_is_empty_not_an_error(tmp_path):
    # --skip-existing reads the journal before the first run ever wrote it.
    assert artifacts.read_jsonl(str(tmp_path / "never_written.jsonl")) == []


def test_append_jsonl_creates_the_dir_and_appends_one_line_per_record(tmp_path):
    path = str(tmp_path / "new" / "journal.jsonl")
    artifacts.append_jsonl(path, [{"a": 1}])
    artifacts.append_jsonl(path, [{"a": 2}, {"a": 3}])
    assert artifacts.read_jsonl(path) == [{"a": 1}, {"a": 2}, {"a": 3}]
