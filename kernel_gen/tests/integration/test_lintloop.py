"""Arm A5 end to end: the real engine, the real linter, the real renderer, no GPU.

Only the model is faked, and it is faked at the one honest seam -- prompt in, text out.
So this exercises the whole chain the run depends on: a kernel that cheats is caught by
the checks, rendered into feedback, carried into a repair prompt, and its replacement
is what lands on disk.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from kernel_gen.core import artifacts
from kernel_gen.core.backend import FakeBackend, _fake_tokens
from kernel_gen.core.critics import lint_critic
from kernel_gen.core.engine import run_rounds
from kernel_gen.core.model import Problem
from kernel_gen.core.prompts import build_repair_prompt
from kernel_gen.core.sampling import SamplingSpec

SPEC = SamplingSpec(think_temperature=None)

REF = '''
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)


def get_inputs():
    return [torch.randn(64, 64)]


def get_init_inputs():
    return []
'''

# Defines a kernel, never launches it, and hands the work back to torch: F1.2 + F1.4.
CHEATING = '''```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


class ModelNew(nn.Module):
    def forward(self, x):
        return torch.relu(x)
```'''

# Same kernel, actually launched, output actually returned.
HONEST = '''```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        relu_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
        return out
```'''

BROKEN = "```python\nclass ModelNew(nn.Module:\n```"  # will not parse

PROBLEM = Problem(level=1, problem_id=19, name="19_ReLU.py", ref_arch_src=REF)
REPAIR_MARKER = "## Your previous solution"


def run(backend, out_dir, *, rounds=3, num_samples=2, **critic_kwargs):
    def base_prompt(problem):
        return f"solve {problem.name}"

    return run_rounds(
        backend,
        [(PROBLEM, s) for s in range(num_samples)],
        base_prompt,
        lambda problem, attempt: build_repair_prompt(base_prompt(problem), attempt),
        SPEC,
        critic=lint_critic(**critic_kwargs),
        rounds=rounds,
        on_round_end=lambda r, active: artifacts.write_attempts(out_dir, active, r),
    )


def test_a_cheating_kernel_is_caught_repaired_and_the_repair_is_what_ships(tmp_path):
    backend = FakeBackend(rules=[(REPAIR_MARKER, HONEST)], default=CHEATING)
    trajs = run(backend, str(tmp_path))

    # Round 0 cheated; the linter said so; round 1 fixed it; round 2 never ran.
    assert len(backend.batches) == 2
    assert [len(t.attempts) for t in trajs] == [2, 2]
    assert all(t.final().review.clean for t in trajs)
    assert all(t.final().round == 1 for t in trajs)

    artifacts.write_kernels(str(tmp_path), trajs)
    shipped = (tmp_path / "level_1_problem_19_sample_0_kernel.py").read_text()
    assert "relu_kernel[(" in shipped  # the launched one, not the cheating one


def test_the_repair_prompt_carries_the_real_findings(tmp_path):
    backend = FakeBackend(rules=[(REPAIR_MARKER, HONEST)], default=CHEATING)
    run(backend, str(tmp_path), num_samples=1)

    repair = backend.batches[1][0]
    assert "F1.2" in repair  # dead kernel: defined, never launched
    assert "relu_kernel" in repair  # the finding names it, and the code is quoted back
    assert REPAIR_MARKER in repair


def test_severity_staging_holds_on_a_real_kernel(tmp_path):
    # This kernel trips F1.2 (fail) and F1.4 (warn -- relu is not a heavy op, so the
    # fallback is not the dominant cost). The default policy shows only the fail.
    backend = FakeBackend(default=CHEATING)
    run(backend, str(tmp_path), num_samples=1)
    default_prompt = backend.batches[1][0]

    backend = FakeBackend(default=CHEATING)
    run(backend, str(tmp_path), num_samples=1, policy="all")
    all_prompt = backend.batches[1][0]

    assert "F1.4" not in default_prompt
    assert "F1.4" in all_prompt and "torch.relu" in all_prompt


def test_findings_that_survive_a_round_are_marked_in_the_next_prompt(tmp_path):
    backend = FakeBackend(default=CHEATING)  # never repairs
    run(backend, str(tmp_path), num_samples=1)

    round_1, round_2 = backend.batches[1][0], backend.batches[2][0]
    assert "still here" not in round_1  # nothing to compare against yet
    assert "still here" in round_2  # …and by round 2 the model has ignored it twice


def test_a_generation_that_does_not_parse_is_told_so(tmp_path):
    backend = FakeBackend(rules=[(REPAIR_MARKER, HONEST)], default=BROKEN)
    trajs = run(backend, str(tmp_path), num_samples=1)

    repair = backend.batches[1][0]
    assert "not valid Python" in repair
    assert "from scratch" in repair

    # Outside a loop this file is written to disk and silently scores zero at eval.
    # Here it gets rewritten, and the rewrite is what ships.
    assert trajs[0].attempts[0].review.data["parse_status"] == "syntax_error"
    assert trajs[0].final().review.clean
    assert "relu_kernel[(" in trajs[0].final().code


def test_check_filtering_narrows_what_the_loop_enforces(tmp_path):
    # --lint-checks F1.2: the torch fallback is never even run, let alone mentioned.
    # policy="all" so the assertion discriminates -- otherwise staging alone would hide it.
    backend = FakeBackend(default=CHEATING)
    run(backend, str(tmp_path), num_samples=1, only={"F1.2"}, policy="all")

    repair = backend.batches[1][0]
    assert "F1.2" in repair
    assert "F1.4" not in repair


# -- surviving a crash -----------------------------------------------------
#
# These drive main() rather than run_rounds, because what they pin lives there: the
# checkpoint callback, --skip-existing, and the run dir eval globs.

#: main() builds prompts with the real KernelBench constructor, which embeds
#: ref_arch_src and never mentions the problem name -- so a marker in the source is the
#: only thing a prompt-keyed rule can discriminate on. Both variants are plain relu, so
#: HONEST is clean for either.
MARKED_REF = REF.replace("import torch.nn as nn", "import torch.nn as nn\n\n# variant: {marker}")
ALPHA = Problem(level=1, problem_id=19, name="19_ReLU.py",
                ref_arch_src=MARKED_REF.format(marker="ALPHA"))
BETA = Problem(level=1, problem_id=20, name="20_ReLU.py",
               ref_arch_src=MARKED_REF.format(marker="BETA"))
CRASH_RULES = [("variant: ALPHA", HONEST), ("variant: BETA", CHEATING)]


class CrashingBackend(FakeBackend):
    """Dies the way EngineCore did in job 2339985: mid-generate, on round 1's batch.

    Overrides ``complete_traced``, not ``complete``: that is the method the sampler
    actually calls, and ``complete`` is now a wrapper over it. Hooking the wrapper would
    make this fixture a no-op and quietly retire the two crash-survival tests below.
    """

    def complete_traced(self, prompts, **kwargs):
        if self.batches:
            raise RuntimeError("CUDA error: unspecified launch failure (simulated)")
        return super().complete_traced(prompts, **kwargs)


def drive_main(monkeypatch, out_dir, backend, *, skip_existing, extra=()):
    from kernel_gen.arms import lintloop
    from kernel_gen.core import backend as backend_module

    monkeypatch.setattr(backend_module, "VLLMBackend", lambda *a, **k: backend)
    monkeypatch.setattr(lintloop, "load_problems", lambda *a, **k: [ALPHA, BETA])
    # think-temperature 0 -> single-pass sampling, so one complete() per round and
    # CrashingBackend's batch index is the round index.
    argv = ["lintloop", "--model", "fake", "--level", "1", "--problems", "19,20",
            "--num-samples", "2", "--rounds", "3", "--think-temperature", "0",
            "--output-dir", out_dir, *extra]
    if skip_existing:
        argv.append("--skip-existing")
    monkeypatch.setattr(sys, "argv", argv)
    lintloop.main()


def flat_kernels(out_dir):
    return sorted(f for f in os.listdir(out_dir) if f.endswith("_kernel.py"))


def test_a_crash_costs_only_the_slots_still_in_flight(tmp_path, monkeypatch):
    from kernel_gen.arms import lintloop

    out = str(tmp_path)
    with pytest.raises(RuntimeError, match="unspecified launch failure"):
        drive_main(monkeypatch, out, CrashingBackend(rules=CRASH_RULES), skip_existing=False)

    # Problem 19 went clean in round 0, so it is journaled and its kernel is on disk
    # even though the process died in round 1 -- 13 hours of GPU time that used to be
    # thrown away. Problem 20 was still in flight and is not recorded: resuming from
    # its half-finished trajectory would score a dirty intermediate.
    done = artifacts.read_jsonl(lintloop.lint_log_path(out))
    assert {(r["problem_id"], r["sample_id"]) for r in done} == {(19, 0), (19, 1)}
    assert len(flat_kernels(out)) == 2


def test_a_resumed_run_dir_holds_every_slot_not_just_the_ones_it_ran(tmp_path, monkeypatch):
    from kernel_gen.arms import lintloop

    out = str(tmp_path)
    with pytest.raises(RuntimeError):
        drive_main(monkeypatch, out, CrashingBackend(rules=CRASH_RULES), skip_existing=False)

    repairing = FakeBackend(rules=[(REPAIR_MARKER, HONEST)] + CRASH_RULES)
    drive_main(monkeypatch, out, repairing, skip_existing=True)

    # The second session ran only problem 20's two slots, but eval globs this dir and
    # "N samples per problem" is a contract the whole downstream is built on -- so the
    # slots it skipped must still be here, and none may be journaled twice.
    keys = [(r["problem_id"], r["sample_id"])
            for r in artifacts.read_jsonl(lintloop.lint_log_path(out))]
    assert sorted(keys) == [(19, 0), (19, 1), (20, 0), (20, 1)]
    assert len(keys) == len(set(keys))
    assert len(flat_kernels(out)) == 4


# -- --ref-dir: the staged bytes are the ones the model is asked about ------
#
# Everything else here monkeypatches load_problems away, so deleting `ref_dir=args.ref_dir`
# from main() left the whole suite green. These drive the real loader against a staged dir.


def _stage_level_dir(tmp_path, marker: str, problem_id: int = 19) -> str:
    level = tmp_path / "level6"
    level.mkdir()
    (level / f"{problem_id}_ReLU.py").write_text(
        MARKED_REF.format(marker=marker), encoding="utf-8"
    )
    return str(level)


def test_ref_dir_puts_the_staged_bytes_in_the_prompt(tmp_path, monkeypatch):
    # The property, not the call signature: what the model is asked about must be the file
    # on disk. The KernelBench prompt constructor embeds ref_arch_src verbatim, so a marker
    # in the staged source is observable in the prompt the backend received.
    from kernel_gen.arms import lintloop
    from kernel_gen.core import backend as backend_module

    ref_dir = _stage_level_dir(tmp_path, "STAGED-AND-SCALED")
    out = str(tmp_path / "runs" / "kb6_run")
    backend = FakeBackend(default=HONEST)

    monkeypatch.setattr(backend_module, "VLLMBackend", lambda *a, **k: backend)
    monkeypatch.setattr(sys, "argv", [
        "lintloop", "--model", "fake", "--dataset", "kernelbook", "--level", "6",
        "--ref-dir", ref_dir, "--problems", "19", "--num-samples", "1",
        "--rounds", "1", "--think-temperature", "0", "--output-dir", out,
    ])
    lintloop.main()

    assert backend.batches, "no prompt was ever built"
    assert "STAGED-AND-SCALED" in backend.batches[0][0], (
        "the prompt does not carry the staged reference -- --ref-dir was ignored and the "
        "row was re-converted in-process"
    )


def test_ref_dir_selects_by_the_staged_id_and_is_recorded_in_the_config(tmp_path, monkeypatch):
    # What this pins: the run dir is keyed on the staged filename's prefix, and the config
    # records ref_dir (a reader cannot otherwise tell which reference a run was prompted
    # from) while --dataset still drives the level -> pseudo_level rename.
    #
    # What it deliberately does NOT claim: that it detects a lost --ref-dir. For KernelBook
    # the staged id IS the dataset row index -- that is the whole reason --problems shards
    # the same way with or without the flag -- so no filename can tell the two loaders
    # apart. Only the reference's *content* can, which is what the test above checks.
    from kernel_gen.arms import lintloop
    from kernel_gen.core import backend as backend_module

    ref_dir = _stage_level_dir(tmp_path, "ONLY-THIS-ROW", problem_id=10001)
    out = str(tmp_path / "runs" / "kb6_run")
    backend = FakeBackend(default=HONEST)

    monkeypatch.setattr(backend_module, "VLLMBackend", lambda *a, **k: backend)
    monkeypatch.setattr(sys, "argv", [
        "lintloop", "--model", "fake", "--dataset", "kernelbook", "--level", "6",
        "--ref-dir", ref_dir, "--problems", "10001", "--num-samples", "1",
        "--rounds", "1", "--think-temperature", "0", "--output-dir", out,
    ])
    lintloop.main()

    config = _read_config(os.path.join(out, "generation_config.yaml"))
    assert config["ref_dir"] == ref_dir
    assert config["pseudo_level"] == 6  # --dataset still drives the rename
    assert flat_kernels(out) == ["level_6_problem_10001_sample_0_kernel.py"]


def test_a_shard_run_records_the_name_eval_can_actually_resolve(tmp_path, monkeypatch):
    # The array run writes to runs/<run>/shard_NN. RUN_NAME is relative to runs/, and
    # checker/runs.py stamps this field onto every SampleRef -- so basename would both
    # print an eval command for a directory that does not exist and make all four shards
    # of a run indistinguishable once their results are pooled.
    out = str(tmp_path / "runs" / "Qwen_kb6_lintloop_triton" / "shard_03")
    drive_main(monkeypatch, out, FakeBackend(rules=CRASH_RULES), skip_existing=False)

    config = _read_config(os.path.join(out, "generation_config.yaml"))
    assert config["run_name"] == "Qwen_kb6_lintloop_triton/shard_03"


def _read_config(path: str) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def test_round_dirs_stay_invisible_to_the_non_recursive_eval_glob(tmp_path):
    backend = FakeBackend(rules=[(REPAIR_MARKER, HONEST)], default=CHEATING)
    trajs = run(backend, str(tmp_path), num_samples=2)
    artifacts.write_kernels(str(tmp_path), trajs)

    # eval_run.py globs the run dir with no recursion: it must see 2 kernels, not 6.
    flat = [f for f in os.listdir(tmp_path) if f.endswith("_kernel.py")]
    assert len(flat) == 2

    # …while round_0 is a complete, separately-evaluable baseline of the same 2 slots.
    round_0 = os.listdir(artifacts.round_dir(str(tmp_path), 0))
    assert len([f for f in round_0 if f.endswith("_kernel.py")]) == 2


# -- --trace ---------------------------------------------------------------
#
# The claim --trace has to earn is that it is a pure addition: same prompts, same
# kernels, same journal, plus a directory that was not there before. So these run the
# whole arm twice and diff it, rather than inspecting the new files in isolation.


def _run_dir_fingerprint(out_dir):
    """Every non-trace file under the run dir, with its bytes."""
    seen = {}
    for root, dirs, files in os.walk(out_dir):
        dirs[:] = [d for d in dirs if d != "traces"]
        for name in files:
            path = os.path.join(root, name)
            seen[os.path.relpath(path, out_dir)] = open(path, "rb").read()
    return seen


def test_trace_off_writes_no_traces_directory(tmp_path, monkeypatch):
    out = str(tmp_path)
    drive_main(monkeypatch, out, FakeBackend(rules=CRASH_RULES), skip_existing=False)

    assert not os.path.exists(os.path.join(out, "traces"))


def test_trace_on_changes_nothing_else_in_the_run_dir(tmp_path, monkeypatch):
    # THE non-regression. A traced run and an untraced run must be byte-identical
    # everywhere except traces/ -- the kernels, the round dirs and lint_loop.jsonl all
    # included. If this ever fails, --trace stopped being free.
    off, on = str(tmp_path / "off"), str(tmp_path / "on")
    drive_main(monkeypatch, off, FakeBackend(rules=CRASH_RULES), skip_existing=False)
    drive_main(monkeypatch, on, FakeBackend(rules=CRASH_RULES), skip_existing=False,
               extra=["--trace", "--trace-topk", "4"])

    baseline, traced = _run_dir_fingerprint(off), _run_dir_fingerprint(on)
    assert set(baseline) == set(traced)
    for name, content in baseline.items():
        if name.endswith("generation_config.yaml"):
            continue  # records the flags themselves, so it is expected to differ
        assert traced[name] == content, name


def test_the_journal_record_does_not_grow_when_tracing(tmp_path, monkeypatch):
    # lint_loop.jsonl is read end-to-end by --skip-existing before every resumed run;
    # the findings and the arrays must not leak into it.
    from kernel_gen.arms import lintloop

    off, on = str(tmp_path / "off"), str(tmp_path / "on")
    drive_main(monkeypatch, off, FakeBackend(rules=CRASH_RULES), skip_existing=False)
    drive_main(monkeypatch, on, FakeBackend(rules=CRASH_RULES), skip_existing=False,
               extra=["--trace", "--trace-topk", "4"])

    baseline = artifacts.read_jsonl(lintloop.lint_log_path(off))
    traced = artifacts.read_jsonl(lintloop.lint_log_path(on))
    assert baseline == traced


def test_a_traced_run_writes_one_npz_per_attempt_that_ran(tmp_path, monkeypatch):
    out = str(tmp_path)
    drive_main(monkeypatch, out, FakeBackend(rules=CRASH_RULES), skip_existing=False,
               extra=["--trace", "--trace-topk", "4"])

    # Round 0 ran all 4 slots; problem 20's 2 slots carried into rounds 1 and 2.
    assert len(_npz(out, 0)) == 4
    assert len(_npz(out, 1)) == 2
    for round_index, expected in ((0, 4), (1, 2)):
        records = artifacts.read_jsonl(
            os.path.join(artifacts.trace_dir(out, round_index), "attempts.jsonl")
        )
        assert len(records) == expected
        assert all(r["trace"] is not None for r in records)


def _npz(out_dir, round_index):
    directory = artifacts.trace_dir(out_dir, round_index)
    return sorted(f for f in os.listdir(directory) if f.endswith(".npz"))


def test_the_trace_config_records_what_the_arrays_cannot(tmp_path, monkeypatch):
    import json

    out = str(tmp_path)
    drive_main(monkeypatch, out, FakeBackend(rules=CRASH_RULES), skip_existing=False,
               extra=["--trace", "--trace-topk", "4"])

    config = json.loads(open(os.path.join(out, "traces", "trace_config.json")).read())
    assert config["logprobs_mode"] == "raw_logprobs"
    assert config["trace_topk"] == 4
    assert config["vocab_size"] == FakeBackend().vocab_size


def test_traces_are_journaled_per_round_so_a_crash_keeps_what_finished(tmp_path, monkeypatch):
    out = str(tmp_path)
    with pytest.raises(RuntimeError, match="unspecified launch failure"):
        drive_main(monkeypatch, out, CrashingBackend(rules=CRASH_RULES),
                   skip_existing=False, extra=["--trace", "--trace-topk", "4"])

    # The process died in round 1, but round 0's four traces survived it.
    assert len(_npz(out, 0)) == 4
    assert not os.path.exists(artifacts.trace_dir(out, 1))


def test_a_resumed_slot_does_not_leave_a_stale_record_pointing_at_the_new_arrays(
    tmp_path, monkeypatch
):
    # The two halves exist separately above -- one resumes without --trace, the other
    # traces without resuming -- and the bug lives exactly where they cross.
    out = str(tmp_path)
    with pytest.raises(RuntimeError, match="unspecified launch failure"):
        drive_main(monkeypatch, out, CrashingBackend(rules=CRASH_RULES),
                   skip_existing=False, extra=["--trace", "--trace-topk", "4"])

    drive_main(monkeypatch, out, FakeBackend(rules=[(REPAIR_MARKER, HONEST)] + CRASH_RULES),
               skip_existing=True, extra=["--trace", "--trace-topk", "4"])

    for round_index in (0, 1, 2):
        directory = artifacts.trace_dir(out, round_index)
        if not os.path.exists(directory):
            continue
        records = artifacts.read_jsonl(os.path.join(directory, "attempts.jsonl"))
        stems = [r["stem"] for r in records]
        assert len(stems) == len(set(stems)), f"round {round_index} journalled a stem twice"

        # …and every record describes the arrays it names: the fake tokenizer is
        # deterministic, so the record's own completion must reproduce the npz.
        for record in records:
            expected, _ = _fake_tokens(record["raw"], 4)
            with np.load(os.path.join(directory, record["trace"]["file"])) as data:
                assert data["token_ids"].tolist() == expected, (
                    f"round {round_index} {record['stem']}: the record's completion is not "
                    f"the one these arrays came from"
                )


def test_a_dirty_attempt_is_traced_even_though_it_is_never_journaled(tmp_path, monkeypatch):
    # Slots that carry into another round are training data too -- arguably the most
    # valuable, since a repair round is evidence about what went wrong. write_traces
    # takes every active slot, not only the finished ones.
    out = str(tmp_path)
    drive_main(monkeypatch, out, FakeBackend(rules=CRASH_RULES), skip_existing=False,
               extra=["--trace", "--trace-topk", "4"])

    round_0 = artifacts.read_jsonl(
        os.path.join(artifacts.trace_dir(out, 0), "attempts.jsonl")
    )
    dirty = [r for r in round_0 if not r["clean"]]
    assert dirty, "the cheating variant should have failed round 0"
    assert all(r["findings"] for r in dirty)
    assert any(f["data"].get("lineno") for r in dirty for f in r["findings"])
