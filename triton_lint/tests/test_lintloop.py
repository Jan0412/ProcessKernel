"""Arm A5 end to end: the real engine, the real linter, the real renderer, no GPU.

Only the model is faked, and it is faked at the one honest seam -- prompt in, text out.
So this exercises the whole chain the run depends on: a kernel that cheats is caught by
the checks, rendered into feedback, carried into a repair prompt, and its replacement
is what lands on disk.
"""

from __future__ import annotations

import os

from kernel_gen.core import artifacts
from kernel_gen.core.backend import FakeBackend
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
