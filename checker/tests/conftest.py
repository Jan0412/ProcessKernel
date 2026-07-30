from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `checker`

import pytest  # noqa: E402

from checker import analyze_source, build_model  # noqa: E402
from checker.core.naming import staged_kernel_filename  # noqa: E402


@pytest.fixture
def analyze():
    """Analyze a source string; returns the ModuleModel."""
    return lambda src, shapes=None: build_model(src, "<test>", shapes)


@pytest.fixture
def check():
    """Run one check over a source string; returns its list of Findings."""

    def run(check_id: str, src: str, shapes=None):
        report = analyze_source(src, "<test>", only={check_id}, fallback_shapes=shapes)
        return [f for f in report.findings if f.check_id == check_id]

    return run


@pytest.fixture
def fired(check):
    """True if the check produced any finding."""
    return lambda check_id, src, shapes=None: bool(check(check_id, src, shapes))


PREAMBLE = """\
import torch
import torch.nn as nn
import triton
import triton.language as tl
"""


def src(body: str) -> str:
    return PREAMBLE + "\n" + body


ELEMENTWISE_KERNEL = '''
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)
'''

#: A complete, well-formed generation: kernel + ModelNew + get_inputs.
GOOD_KERNEL_FILE = src(
    ELEMENTWISE_KERNEL
    + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out

def get_inputs():
    return [torch.rand([4, 8]), torch.rand([4, 8])]
"""
)

#: The same, except the kernel is never launched -- F1.2 fires.
DEAD_KERNEL_FILE = src(
    ELEMENTWISE_KERNEL
    + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        return x + y
"""
)


@pytest.fixture
def make_run(tmp_path):
    """Build a run folder on disk and return its path.

    *files* maps ``(problem_id, sample_id)`` to source; *eval_results* is written as
    ``eval_results.json`` when given. Passing ``config=""`` writes no
    generation_config.yaml, which is how we exercise the level-from-run-name path.
    """

    def build(
        name: str = "TestModel_level1_triton",
        *,
        level: int = 1,
        config: str | None = None,
        files: dict[tuple[int, int], str] | None = None,
        eval_results: dict | None = None,
    ) -> str:
        run = tmp_path / name
        run.mkdir()

        if config is None:
            config = (
                f"pseudo_level: {level}\n"
                f"model: test/model\n"
                f"backend: triton\n"
                f"num_samples: 2\n"
                f"run_name: {name}\n"
            )
        if config:
            (run / "generation_config.yaml").write_text(config)

        if files is None:
            files = {(1, 0): GOOD_KERNEL_FILE}
        for (problem_id, sample_id), source in files.items():
            (run / staged_kernel_filename(level, problem_id, sample_id)).write_text(source)

        if eval_results is not None:
            (run / "eval_results.json").write_text(json.dumps(eval_results))

        return str(run)

    return build


@pytest.fixture
def fake_kernelbench(tmp_path, monkeypatch):
    """Point runs.py at a tiny KernelBench/ + timing/ tree; yields (kb_dir, timing_dir).

    The lru_caches on the index, the baselines and the reference shapes are keyed only
    by level/problem/GPU, so they must be cleared around any test that redirects those
    directories -- otherwise a cached lookup from the real tree leaks in.
    """
    import checker
    from checker import runs

    kb = tmp_path / "KernelBench"
    (kb / "level1").mkdir(parents=True)
    (kb / "level1" / "1_Mul.py").write_text(
        "import torch\n\ndef get_inputs():\n    return [torch.rand([2, 2])]\n"
    )
    (kb / "level1" / "2_Add.py").write_text(
        "import torch\n\n"
        "def get_inputs():\n"
        "    return [torch.rand([4, 8]), torch.rand([4, 8])]\n"
    )
    (kb / "level1" / "notes.txt").write_text("not a problem file")
    (kb / "level1" / "helpers.py").write_text("# no leading problem id")

    timing = tmp_path / "timing"
    (timing / "A100").mkdir(parents=True)
    (timing / "A100" / "baseline_time_torch.json").write_text(
        json.dumps(
            {
                "level1": {
                    "1_Mul.py": {"mean": 2.0, "min": 1.8, "max": 2.2},
                    "2_Add.py": {"mean": 2.0, "min": 1.5, "max": 2.5},
                }
            }
        )
    )

    monkeypatch.setattr(runs, "KERNELBENCH_DIR", str(kb))
    monkeypatch.setattr(runs, "TIMING_DIR", str(timing))
    _clear_caches(runs, checker)
    yield kb, timing
    _clear_caches(runs, checker)


def _clear_caches(runs, checker) -> None:
    runs._level_index.cache_clear()
    runs._baselines.cache_clear()
    checker._reference_shapes.cache_clear()
