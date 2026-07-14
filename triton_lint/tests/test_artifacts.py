"""The on-disk contracts: the YAML config, and where kernel files are allowed to land.

Nothing tests the ``generation_config.yaml`` coupling today, and it is the sharpest
edge in the repo: the writer is PyYAML, the reader is a 12-line hand-rolled scanner,
and the two agree only by convention. The level key is the part that bites -- see
``kernel_gen/core/artifacts.py``'s docstring.
"""

from __future__ import annotations

import os

from kernel_gen.core.artifacts import round_dir, write_attempts, write_config, write_kernels
from kernel_gen.core.model import Attempt, Problem, Review, Trajectory

from triton_lint.runs import load_run

BENCH_CONFIG = {
    "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "level": 1,
    "num_samples": 10,
    "backend": "triton",
    "run_name": "test_run",
    "lint_checks": "F1.2,F1.4",
}


def _traj(problem_id: int, sample_id: int, *attempts: Attempt) -> Trajectory:
    problem = Problem(level=1, problem_id=problem_id, name="x.py", ref_arch_src="")
    return Trajectory(problem=problem, sample_id=sample_id, attempts=list(attempts))


def _attempt(round_: int, code: str, *, clean=False, n_fail=0) -> Attempt:
    return Attempt(
        round=round_,
        raw=code,
        code=code,
        review=Review(text="", clean=clean, data={"n_fail": n_fail, "parse_status": "ok"}),
    )


# -- the YAML contract -----------------------------------------------------


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
    # and runs.py's scanner drops those. This is why --lint-checks is a string.
    write_config(str(tmp_path), BENCH_CONFIG, dataset="kernelbench")
    raw = (tmp_path / "generation_config.yaml").read_text()
    assert "lint_checks: F1.2,F1.4" in raw


def test_round_0_gets_its_own_config_so_it_is_a_run_dir_in_its_own_right(tmp_path):
    write_config(str(tmp_path), BENCH_CONFIG, dataset="kernelbench")

    # round_0 is the unrefined baseline; eval and load_run() get pointed straight at it.
    assert load_run(round_dir(str(tmp_path), 0)).level == 1


# -- where files land ------------------------------------------------------


def test_final_kernels_go_flat_and_intermediates_stay_invisible(tmp_path):
    trajs = [_traj(19, 0, _attempt(0, "dirty", n_fail=2), _attempt(1, "clean", clean=True))]

    write_attempts(str(tmp_path), trajs, round_index=0)
    write_attempts(str(tmp_path), trajs, round_index=1)
    write_kernels(str(tmp_path), trajs)

    # eval_run.py globs the run dir non-recursively: it must see exactly one kernel.
    flat = sorted(f for f in os.listdir(tmp_path) if f.endswith("_kernel.py"))
    assert flat == ["level_1_problem_19_sample_0_kernel.py"]
    assert (tmp_path / flat[0]).read_text() == "clean"

    # …and each round dir is a valid run dir holding that round's version.
    r0 = round_dir(str(tmp_path), 0)
    assert (
        open(os.path.join(r0, "level_1_problem_19_sample_0_kernel.py")).read() == "dirty"
    )


def test_a_slot_that_never_went_clean_is_still_written(tmp_path):
    # "N samples per problem" is a contract; dropping the dirty slots would bias
    # pass@k, the sweep and the reranker's lists toward the easy problems.
    trajs = [
        _traj(19, 0, _attempt(0, "best", n_fail=1), _attempt(1, "worse", n_fail=4)),
        _traj(19, 1, _attempt(0, "ok", clean=True)),
    ]
    assert write_kernels(str(tmp_path), trajs) == 2
    assert (tmp_path / "level_1_problem_19_sample_0_kernel.py").read_text() == "best"
