"""runs.py -- the run-folder data layer: config, eval results, references, baselines."""

from __future__ import annotations

import os

import pytest

from checker import runs
from checker.runs import (
    SampleRef,
    baseline_time,
    iter_samples,
    load_eval_results,
    load_run,
    reference_filename,
    reference_path,
    reference_source,
    speedup,
)

EVAL_RESULTS = {
    "2": [
        {
            "sample_id": 0,
            "compiled": True,
            "correctness": True,
            "runtime": 1.0,
            "runtime_stats": {"min": 0.9, "mean": 1.0},
            "metadata": {"hardware": "NVIDIA A100-SXM4-80GB"},
        },
        {
            "sample_id": 1,
            "compiled": True,
            "correctness": False,
            "runtime": None,
            "metadata": {"runtime_error_name": "CUDA error"},
        },
        {"compiled": False},  # no sample_id -- not an evaluated sample
    ]
}


def sample(**overrides) -> SampleRef:
    fields = {
        "run_name": "run",
        "level": 1,
        "problem_id": 2,
        "sample_id": 0,
        "kernel_path": "k.py",
        "compiled": True,
        "correct": True,
        "runtime": 1.0,
        "runtime_min": 0.9,
        "hardware": "NVIDIA A100-SXM4-80GB",
        "error_name": None,
    }
    fields.update(overrides)
    return SampleRef(**fields)


class TestReadYamlScalars:
    def test_reads_flat_keys_only(self, tmp_path):
        path = tmp_path / "generation_config.yaml"
        path.write_text(
            "# leading comment\n"
            "run_name: 'my_run'  # trailing comment\n"
            "level: 2\n"
            "nested:\n"
            "  indented: ignored\n"
            "- list_item\n"
            "no_colon_here\n"
            '"quoted": "value"\n'
        )
        cfg = runs._read_yaml_scalars(str(path))
        assert cfg["run_name"] == "my_run"  # comment and quotes stripped
        assert cfg["level"] == "2"
        assert "indented" not in cfg  # nested keys are not flat scalars
        assert "no_colon_here" not in cfg

    def test_missing_file_is_empty(self, tmp_path):
        assert runs._read_yaml_scalars(str(tmp_path / "absent.yaml")) == {}


class TestLoadRun:
    def test_from_generation_config(self, make_run):
        info = load_run(make_run("Qwen_level1_triton", level=1))
        assert info.run_name == "Qwen_level1_triton"
        assert info.level == 1
        assert info.model == "test/model"
        assert info.backend == "triton"
        assert info.num_samples == 2

    def test_level_falls_back_to_run_name(self, make_run):
        """No config on disk: the level is still recoverable from `..._level3_...`."""
        info = load_run(make_run("Qwen_level3_triton", level=3, config=""))
        assert info.level == 3
        assert info.run_name == "Qwen_level3_triton"
        assert info.model is None

    def test_trailing_slash_is_stripped(self, make_run):
        assert load_run(make_run("Qwen_level1_triton") + "/").run_name == "Qwen_level1_triton"

    def test_non_numeric_num_samples_is_none(self, make_run):
        run_dir = make_run("r_level1", config="pseudo_level: 1\nnum_samples: many\n")
        assert load_run(run_dir).num_samples is None

    def test_raises_when_level_is_undeterminable(self, make_run):
        with pytest.raises(ValueError, match="cannot determine level"):
            load_run(make_run("no_level_in_this_name", config=""))


class TestIterSamples:
    def test_joins_eval_results_to_kernel_paths(self, make_run):
        run_dir = make_run(eval_results=EVAL_RESULTS)
        samples = list(iter_samples(run_dir))

        assert len(samples) == 2  # the entry without a sample_id is skipped
        first, second = samples

        assert first.problem_id == 2 and first.sample_id == 0
        assert first.correct and first.compiled
        assert first.runtime == 1.0
        assert first.runtime_min == 0.9
        assert first.hardware == "NVIDIA A100-SXM4-80GB"
        assert first.kernel_path == os.path.join(
            run_dir, "level_1_problem_2_sample_0_kernel.py"
        )

        assert not second.correct
        assert second.runtime is None
        assert second.runtime_min is None  # no runtime_stats block
        assert second.error_name == "CUDA error"

    def test_limit_stops_early(self, make_run):
        run_dir = make_run(eval_results=EVAL_RESULTS)
        assert len(list(iter_samples(run_dir, limit=1))) == 1

    def test_load_eval_results(self, make_run):
        assert load_eval_results(make_run(eval_results=EVAL_RESULTS)) == EVAL_RESULTS


class TestReferences:
    def test_filename_and_source(self, fake_kernelbench):
        assert reference_filename(1, 2) == "2_Add.py"
        assert reference_path(1, 2).endswith("level1/2_Add.py")
        assert "get_inputs" in reference_source(1, 2)

    def test_unknown_problem(self, fake_kernelbench):
        assert reference_filename(1, 999) is None
        assert reference_path(1, 999) is None
        assert reference_source(1, 999) is None

    def test_missing_level_directory(self, fake_kernelbench):
        assert reference_filename(9, 1) is None

    def test_indexed_file_that_disappeared(self, fake_kernelbench):
        kb, _ = fake_kernelbench
        assert reference_filename(1, 2) == "2_Add.py"  # populates the cached index
        (kb / "level1" / "2_Add.py").unlink()
        assert reference_source(1, 2) is None


class TestBaselines:
    @pytest.mark.parametrize(
        "hardware,expected",
        [
            ("NVIDIA A100-SXM4-80GB", "A100"),
            ("NVIDIA H100 80GB HBM3", "H100"),
            ("NVIDIA L40S", "A100"),  # unknown hardware defaults to A100
            (None, "A100"),
        ],
    )
    def test_gpu_dir(self, hardware, expected):
        assert runs._gpu_dir(hardware) == expected

    def test_baseline_time(self, fake_kernelbench):
        assert baseline_time(1, 2, "NVIDIA A100-SXM4-80GB")["mean"] == 2.0

    def test_baseline_time_unknown_problem(self, fake_kernelbench):
        assert baseline_time(1, 999) is None

    def test_baseline_time_without_a_timing_file(self, fake_kernelbench):
        """There is no timing/H100/ tree here -- report nothing rather than guess."""
        assert baseline_time(1, 2, "NVIDIA H100 80GB HBM3") is None


class TestSpeedup:
    def test_ratio_of_baseline_to_runtime(self, fake_kernelbench):
        assert speedup(sample(runtime=1.0)) == 2.0  # baseline mean 2.0 ms

    def test_none_when_incorrect(self, fake_kernelbench):
        assert speedup(sample(correct=False)) is None

    def test_none_when_unmeasured(self, fake_kernelbench):
        assert speedup(sample(runtime=None)) is None

    def test_none_without_a_baseline(self, fake_kernelbench):
        assert speedup(sample(problem_id=999)) is None
