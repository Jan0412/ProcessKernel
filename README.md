# ProcessKernel — Quality Signals for LLM-Generated Triton Kernels

Large language models can generate GPU kernels that **pass every correctness test and
are still bad** — they fall back to PyTorch for the hard part, launch a decoy kernel,
or waste memory bandwidth on intermediates a competent author would fuse away. This
repository studies how to *detect* that gap and *close* it, without a human in the loop.

Three ideas are developed and evaluated against
[KernelBench](https://github.com/ScalingIntelligence/KernelBench):

1. **A deterministic, GPU-free static linter** (`triton_lint/`) that reads a generated
   kernel and reports, in bytes and microseconds, whether it cheated or is provably slow.
2. **A self-refinement loop** (`kernel_gen/`) that feeds those findings — and execution
   feedback — back to the model and asks it to repair its own kernel.
3. **A learned reranker** (`reranker/`) that picks the best of *N* samples so the loop
   spends its budget on the candidate most likely to be correct and fast.

---

## Repository layout

| Path | What it is |
|---|---|
| [`triton_lint/`](triton_lint/README.md) | The static analyzer. Pure-stdlib `ast`, ~1 ms/file, no GPU. **Start here — it has its own detailed README.** |
| [`kernel_gen/`](kernel_gen/) | Kernel generation with vLLM: baseline sampling, execution-feedback rounds, the lint-repair loop, and reranked selection. |
| [`reranker/`](reranker/) | Pointwise / pairwise / listwise reranker that scores candidates and keeps the best. |
| [`autotune/`](autotune/) | Launch-config sweeps, correctness gates, and the eval runner (`autotune.eval_run`). |
| [`timing/`](timing/README.md) | Baseline eager PyTorch runtimes per problem/GPU, used to compute speedups offline. |
| [`scripts/`](scripts/) | SLURM + local driver scripts wiring the pipeline together. |
| [`notebooks/`](notebooks/) | Analysis: reranker eval, diversity vs. correctness, score-outcome, Fast@k. |
| `KernelBench/` | Vendored benchmark (git-ignored; installed via the `pyproject` git source). |
| `runs/` | Generated kernels + eval results (git-ignored artifacts). |

---

## The experiment arms

Every arm is generated with the same model, sampler, and budget so differences are
attributable to the intervention. Round 0 of a loop doubles as the paired baseline.

| Arm | Intervention | Entry point |
|---|---|---|
| **A1** | Single-shot baseline generation | `kernel_gen.generate_kernels_samples` |
| **A2** | Best-of-*N* via launch-config sweep (the champion) | `autotune.sweep` / `scripts/autotune_sweep.sh` |
| **A3** | Round-2 re-prompt with **timing** feedback | `kernel_gen.generate_kernels_feedback --arm timing` |
| **A4** | Round-2 re-prompt with **tuning** feedback (full sweep table) | `kernel_gen.generate_kernels_feedback --arm tuning` |
| **A5** | **Lint loop**: generate → lint → repair, up to *N* rounds | `kernel_gen.arms.lintloop` |
| — | Reranked best-of-*N* selection | `kernel_gen.generate_kernels_reranked` |

Arms A3/A4 are seeded from the *same* A2 champion; A5's round 0 is the paired baseline
for every slot, so "how many broken samples did it fix, and did it break any that
already worked?" is answerable per-slot rather than across two independent draws.

---

## Quick start

```bash
# Install (Python 3.12, uv)
uv sync

# Lint one generated kernel
uv run python -m triton_lint check runs/<run>/level_1_problem_23_sample_0_kernel.py

# Scan a whole run folder to JSONL
uv run python -m triton_lint scan runs/<run> --out linter_findings.jsonl --workers 32

# Run the generate → lint → repair loop (arm A5)
uv run python -m kernel_gen.arms.lintloop --level 1 --all --rounds 3 --num-samples 10

# Evaluate a run (refined vs. its paired round-0 baseline)
uv run python -m autotune.eval_run --run-dir runs/<run>                --level 1
uv run python -m autotune.eval_run --run-dir runs/<run>/rounds/round_0 --level 1
```

Cluster jobs go through `scripts/*.sh` (SLURM) and `reranker/scripts/*.sh`.

---

## Tests

```bash
uv run --group dev pytest triton_lint/tests/
```

---

## Notes

- `runs/`, `KernelBench/`, `mlruns/`, and SLURM `*.err`/`*.out` logs are **git-ignored
  artifacts** — they are reproduced by the pipeline, not versioned.
- Component-level detail lives in each subdirectory's own README, most notably
  [`triton_lint/README.md`](triton_lint/README.md), which documents every check with
  its failure mode, false-positive guards, and paper references.
