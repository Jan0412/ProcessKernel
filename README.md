# ProcessKernel — Quality Signals for LLM-Generated Triton Kernels

Large language models can generate GPU kernels that **pass every correctness test and
are still bad** — they fall back to PyTorch for the hard part, launch a decoy kernel,
or waste memory bandwidth on intermediates a competent author would fuse away. This
repository studies how to *detect* that gap and *close* it, without a human in the loop.

Three ideas are developed and evaluated against
[KernelBench](https://github.com/ScalingIntelligence/KernelBench):

1. **A deterministic, GPU-free static analyzer** (`checker/`) that reads a generated
   kernel and reports, in bytes and microseconds, whether it cheated or is provably slow —
   and, separately, whether the evaluator can load it at all.
2. **A self-refinement loop** (`kernel_gen/`) that feeds those findings — and execution
   feedback — back to the model and asks it to repair its own kernel.
3. **A learned reranker** (`reranker/`) that picks the best of *N* samples so the loop
   spends its budget on the candidate most likely to be correct and fast.

---

## Repository layout

| Path | What it is |
|---|---|
| [`checker/`](checker/README.md) | The static analyzers. Pure-stdlib `ast`, no GPU. **Start here — it has its own detailed README.** |
| &nbsp;&nbsp;`checker/core/` | What both analyzers are built from: the AST front end, the `Finding` vocabulary, and the `Check` / `Analyzer` / `Renderer` base classes. |
| &nbsp;&nbsp;`checker/lint/` | *"Is this good Triton?"* — families F1 (is the kernel real) and F2 (what it wastes). |
| &nbsp;&nbsp;`checker/submission/` | *"Can the evaluator load this?"* — S1.0–S1.3. `compile()`, an entry class, a reachable `forward`, bound module aliases. Answers loadability only, never correctness. |
| [`kernel_gen/`](kernel_gen/) | Kernel generation with vLLM: baseline sampling, execution-feedback rounds, the lint-repair loop, and reranked selection. |
| [`reranker/`](reranker/) | Pointwise / pairwise / listwise reranker that scores candidates and keeps the best. |
| — | Evaluation is **not** in this repo. Runs are scored by the KernelBench checkout at `/path/to/workdir/KernelBench` via `scripts/eval_from_generations.py`. |
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
| **A2** | Best-of-*N* via launch-config sweep | *retired — the `autotune/` package was removed* |
| **A3** | Round-2 re-prompt with **timing** feedback | *retired with A2 (was seeded from its champion)* |
| **A4** | Round-2 re-prompt with **tuning** feedback (full sweep table) | *retired with A2 (was seeded from its champion)* |
| **A5** | **Lint loop**: generate → lint → repair, up to *N* rounds | `kernel_gen.arms.lintloop` |
| — | Reranked best-of-*N* selection | `kernel_gen.generate_kernels_reranked` |

A2–A4 are retired with the sweep. A5's round 0 is the paired baseline for every slot, so
"how many broken samples did it fix, and did it break any that already worked?" is
answerable per-slot rather than across two independent draws.

---

## Quick start

```bash
# Install (Python 3.12, uv)
uv sync

# Lint one generated kernel
uv run python -m checker check runs/<run>/level_1_problem_23_sample_0_kernel.py

# Scan a whole run folder to JSONL
uv run python -m checker scan runs/<run> --out linter_findings.jsonl --workers 32

# Run the generate → lint → repair loop (arm A5)
uv run python -m kernel_gen.arms.lintloop --level 1 --all --rounds 3 --num-samples 10

# Evaluate a run (refined vs. its paired round-0 baseline). RUN_NAME is relative to runs/.
cd /path/to/workdir/KernelBench
sbatch --export=ALL,RUN_NAME=<run>,LEVEL=1,NUM_SAMPLES_PER_PROBLEM=10 \
    slum_scripts/eval_from_generations.sh
sbatch --export=ALL,RUN_NAME=<run>/rounds/round_0,LEVEL=1,NUM_SAMPLES_PER_PROBLEM=10 \
    slum_scripts/eval_from_generations.sh
```

Cluster jobs go through `scripts/*.sh` (SLURM) and `reranker/scripts/*.sh`.

---

## Tests

```bash
uv run --group dev pytest                    # both suites (checker + kernel_gen)
uv run --group dev pytest kernel_gen/tests   # just the generation pipeline
uv run --group dev pytest kernel_gen/tests/unit          # pure functions
uv run --group dev pytest kernel_gen/tests/integration   # cross-module seams
uv run --group dev pytest kernel_gen/tests/properties     # metamorphic (gemtest) + property (Hypothesis)
```

`kernel_gen/tests/` is split into `unit/` (one file per `core/` module), `integration/`
(the seams) and `properties/` (metamorphic via **gemtest** + property-based via
**Hypothesis**). Two quality gates back it, both kept out of the inner-loop `pytest` so
it stays fast:

```bash
# coverage FLOOR -- catches untouched code. kernel_gen/core, checker/core and
# checker/submission are all at 100% line and branch; the gate fails under 95%.
uv run --group dev pytest --cov --cov-branch

# mutation testing -- catches untested BEHAVIOUR, the real anti-alibi bar
uv run --group dev pytest kernel_gen/tests --gremlins --gremlin-targets=kernel_gen/core
uv run --group dev pytest checker/tests --gremlins --gremlin-targets=checker/core,checker/submission
```

The coverage sources in `pyproject.toml` are **dotted module names**, not paths: a path is
silently reported as "module was never imported" and measures nothing.

Mutation testing is what proves the tests actually assert something: **pytest-gremlins**
corrupts `core/` and reports how many mutants the suite kills ("zaps"). Line coverage
says a line ran; only this says a test would have noticed it being wrong. Do **not** pass
`--gremlin-parallel` — it collides with coverage.py's data file and errors out.

Known-but-unfixed data-flow bugs are tracked as strict-xfail tests plus a row in
`checker/tests/KERNEL_GEN_BUGS.md`.

Three whole-corpus sweeps in `scripts/` back claims that a unit test cannot:
`verify_report_parity.py` proves a refactor left `analyze_source` byte-identical over every
shipped kernel, `verify_extraction_parity.py` hashes what the extractor picks from each
captured completion, and `verify_submission_gate.py` re-measures how many lint-clean
kernels the evaluator still cannot load.

> **Run dirs predate KGEN-14.** Kernels already on disk were written when `clean` meant only
> "the linter had nothing to say", so roughly 3.7% of the ones marked clean cannot be
> loaded at all. They are not retro-fixed; new runs stop mislabelling them.

---

## Notes

- `runs/`, `KernelBench/`, `mlruns/`, and SLURM `*.err`/`*.out` logs are **git-ignored
  artifacts** — they are reproduced by the pipeline, not versioned.
- Component-level detail lives in each subdirectory's own README, most notably
  [`checker/README.md`](checker/README.md), which documents every check with
  its failure mode, false-positive guards, and paper references.
