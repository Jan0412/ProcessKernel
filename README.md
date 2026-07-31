# ProcessKernel — Quality Signals for LLM-Generated Triton Kernels

Large language models can generate GPU kernels that **pass every correctness test and
are still bad** — they fall back to PyTorch for the hard part, launch a decoy kernel,
or waste memory bandwidth on intermediates a competent author would fuse away. This
repository studies how to *detect* that gap and *close* it, without a human in the loop.

Four ideas are developed and evaluated against
[KernelBench](https://github.com/ScalingIntelligence/KernelBench) and
[KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook):

1. **Deterministic, GPU-free static analysis** (`checker/`) that reads a generated
   kernel and reports, in bytes and microseconds, whether it cheated or is provably slow —
   and, separately, whether the evaluator can load it at all.
2. **A self-refinement loop** (`kernel_gen/arms/lintloop.py`) that feeds those findings
   back to the model and asks it to repair its own kernel.
3. **A learned reranker** (`reranker/`) that picks the best of *N* samples so the loop
   spends its budget on the candidate most likely to be correct and fast.
4. **Per-token confidence traces** (`--trace`) capturing what the model weighed at every
   step of a generation, joined to the linter's findings by line number — training data
   for a process reward model, at no extra generation cost.

---

## Repository layout

| Path | What it is |
|---|---|
| [`checker/`](checker/README.md) | The static analyzers. Pure-stdlib `ast`, no GPU. **Start here — it has its own detailed README.** |
| &nbsp;&nbsp;`checker/core/` | What both analyzers are built from: the AST front end, the `Finding` vocabulary, and the `Check` / `Analyzer` / `Renderer` base classes. |
| &nbsp;&nbsp;`checker/lint/` | *"Is this good Triton?"* — families F1 (is the kernel real) and F2 (what it wastes). |
| &nbsp;&nbsp;`checker/submission/` | *"Can the evaluator load this?"* — S1.0–S1.3. `compile()`, an entry class, a reachable `forward`, bound module aliases. Answers loadability only, never correctness. |
| [`kernel_gen/`](kernel_gen/) | Kernel generation with vLLM. |
| &nbsp;&nbsp;`kernel_gen/core/` | The loop-capable machinery: the `Backend` seam, the two-pass think sampler, prompt builders, completion extraction, the round-major engine, artifact writers, and the per-token trace record. Testable without a GPU against a scripted fake backend. |
| &nbsp;&nbsp;`kernel_gen/arms/` | One module per experiment arm. Currently `lintloop.py` (A5). |
| &nbsp;&nbsp;`kernel_gen/readout.py` | Joins a run's `eval_results.json`, its round-0 baseline's, and `lint_loop.jsonl` into the paired per-slot transition table. |
| &nbsp;&nbsp;`kernel_gen/inspect_trace.py` | Reads a captured trace back: per-token frame, findings joined by line number, the plan-vs-code confidence split. |
| &nbsp;&nbsp;`kernel_gen/convert_kernelbook.py` | Stages KernelBook rows as a KernelBench local level (up-scaling its placeholder `4` shapes). |
| [`reranker/`](reranker/) | Pointwise / pairwise / listwise reranker that scores candidates and keeps the best. |
| — | Evaluation is **not** in this repo. Runs are scored by the KernelBench checkout at `/path/to/workdir/KernelBench` via its `slum_scripts/eval_from_generations.sh`. |
| [`timing/`](timing/README.md) | Baseline eager PyTorch runtimes per problem/GPU, used to compute speedups offline. |
| [`scripts/`](scripts/) | SLURM + local drivers, the staging pre-flight tools, and the whole-corpus verification sweeps. |
| [`notebooks/`](notebooks/) | Analysis: reranker eval, diversity vs. correctness, score-outcome, Fast@k, prompt token stats, speculative decoding. |
| `KernelBench/` | Vendored benchmark and staged level dirs (git-ignored; the package installs via the `pyproject` git source). |
| `runs/` | Generated kernels, traces + eval results (git-ignored artifacts). |

---

## Datasets

| Dataset | How it is addressed | Notes |
|---|---|---|
| KernelBench levels 1–4 | `--level N` | Loaded from HuggingFace. The `code` column is byte-identical to the staged files, so `--ref-dir` is optional here. |
| KernelBook | `--dataset kernelbook --level 6` | ~17.4k rows staged as a KernelBench local level by `scripts/create_kernelbook.sh`. **`--ref-dir` is not optional in practice** — without it the row is re-converted in-process *unscaled*, so the model is prompted about a 4×4 problem and graded on the 2048×2048 one. Too large for one job; shard it over a SLURM array. |

A staged level dir is the pipeline's single source of truth: generation prompts from it,
the linter reads its shapes, and KernelBench's eval scores against it. Pre-flight one with
`scripts/check_level_dir.py` before trusting it, and diff a restage against the previous
one with `scripts/diff_level_dirs.py` — replacing a staged dir is not a refresh, it is a
change of benchmark.

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

A5's repair prompt carries **both** analyzers: the linter's findings normally, and the
submission gate's blocking message instead whenever the file cannot be loaded at all — a
kernel is not `clean` (and does not stop its own loop) until it passes both.

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

# The same on KernelBook, prompting from the staged references eval will score against
uv run python -m kernel_gen.arms.lintloop --dataset kernelbook --level 6 \
    --ref-dir KernelBench/level6 --rows 0-499 --rounds 3

# Add --trace to also capture PRM training data. Changes nothing about what is generated.
uv run python -m kernel_gen.arms.lintloop --level 1 --all --rounds 3 --trace

# No GPU: render round 0's prompt and exit
uv run python -m kernel_gen.arms.lintloop --model x --level 1 --problems 0 --dry-run

# Evaluate a run (refined vs. its paired round-0 baseline). RUN_NAME is relative to runs/.
cd /path/to/workdir/KernelBench
sbatch --export=ALL,RUN_NAME=<run>,LEVEL=1,NUM_SAMPLES_PER_PROBLEM=10 \
    slum_scripts/eval_from_generations.sh
sbatch --export=ALL,RUN_NAME=<run>/rounds/round_0,LEVEL=1,NUM_SAMPLES_PER_PROBLEM=10 \
    slum_scripts/eval_from_generations.sh

# Once both evals exist: the paired transition table (fixed / broken / kept / neither)
uv run python -m kernel_gen.readout --run-dir runs/<run>

# Read a captured trace back
uv run python -m kernel_gen.inspect_trace --run-dir runs/<run>
```

Cluster jobs go through `scripts/*.sh` (SLURM) and `reranker/scripts/*.sh`. The A5 driver
is `scripts/lintloop.sh`, whose knobs are environment variables — `SMOKE=1`, `TRACE=1`,
`THINK_TEMP=0` (single-pass, no plan), `DATASET=kernelbook`, `NUM_SAMPLES`, `ROUNDS` —
each routing to its own output dir so two experiments are never confused. KernelBook runs
in array mode (`--array=0-31%14`), each task taking a balanced slice from
`scripts/shard_ids.py` and writing its own `shard_NN/` run dir, because the JSONL journals
are appended without locking.

### Trace output

`--trace` adds a second output and changes nothing about the first. Alongside the kernels
it writes `traces/round_{r}/`: one `.npz` per attempt holding the token ids and the top-20
alternatives the model weighed at every step, plus an `attempts.jsonl` holding the `## Plan`
prose, the full prompt, the extracted code, the linter's findings with their line numbers,
the plan/code seam offsets, and DeepConf's group-confidence summaries. The numbers were
always computed and always thrown away at generation time, so capture costs no extra GPU.
`traces/` sits where the run-dir globs cannot see it, and eval never reads it.

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
# checker/submission are at 100% line and 99.8% branch; the gate fails under 95%.
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
`kernel_gen/tests/KERNEL_GEN_BUGS.md` (generation pipeline) and `checker/tests/BUGS.md`
(the analyzers). A row there means "found, reproduced, not yet fixed"; rows marked
*(fixed)* carry a regression test that was verified red against the unfixed code.

Five whole-corpus sweeps in `scripts/` back claims that a unit test cannot:

| Sweep | What it settles |
|---|---|
| `verify_report_parity.py` | A refactor left `analyze_source` byte-identical over every shipped kernel. |
| `verify_extraction_parity.py` | Hashes what the extractor picks from each captured completion, so a ranking change's blast radius is a diff, not a guess. |
| `verify_submission_gate.py` | How many lint-clean kernels the evaluator still cannot load (367 of 9,886, 3.7% — the KGEN-14 headline). |
| `verify_rank_blast_radius.py` | Which trajectories ship a different attempt under a changed rank ordering, and that every move is unloadable → loadable. |
| `verify_attribution.py` | How many rounds of the mechanism table are explained by no check at all (29.9% → 0.05% after KGEN-18/22). |

> **Run dirs predate KGEN-14.** Kernels already on disk were written when `clean` meant only
> "the linter had nothing to say", so roughly 3.7% of the ones marked clean cannot be
> loaded at all. They are not retro-fixed; new runs stop mislabelling them.

---

## Notes

- `runs/`, `KernelBench/`, `mlruns/`, and SLURM `*.err`/`*.out` logs are **git-ignored
  artifacts** — they are reproduced by the pipeline, not versioned.
- The linter's own numbers are never the headline. The loop optimizes them by
  construction, so "findings went down" is circular; the claim is made on
  `eval_results.json`, and `lint_loop.jsonl` is read for the mechanism.
- Component-level detail lives in each subdirectory's own README, most notably
  [`checker/README.md`](checker/README.md), which documents every check with
  its failure mode, false-positive guards, and paper references.
