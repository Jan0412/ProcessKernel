# Listwise (LambdaRank) Kernel Reranker

Trains the kernel reranker to **rank a problem's candidate Triton kernels** so that
the *fastest correct* kernel is scored highest, the slower-correct ones next, and the
compiled-but-wrong ones last. It does this with a **LambdaRank** loss over per-problem
candidate *lists*, where each candidate's target relevance is graded by its **speedup
over the PyTorch baseline** (the same quantity KernelBench's `fast_p` is built on).

The model stays a *pointwise* cross-encoder — it scores one `(reference architecture,
candidate kernel)` pair at a time and emits one scalar. The "list" exists only to compute
the gradient during training, so at inference you can rank **any number** of candidates;
`list_size` is a training-time knob, not a deployment limit.

---

## 1. How to use it

### 1.1 Run it

Everything (source dataset → speed-graded lists → training) runs from one entrypoint;
the data artifacts are built automatically if missing.

```bash
# Full training (GPU node). Builds data/dataset_listwise.jsonl + lists if absent.
bash reranker/scripts/train_listwise_local.sh

# Smoke test (a few optimizer steps + one eval), e.g. on a small GPU:
bash reranker/scripts/train_listwise_local.sh reranker/configs/listwise_config.yaml \
     train.max_steps=5 train.eval_steps=5

# Override any leaf config value as dotted key=value (CLI wins over the YAML):
bash reranker/scripts/train_listwise_local.sh reranker/configs/listwise_config.yaml \
     listwise.sigma=2.0 listwise.list_size=24 train.lr=5e-6
```

Build **only** the data (CPU, no GPU needed) to inspect the lists first:

```bash
bash reranker/scripts/build_lists.sh reranker/configs/listwise_config.yaml
# -> writes data/dataset_listwise.jsonl, data/lists_train.jsonl,
#    data/lists_val.jsonl, data/lists_splits.json
```

You can also call the modules directly with `uv run` (the scripts just wrap these and set
PYTHONPATH). Run from the repo root:

```bash
uv run python -m reranker.src.data.build_dataset --config reranker/configs/listwise_config.yaml
uv run python -m reranker.src.listwise.lists     --config reranker/configs/listwise_config.yaml
uv run python -m reranker.src.listwise.train     --config reranker/configs/listwise_config.yaml
```

### 1.2 Build the dataset for the lists (before training)

`train.py` builds the data automatically if it's missing, but you should build it
**explicitly first** (it's CPU-only — do it on a login node) so you can inspect the lists
and confirm coverage before spending GPU time. It is a two-step pipeline.

**Prerequisites** (must exist before building):

1. **Evaluation runs** — every folder in `data.run_dirs` must contain
   - `eval_results.json` (per-problem `{sample_id, compiled, correctness, runtime, runtime_stats}`), and
   - the staged kernel sources `level_{L}_problem_{P}_sample_{S}_kernel.py`.
   A run **without** `eval_results.json` is silently skipped (with a `[WARN]`) and contributes nothing.
2. **KernelBench problems** — `KernelBench/` under `data.kernelbench_dir` (for the reference-architecture sources).
3. **Baseline timings** — the JSON at `data.baseline_timing_json` for the **same hardware** your runs were
   timed on (default `../timing/A100/baseline_time_torch.json`). Without it, every correct kernel loses its
   speedup and gets dropped, so the lists end up empty.

**Step 1 — build the source dataset** (joins runs + reference sources + baseline speedup; drops
non-compiling kernels because `negative_mode: compiled_wrong`):

```bash
uv run python -m reranker.src.data.build_dataset --config reranker/configs/listwise_config.yaml
# -> data/dataset_listwise.jsonl
```

Check the printed summary — in particular the **speedup coverage** line, which must show that
your correct kernels actually have a baseline:

```
rows           : 1642
positives      : 280  (17.1%)
dropped (compile-fail negatives): 446
speedup        : 280/280 correct have a baseline (0 correct lack a baseline -> speedup None)
```

**Step 2 — build the lists + split** (fresh problem-level train/val split, one speed-graded list per problem):

```bash
uv run python -m reranker.src.listwise.lists --config reranker/configs/listwise_config.yaml
# -> data/lists_train.jsonl, data/lists_val.jsonl, data/lists_splits.json
```

Check that you got non-empty train lists with a mix of relevances:

```
train: 36 lists (mean 11.0 candidates, 3.1 positives; 46 problems skipped — no positive / too few / all-equal)
val  :  4 lists (mean 12.2 candidates, 4.8 positives; 10 problems skipped ...)
```

Or run both steps at once with the wrapper (uses `uv run` + sets PYTHONPATH):

```bash
bash reranker/scripts/build_lists.sh reranker/configs/listwise_config.yaml
```

**Rebuild when** you change `data.run_dirs`, `negative_mode`, `baseline_timing_json`, or any
`listwise.*` knob that affects list construction (`list_size`, `max_positives`, `speedup_lo/hi`,
`dedup_by_code_hash`, the split/seed fields). Delete the stale artifacts (or just rerun the two
commands — they overwrite) so `train.py` doesn't reuse old lists:

```bash
rm -f reranker/data/dataset_listwise.jsonl reranker/data/lists_*.jsonl reranker/data/lists_splits.json
```

> Note: paths in the config are resolved relative to the project root (`reranker/`), so
> `data/dataset_listwise.jsonl` lives at `reranker/data/dataset_listwise.jsonl`.

### 1.3 Configuration reference

Config is a single nested YAML ([`reranker/configs/listwise_config.yaml`](../../configs/listwise_config.yaml)).
Every leaf is overridable from the CLI as `section.key=value`. Below is **every** parameter
that affects listwise training, grouped by section.

#### `listwise:` — the listwise-specific knobs (most important)

| Key | Default | What it does |
|---|---|---|
| `list_size` | `16` | **L** — the fixed budget of candidates per problem's training list. Caps how much any one problem contributes (balance) and bounds memory (each list runs up to L forward passes). Larger = richer ranking signal per list but more memory. |
| `min_list_size` | `2` | Skip a problem whose deduped candidate pool is smaller than this (nothing to rank). |
| `max_positives` | `6` | Cap on how many correct kernels go into one list, so negatives stay represented. If a problem has more correct kernels, they are randomly sub-sampled (preserving the speed spread). |
| `speedup_lo` | `0.25` | Speedup mapped to relevance-fraction `p = 0` (log2 scale). A correct kernel ≤ `0.25×` baseline grades at the bottom of the correct band. |
| `speedup_hi` | `4.0` | Speedup mapped to `p = 1`. A correct kernel ≥ `4×` baseline grades at the top. Between `lo` and `hi`, `p` interpolates on a **log2** scale (so `1×` → `p≈0.5` with the defaults). |
| `sigma` | `1.0` | Logistic slope `σ` in the LambdaRank pairwise term `softplus(-σ(s_i - s_j))`. Higher `σ` = sharper preference / larger gradients on mis-ordered pairs. |
| `dedup_by_code_hash` | `true` | Drop near-identical kernels (`sha1(kernel_src)`) before building a list, so duplicate low-temperature samples don't eat list slots. |
| `split_ratios` | `[0.85, 0.15]` | Fresh **problem-level** train/val split (no test set; listwise needs none). All candidates of a problem stay in one split. |
| `split_seed` | `42` | RNG seed for the train/val split. |
| `stratify_by_level` | `true` | Stratify the split by KernelBench level so each split covers all levels. |
| `list_seed` | `42` | RNG seed for candidate sampling inside each list. |
| `lists_train_jsonl` | `data/lists_train.jsonl` | Output: one JSON line per training list. |
| `lists_val_jsonl` | `data/lists_val.jsonl` | Output: one JSON line per validation list (used for `eval_list_loss`/`eval_list_pair_acc`). |
| `lists_splits_json` | `data/lists_splits.json` | Output: the `{level:problem_id -> split}` map, reused by the pointwise val set. |

#### `data:` — where candidates and baselines come from

| Key | Default | What it does |
|---|---|---|
| `run_dirs` | (4 level-1 runs) | KernelBench evaluation run folders. Candidates are **pooled across runs** by `(level, problem_id)`. |
| `level` | `[1,1,1,1]` | Per-run-dir KernelBench level (scalar broadcasts to all run dirs). |
| `kernelbench_dir` | `..` | Repo root holding `KernelBench/` (for reference-architecture sources). |
| `dataset_jsonl` | `data/dataset_listwise.jsonl` | Dedicated source dataset so dropping compile-fails + adding `speedup` never clobbers the pointwise/pairwise `data/dataset.jsonl`. |
| `negative_mode` | `compiled_wrong` | **Excludes non-compiling kernels entirely.** Compilation is a cheap deterministic post-generation check, so the reranker only ranks compiling candidates; the only negatives are compiled-but-wrong. |
| `baseline_timing_json` | `../timing/A100/baseline_time_torch.json` | Per-problem PyTorch-eager baseline runtimes (resolved relative to `reranker/`). `build_dataset` joins these to compute `speedup = baseline / kernel_runtime` per correct kernel. **Use the JSON for the hardware your runs were timed on** (A100 here). |

#### `model:`

| Key | Default | What it does |
|---|---|---|
| `base_model` | `Qwen/Qwen3-Reranker-0.6B` | The backbone to fine-tune. |
| `head_type` | `seq_cls` | `seq_cls` = a scalar classification head (trained from scratch). `yes_no_lm` = the native Qwen3-Reranker head, score = `logit("yes") − logit("no")`. The loss is head-agnostic; the head only changes how the per-candidate scalar is produced. |
| `max_length` | `4096` | Max tokens per `(ref, kernel)` sequence. Drives memory (a list = up to `list_size × max_length` tokens). |
| `reserve_ref_tokens` | `1024` | Tokens reserved for the reference architecture; the candidate kernel's tail is truncated to fit. |
| `attn_implementation` | `sdpa` | Attention kernel (`sdpa` / `eager` / `flash_attention_2`). |

#### `train:` — standard HF `TrainingArguments`, with one listwise twist

> **Important:** `per_device_train_batch_size` counts **lists (queries)**, not candidates.
> Each list does up to `list_size` forward passes, so the real per-step sequence count is
> `per_device_train_batch_size × list_size`. Keep the batch at 1–2 and use gradient
> accumulation to reach the effective list-batch size.

| Key | Default | What it does |
|---|---|---|
| `per_device_train_batch_size` | `1` | Number of **lists** per device per step. |
| `gradient_accumulation_steps` | `16` | Accumulate this many list-batches before an optimizer step (effective list batch = product). |
| `per_device_eval_batch_size` | `16` | Pointwise eval batch (single candidates), used for the ranking metrics. |
| `epochs` / `max_steps` | `3` / `-1` | Training length. `max_steps=-1` = full; set a small `max_steps` for a smoke test. |
| `lr` | `1e-5` | Learning rate. |
| `warmup_ratio`, `weight_decay` | `0.01`, `0.05` | Optimizer schedule / regularization. |
| `bf16` / `fp16` | `true` / `false` | Mixed precision (the loss itself is computed in fp32 for stability). |
| `gradient_checkpointing` | `true` | Trade compute for memory — recommended, since a list forward is large. |
| `logging_steps`, `eval_steps`, `save_steps` | `5`, `10`, `10` | Logging / eval / checkpoint cadence. |
| `metric_for_best_model` | `eval_ndcg` | Model selection metric (the real per-problem ranking objective). |
| `greater_is_better` | `true` | Direction for the above. |
| `save_total_limit` | `2` | Keep at most N checkpoints. |
| `output_dir` | `data/checkpoints_listwise` | Checkpoint dir (separate from pointwise/pairwise). |
| `seed`, `dataloader_num_workers` | `42`, `8` | Reproducibility / dataloading. |

#### `mlflow:`

| Key | Default | What it does |
|---|---|---|
| `db_file` | `mlflow.db` | SQLite tracking store (auto-created). |
| `experiment` | `KernelReranker_Listwise` | MLflow experiment name. |
| `run_name` | `null` | `null` → `"{base_model}_listwise_{timestamp}"`. |

### 1.4 Outputs

- `data/dataset_listwise.jsonl` — one row per compiling candidate, with `speedup`, `runtime_min`, `label`, …
- `data/lists_train.jsonl`, `data/lists_val.jsonl` — one JSON line per problem's list:
  `{"level", "problem_id", "candidates": [{"run_name", "sample_id", "rel"}, …]}`
- `data/lists_splits.json` — `{ "1:7": "train", … }`.
- `data/checkpoints_listwise/final/` — best model + tokenizer + `reranker_head.json`, also logged to MLflow.

Metrics logged per eval: pointwise **ranking** metrics on the full val pool
(`eval_ndcg`, `eval_pass_at_1`, `eval_recall_at_1`, `eval_coverage`) plus listwise
`eval_list_loss` and `eval_list_pair_acc` (fraction of `r_i>r_j` pairs the model orders correctly).

---

## 2. What the approach does

### 2.1 Data scope: compiling kernels only
Non-compiling kernels are dropped at dataset-build time (`negative_mode: compiled_wrong`).
Compilation is a cheap deterministic check you run anyway, so the reranker never wastes
capacity learning "does it compile" — it only ever ranks *compiling* candidates, both in
training and at deployment. The only negatives are **compiled-but-wrong** (the hard, meaningful
discrimination).

### 2.2 One speed-graded list per problem
Candidates are pooled across all run folders by `(level, problem_id)` (different models solve
different problems, so pooling unions the scarce positives). For each problem we build **one**
fixed-size list: dedup near-identical kernels, take up to `max_positives` correct kernels, fill
the rest of the `list_size` budget with compiled-but-wrong negatives. One list per problem means
a candidate-rich problem can't dominate the gradient.

### 2.3 Graded relevance from baseline speedup
Each candidate gets a target relevance `r`:

- **Negative** (compiled-but-wrong): `r = 0`.
- **Correct**: `r = 1 + p`, where `p ∈ [0,1]` is the normalized speedup over the per-problem
  PyTorch baseline (`speedup = baseline_runtime / kernel_runtime`), mapped on a log2 scale:

```
p = clip( ( log2(speedup) − log2(speedup_lo) ) / ( log2(speedup_hi) − log2(speedup_lo) ), 0, 1 )
```

So every correct kernel (`r ∈ [1,2]`) outranks every wrong one (`r = 0`), and among the correct
ones, faster ⇒ higher `r`. A correct kernel with **no** baseline speedup is **dropped** (never
guessed), and a problem that loses all its positives that way is skipped — so a genuinely fast
kernel can never be mislabeled as merely-correct.

This makes the training target identical to the benchmark you ultimately care about: rank the
*fastest correct* kernel to the top.

---

## 3. How LambdaRank works (the loss)

LambdaRank turns "produce a ranking that maximizes NDCG" into a pairwise gradient that you can
backprop, by **weighting each pairwise preference by how much swapping the two items would change
NDCG**. We optimize per problem (per list) and average over problems.

### 3.1 Ingredients for one list

A list has candidates `i = 1..n` with model scores `s_i` (one scalar per candidate) and target
relevances `r_i`.

**Gain** (exponential in relevance — rewards getting high-relevance items high):

$$ g_i = 2^{r_i} - 1 $$

**Discount** at a 0-based rank position `ρ` (positions near the top matter far more):

$$ D(\rho) = \frac{1}{\log_2(\rho + 2)} $$

**DCG / IDCG.** DCG of an ordering sums `g · D(rank)` over items in that order; **IDCG** is the
DCG of the *ideal* ordering (items sorted by relevance descending). NDCG = DCG / IDCG ∈ [0,1].

### 3.2 The per-pair NDCG delta

Let `ρ_i` be candidate `i`'s rank under the model's current scores (sort by `s`, best = 0). The
change in NDCG from swapping the rank positions of `i` and `j` (holding everything else) is

$$ |\Delta\text{NDCG}_{ij}| = \frac{\,|g_i - g_j|\;\cdot\;|D(\rho_i) - D(\rho_j)|\,}{\text{IDCG}} $$

This is large when the two items differ a lot in relevance **and** sit where the discount changes
steeply (i.e. near the top). It is the "how much does this pair matter" weight, and is treated as
a **constant** (detached from the gradient — the defining trick of LambdaRank).

### 3.3 The loss

Over all ordered pairs where `i` should outrank `j` (i.e. `r_i > r_j`):

$$
\mathcal{L}_{\text{list}}
= \frac{1}{|P|} \sum_{(i,j)\in P} |\Delta\text{NDCG}_{ij}| \;\cdot\; \log\!\big(1 + e^{-\sigma (s_i - s_j)}\big),
\qquad P = \{(i,j): r_i > r_j\}
$$

The term `log(1 + e^{−σ(s_i − s_j)})` (= `softplus(−σ(s_i−s_j))`) is the RankNet logistic cost: it is
small when `s_i ≫ s_j` (correctly ordered) and grows linearly when `s_i ≪ s_j` (wrongly ordered).
Multiplying by `|ΔNDCG_ij|` makes the optimizer spend its effort where the ranking metric improves
most — overwhelmingly on getting the **top** of the list right, which is exactly "pick the fastest
correct kernel."

The **batch loss** is the mean of `L_list` over the lists in the batch that have at least one valid
pair. Lists with `< 2` candidates, no positive, or all-equal relevance contribute no gradient
(`lambdarank_loss` returns `None`); a fully degenerate batch yields a zero-gradient loss that keeps
the autograd graph intact.

### 3.4 Why the detached ΔNDCG still trains correctly

Differentiating one pair's cost w.r.t. the scores gives the "lambda" gradient

$$ \lambda_{ij} = -\,|\Delta\text{NDCG}_{ij}| \cdot \frac{\sigma}{1 + e^{\sigma(s_i - s_j)}} $$

which pushes `s_i` **up** and `s_j` **down** with a force proportional to both the current ordering
error and the pair's NDCG importance. Summing these per item is the standard LambdaRank update; here
we get it automatically from autograd because only the `softplus` term carries gradient while
`|ΔNDCG_ij|` is a constant weight.

### 3.5 Reference implementation

See [`trainer.py`](trainer.py) → `lambdarank_loss(scores, rels, sigma)` for the exact tensor code
(it builds the `n×n` pair tensors, masks to `r_i > r_j`, weights by the detached `ΔNDCG`, and
normalizes by the number of valid pairs), and `ListwiseTrainer.compute_loss`, which runs one forward
over all candidates of the batch, `torch.split`s the scores back per list via `group_sizes`, and
averages `lambdarank_loss` over the lists.

---

## 4. Files

| File | Responsibility |
|---|---|
| [`lists.py`](lists.py) | Build the fresh train/val split and materialize one speed-graded list per problem. |
| [`dataset.py`](dataset.py) | `ListwiseDataset` (encode each candidate) + `ListwiseCollator` (flatten lists, carry `group_sizes`). |
| [`trainer.py`](trainer.py) | `lambdarank_loss` + `ListwiseTrainer` (loss, collator-swap, listwise eval metrics). |
| [`train.py`](train.py) | End-to-end entrypoint (build-if-missing, train, validate, save, MLflow). |

Reuses, unchanged, from `reranker/src`: the `SequenceEncoder` (encoding), the backbone +
`HeadInfo` (model), `RerankerDataset`/`RerankerCollator` (pointwise eval), `make_compute_metrics`
(ranking metrics), and `build_training_args`/`setup_mlflow`/`RerankerTrainer` (HF + MLflow wiring).

---

## 5. Notes & caveats

- **Inference is count-independent.** The model scores candidates one at a time; `list_size` only
  shapes training. At deployment, score every compiling candidate and take the argmax.
- **Speed signal needs ≥2 correct kernels per problem** to contribute fast-vs-slow ordering; on
  problems with a single correct kernel it only contributes correct-vs-wrong ordering.
- **Baselines are hardware-specific.** Point `data.baseline_timing_json` at the timing JSON for the
  GPU your runs were measured on (A100 here). Coverage is reported by `build_dataset`
  (`X/Y correct have a baseline`).
- **A run with no `eval_results.json` is silently skipped** by `build_dataset` (with a `[WARN]`),
  so it contributes no candidates until its eval results are present.
- **`per_device_train_batch_size` is in lists, not candidates** — see the warning in §1.3.
