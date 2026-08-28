"""Configuration for the kernel reranker training pipeline.

A single nested dataclass loaded from a YAML file. Leaf values can be overridden
from the CLI with dotted `key=value` pairs, e.g.

    python -m reranker.train --config configs/default.yaml train.epochs=1 model.base_model=foo

Paths in the `data` / `mlflow` sections are resolved relative to the project root
(the directory that contains `configs/`), so the pipeline behaves the same whether
invoked from the project root or from a SLURM submit dir.
"""

from __future__ import annotations

import argparse
import os
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional, Union

import yaml

# Project root = reranker/  (this file is reranker/src/config.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# The one baseline every pipeline grades against. H100 because that is what the runs report
# ("NVIDIA H100 80GB HBM3"); the repo-local timing/H100 is a *different* measurement of the
# same hardware -- 15,823 of the 16,311 shared problems disagree on `mean` -- so naming the
# file once here is what stops two pipelines silently dividing by two different numbers.
BASELINE_TIMING_JSON = (
    "/path/to/workdir/KernelBench/results/timing/H100/baseline_time_torch.json"
)


@dataclass
class DataConfig:
    run_dirs: list[str] = field(default_factory=list)
    # Which lint-loop rounds to read out of each run. null = the run root, i.e. only the
    # kernel each sample finished on. A list expands every shard into rounds/round_R, giving
    # the earlier attempts as extra candidates. Never both: the root kernel is byte-identical
    # to its own final round, so a run read twice would grade one kernel as two candidates.
    rounds: Optional[list[int]] = None
    level: Union[int, list[int]] = 1
    kernelbench_dir: str = ".."
    dataset_jsonl: str = "data/dataset.jsonl"
    splits_json: str = "data/splits.json"
    split_ratios: list[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    split_seed: int = 42
    stratify_by_level: bool = True
    # Which negatives to include in the dataset:
    #   all_negative  -> every non-(compiled & correct) kernel (default)
    #   compiled_wrong -> only kernels that compiled but are incorrect (drop
    #                     compile-failure negatives); positives are kept either way.
    negative_mode: str = "all_negative"
    # Per-problem PyTorch-eager baseline runtimes (KernelBench `timing/<hw>/...`),
    # joined by build_dataset to emit a `speedup = baseline / kernel_runtime` per
    # correct candidate. One path for every pipeline (see PRMConfig) so the ORM and the
    # PRM cannot grade the same speedup against two different GPUs.
    baseline_timing_json: str = BASELINE_TIMING_JSON

    def levels_for_run_dirs(self) -> list[int]:
        """Return a per-run-dir level list, broadcasting a scalar `level`."""
        if isinstance(self.level, list):
            if len(self.level) != len(self.run_dirs):
                raise ValueError(
                    f"data.level list has {len(self.level)} entries but there are "
                    f"{len(self.run_dirs)} run_dirs"
                )
            return [int(x) for x in self.level]
        return [int(self.level)] * len(self.run_dirs)


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen3-Reranker-0.6B"
    max_length: int = 4096
    head_type: str = "seq_cls"  # seq_cls | yes_no_lm
    reserve_ref_tokens: int = 1024
    attn_implementation: str = "eager"


@dataclass
class TrainConfig:
    output_dir: str = "data/checkpoints"
    epochs: int = 3
    lr: float = 1e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    metric_for_best_model: str = "eval_pr_auc"
    greater_is_better: bool = True
    pos_weight: Optional[float] = None
    seed: int = 42
    max_steps: int = -1
    dataloader_num_workers: int = 4


@dataclass
class PairwiseConfig:
    """Pairwise-training settings (used only by reranker.src.pairwise.*).

    The pairwise build reads the same labeled source dataset as the pointwise
    pipeline (``data.dataset_jsonl``) but creates its own fresh problem-level
    train/val split (no test) and materializes (positive, negative) pairs.
    """

    pairs_train_jsonl: str = "data/pairs_train.jsonl"
    pairs_val_jsonl: str = "data/pairs_val.jsonl"
    pairs_splits_json: str = "data/pairs_splits.json"
    # Fresh problem-level split (train / val only; pairwise needs no test set).
    split_ratios: list[float] = field(default_factory=lambda: [0.85, 0.15])
    split_seed: int = 42
    stratify_by_level: bool = True
    pair_mode: str = "compiled_wrong"        # compiled_wrong | all_negative | speed
    max_negatives_per_positive: Optional[int] = None  # cap partners per anchor (None = full cross product)
    max_pairs_per_problem: Optional[int] = None  # cap each problem's total pairs (None = uncapped)
    dedup_by_code_hash: bool = True  # drop duplicate kernel sources before pairing (mirrors listwise)
    loss_type: str = "logistic"              # logistic | margin
    margin: float = 1.0
    pair_seed: int = 42
    # `speed` pair_mode (fast_p): grade compiling kernels like listwise
    # (wrong -> 0, correct -> 1 + speed_p) and pair every rel gap >= min_rel_gap.
    speedup_lo: float = 0.25      # speedup mapped to p=0 (log2 lower bound)
    speedup_hi: float = 4.0       # speedup mapped to p=1 (log2 upper bound)
    speed_quant: float = 0.0      # deadband: snap p to this grid (0 = off) so sub-noise
                                  # speedup differences grade equally -> no spurious pair
    min_rel_gap: float = 0.0      # min relevance difference to form a pair
    weighted_loss: bool = False   # optional: SOFT weighting via group-split + alpha
                                  # (the pairwise analogue of listwise's grouped ΔNDCG)
    loss_alpha: float = 0.5       # weight of correctness vs speed pairs in the loss
                                  # (0.5 = equal; lower pushes harder on fast-vs-slow).
                                  # Only used when weighted_loss is on.


@dataclass
class ListwiseConfig:
    """Listwise (LambdaRank) training settings (used only by reranker.src.listwise.*).

    Reads the same labeled source dataset as the pointwise pipeline
    (``data.dataset_jsonl``), builds its own fresh problem-level train/val split
    (no test), and materializes one fixed-size, speed-graded candidate *list* per
    eligible problem. Relevance: negatives (compiled-but-wrong) get 0; correct
    kernels get ``1 + p`` where ``p`` is the normalized speedup over the per-problem
    PyTorch baseline (``data.baseline_timing_json``). Non-compiling kernels are
    excluded upstream via ``data.negative_mode = compiled_wrong``.
    """

    lists_train_jsonl: str = "data/lists_train.jsonl"
    lists_val_jsonl: str = "data/lists_val.jsonl"
    lists_splits_json: str = "data/lists_splits.json"
    # Fresh problem-level split (train / val only; listwise needs no test set).
    split_ratios: list[float] = field(default_factory=lambda: [0.85, 0.15])
    split_seed: int = 42
    stratify_by_level: bool = True
    list_size: int = 16          # L: fixed budget of candidates per problem
    min_list_size: int = 2       # skip problems with fewer (deduped) candidates
    max_positives: int = 10      # cap positives per list (spread-preserving subsample)
    max_negatives: int = 6       # cap negatives per list so speed pairs aren't drowned
    min_positives: int = 1       # skip problems with fewer positives (guardrail; 1 = keep all)
    speedup_lo: float = 0.25     # speedup mapped to p=0 (log2 lower bound)
    speedup_hi: float = 2.5      # speedup mapped to p=1 (log2 upper bound; ~p95 of data)
    speedup_stat: str = "mean"   # which dataset speedup grades the lists:
                                 #   mean -> `speedup` (KernelBench fast_p convention)
                                 #   min  -> `speedup_min` (noise-robust min/min timing)
    speed_quant: float = 0.0     # deadband: snap p to this grid (0 = off) so sub-noise
                                 # speedup differences don't create spurious ranking pairs
    dedup_by_code_hash: bool = True
    sigma: float = 1.0           # logistic slope in the LambdaRank loss
    loss_alpha: float = 0.5      # weight of correctness vs speed pairs in the loss
                                 # (0.5 = equal; lower pushes harder on fast-vs-slow)
    speed_gap_eval: float = 0.25  # min rel gap for the eval_speed_pair_acc_big metric
                                  # (speed-pair accuracy on clearly-separated pairs only)
    list_seed: int = 42


@dataclass
class PRMConfig:
    """Process-reward-model labeling build (``reranker.src.prm.*``); see prm_plan/PLAN.md §7.

    Read by `prm/build.py`, `prm/splits.py` and `prm/stats.py` only — the pointwise,
    pairwise and listwise pipelines never look at this section.
    """

    run_dirs: list[str] = field(default_factory=list)
    rounds: list[int] = field(default_factory=lambda: [0, 1, 2])
    baseline_timing_json: str = BASELINE_TIMING_JSON

    label_mode: str = "graded"    # graded | binary  — validated by targets.target_for
    speedup_stat: str = "min"     # min | mean
    speedup_lo: float = 0.2       # speedup mapped to p=0 -> target 0.50
    speedup_hi: float = 4.0       # speedup mapped to p=1 -> target 1.00
    speed_quant: float = 0.1      # snap p to this grid (0 = off): 11 target values, not a continuum

    prose_lines_per_chunk: int = 1
    code_steps_per_chunk: int = 1
    min_frac: float = 0.0         # row filter over cuts, applied in build.py; 0 = unfiltered

    tokenizer: str = "Qwen/Qwen3-Reranker-4B"
    max_length: int = 16384       # tokens over prompt + raw; over-length samples are dropped

    # Own split knobs rather than data.split_*: a prm_config.yaml has no `data` section,
    # so borrowing them would tie this build to a section it never otherwise reads.
    split_ratios: list[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    split_seed: int = 42

    out_dir: str = "data/prm"
    num_workers: int = 8
    max_shards: Optional[int] = None  # smoke builds only: cap shards per run (None = all)

    def validate(self) -> None:
        """Fail before a build starts rather than partway through writing one."""
        # `_coerce` has no list case, so `prm.rounds=[0]` from the CLI arrives as the
        # *string* "[0]" and would iterate as characters. List values belong in a file.
        for name in ("run_dirs", "rounds"):
            value = getattr(self, name)
            if not isinstance(value, list) or not value:
                raise ValueError(f"prm.{name} must be a non-empty list, got {value!r}")
        if not all(isinstance(d, str) for d in self.run_dirs):
            raise ValueError(f"prm.run_dirs must be a list of paths, got {self.run_dirs!r}")
        if not all(isinstance(r, int) and not isinstance(r, bool) for r in self.rounds):
            raise ValueError(f"prm.rounds must be a list of ints, got {self.rounds!r}")
        if self.prose_lines_per_chunk < 1 or self.code_steps_per_chunk < 1:
            raise ValueError(
                "prm chunk sizes must be >= 1, got "
                f"{self.prose_lines_per_chunk=} {self.code_steps_per_chunk=}"
            )
        if not 0 <= self.min_frac < 1:
            raise ValueError(f"prm.min_frac must be in [0, 1), got {self.min_frac}")
        if self.max_length < 1:
            raise ValueError(f"prm.max_length must be >= 1, got {self.max_length}")
        if self.num_workers < 1:
            raise ValueError(f"prm.num_workers must be >= 1, got {self.num_workers}")
        # Three, because _split_ids gives the remainder to test. Non-negative, or n_train
        # runs past the end of the list and every problem silently lands in train.
        ratios = self.split_ratios
        if (
            not isinstance(ratios, list)
            or len(ratios) != 3
            # bool is an int, and YAML reads `[yes, no, no]` as one -- same trap as rounds.
            or not all(
                isinstance(r, (int, float)) and not isinstance(r, bool) and r >= 0
                for r in ratios
            )
            or abs(sum(ratios) - 1.0) > 1e-6
        ):
            raise ValueError(
                f"prm.split_ratios must be three non-negative ratios summing to 1, got {ratios!r}"
            )
        # int, not just >= 1: a float slices no list, and `prm.max_shards=1e3` is a float.
        if self.max_shards is not None and (
            not isinstance(self.max_shards, int) or self.max_shards < 1
        ):
            raise ValueError(f"prm.max_shards must be an int >= 1 or null, got {self.max_shards!r}")
        # The knobs targets.py grades on, checked here so a bad one cannot first surface
        # inside a pool worker with part files already on disk. Imported at call time, not
        # at module scope: this module must stay a leaf, or the day data/labels.py wants
        # _resolve from it the cycle config -> prm.targets -> data.labels -> config closes
        # and every entry point dies on import.
        from reranker.src.prm.targets import check_knobs

        check_knobs(
            self.label_mode, self.speedup_stat, self.speedup_lo, self.speedup_hi, self.speed_quant
        )
        # Absent, the build grades nothing and writes a training set that is all zeros.
        baseline = _resolve(self.baseline_timing_json)
        if not os.path.isfile(baseline):
            raise ValueError(f"prm.baseline_timing_json is not a file: {baseline}")


# The two prefix sources and the three train-set selection modes (PLAN_v2 §6). Here rather
# than in prm/rollout/prefixes.py because `validate()` gates on them and this module must stay a
# leaf: prefixes.py imports config, so config importing prefixes would close the cycle.
CUT, BEAM = "cut", "beam"
SOURCES = (CUT, BEAM)
RANDOM, ENTROPY, PRM_SPREAD = "random", "entropy", "prm_spread"
SELECTIONS = (RANDOM, ENTROPY, PRM_SPREAD)


@dataclass
class PRMRolloutConfig:
    """The v2 rollout campaign (``reranker.src.prm.rollout.*``); see prm_plan/PLAN_v2.md §7.

    v1 measures a prefix by inheriting its completion's final label. This measures it: cut a
    prefix, generate ``K`` continuations, evaluate each, and let ``V̂`` be what comes back.
    Reads v1's parts and splits; writes its own ``out_dir``.
    """

    # --- inputs ---------------------------------------------------------------------
    # prm_v2, not prm: the older build predates §8 and its rows carry no system_prompt_sha1,
    # so job A reads it happily and job B has nothing to resolve a prompt from.
    parts_glob: str = "data/prm_v2/parts/*.jsonl"
    splits_json: str = "data/prm_v2/splits.json"
    baseline_timing_json: str = BASELINE_TIMING_JSON
    # {run_name: short tag}. Also the run *filter*: one parts dir holds every run v1 built,
    # and a campaign takes one of them. The tag is in `list_key`, so two runs sharing one
    # would merge their lists and break N2.
    run_tags: dict[str, str] = field(default_factory=dict)

    # --- prefix selection -----------------------------------------------------------
    source: str = CUT                  # cut | beam  — beam is stage 2, not built (§11)
    rounds: list[int] = field(default_factory=lambda: [0, 1, 2])
    depths_per_group: int = 4          # how many cut depths per (run, level, problem, round)
    min_rel_depth: float = 0.1         # below it every prefix of a problem shares the base rate
    max_rel_depth: float = 0.9         # above it the outcome is decided and nothing is learnable
    min_list_size: int = 2             # a list of one has no pair; drop it before it is paid for
    max_list_size: int = 8             # the ceiling: 25-sample groups are ~5x the §9 eval budget
    train_selection: str = RANDOM      # random | entropy | prm_spread — only random is built (§6)
    prm_v1_checkpoint: Optional[str] = None   # required only by prm_spread
    select_seed: int = 42
    # val_selection is NOT a knob -- N3 fixes it to random in prefixes.py itself.

    # --- rollouts -------------------------------------------------------------------
    K: int = 5
    min_rollouts: int = 3              # below this many surviving rollouts, V̂ is not measured
    temperature: float = 0.6
    # 0 means the source run sampled in ONE pass, exactly as lintloop.py reads it
    # (`args.think_temperature if args.think_temperature > 0 else None`). Ask `two_pass`,
    # never `think_temperature is not None`: this is a float and 0.0 is not None, so that
    # test reads a single-pass run as two-pass and prefills a "## Plan" it never had.
    think_temperature: float = 1.0
    max_new_tokens: int = 16384
    # The rest of the source run's sampler. Defaults are the flags' own off-positions, so a
    # run generated before they existed -- the kb6 corpus carries none of these keys --
    # continues to match. A v6 run sets all three from its generation_config.yaml.
    enable_thinking: bool = False
    top_p: float = 1.0
    top_k: int = 0
    gen_model: str = "openai/gpt-oss-120b"    # must match the source run's model
    max_num_seqs: int = 64
    max_model_len: int = 40960
    # How much of a unit job B hands one generate() call. A unit is not a batch: measured,
    # one is 14,005 prefixes -> 70,025 rollouts, ~770 MB of prompt strings held with nothing
    # written until they all return. Only memory and checkpointing, never what is sampled.
    prefixes_per_batch: int = 256

    # --- eval -----------------------------------------------------------------------
    eval_runs_dir: str = "/path/to/workdir/KernelBench/runs"
    eval_run_name: str = "prm_rollout_v1"     # shard dirs: {eval_run_name}_s00 ...
    eval_shards: int = 4               # sized so each eval job finishes inside its wall clock
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    eval_timeout: int = 300
    dedup_by_code_sha1: bool = True

    # --- targets (same knobs and same functions as v1) -------------------------------
    label_mode: str = "graded"
    speedup_stat: str = "min"
    speedup_lo: float = 0.2
    speedup_hi: float = 4.0
    speed_quant: float = 0.1

    # --- scoring (rank_eval only; no training settings live here) --------------------
    base_model: str = "Qwen/Qwen3-Reranker-4B"
    max_length: int = 16384
    depth_buckets: int = 4             # how many rel_depth bands the reports slice into

    # --- v3: ORM-imputed labelling ---------------------------------------------------
    # One per campaign, never mixed: level 6 has no evals and is imputed; level 1 is
    # fully evaluated and is measured.
    label_source: str = "measured"

    orm_checkpoint: Optional[str] = None
    # The ORM's own training lists. Their (run, problem, sample) keys are the kernels the
    # curve must NOT be fit on -- a memorized score is sharper than a fresh one.
    orm_lists_glob: Optional[str] = None
    orm_max_length: int = 6144            # listwise_base.yaml -- NOT self.max_length
    orm_reserve_ref_tokens: int = 1024
    orm_batch_size: int = 16

    # Fit the curve on the memorized anchors too. Off by default; on where excluding them
    # would leave a fit sample selected on the label. See fit_isotonic.
    curve_fit_orm_seen: bool = False

    curve_bins: int = 40
    curve_iters: int = 16
    offset_kappa: Union[str, float] = "auto"  # a number forces the fixed n/(n+kappa) form
    offset_clamp: float = 4.0             # ORM-score units
    use_anchors: bool = True
    anchor_rounds: list[int] = field(default_factory=lambda: [0])
    # Drop every list of a problem whose anchors all failed -- the ORM invented its order.
    # Off by default: scored on level 1 it also costs 16% of the good lists at level 6's
    # anchor density. See lists.build_lists.
    drop_dead_problems: bool = False
    # Anchors are re-graded here, not at build time: graded_target takes the speedup ratio
    # and v1 rows store speedup_min. 0.0 is what the ORM trained under.
    calib_speed_quant: float = 0.0

    out_dir: str = "data/prm_rollout"
    num_workers: int = 8

    @property
    def two_pass(self) -> bool:
        """Did the source run split the plan and the code into two calls?

        The one place the 0-means-single-pass convention is spelled out, so a caller cannot
        reinvent it as `is not None` and get the answer backwards on a v6 run.
        """
        return self.think_temperature > 0

    def validate(self) -> None:
        """Fail before a campaign starts rather than partway through one.

        Everything here is cheap and local. The knobs that can only be checked against the
        corpus -- that a tagged run has rows, that a problem has a split -- raise in
        `prefixes.py`, where the corpus is.
        """
        # v3: checked first so a bad label_source/offset/quant fails on its own knob, not
        # on run_tags being empty by default.
        if self.label_source not in ("measured", "imputed"):
            raise ValueError(f"label_source must be measured|imputed, got {self.label_source!r}")
        if self.label_source == "imputed" and not self.orm_checkpoint:
            raise ValueError("label_source=imputed needs orm_checkpoint")
        if self.offset_kappa != "auto" and float(self.offset_kappa) <= 0:
            raise ValueError(f"offset_kappa must be 'auto' or > 0, got {self.offset_kappa!r}")
        if not 0 <= self.calib_speed_quant < 1:
            raise ValueError(f"calib_speed_quant out of range: {self.calib_speed_quant}")

        # The same pair lintloop.py rejects at startup, rejected here for the same reason:
        # both knobs open the assistant turn, so together the plan and the code are written
        # inside a <think> block the model never closes. No source run can have been
        # generated this way, so no rollout may continue one this way either.
        if self.enable_thinking and self.two_pass:
            raise ValueError(
                f"prm_rollout.enable_thinking with think_temperature="
                f"{self.think_temperature} is a regime lintloop refuses to generate in -- "
                "a native-thinking run is single-pass, so set think_temperature: 0"
            )
        if not 0 < self.top_p <= 1:
            raise ValueError(f"prm_rollout.top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"prm_rollout.top_k must be >= 0 (0 is off), got {self.top_k}")

        # `_coerce` has no list case, so `prm_rollout.rounds=[0]` from the CLI arrives as
        # the *string* "[0]" and would iterate as characters. List values belong in a file.
        if not isinstance(self.rounds, list) or not self.rounds:
            raise ValueError(f"prm_rollout.rounds must be a non-empty list, got {self.rounds!r}")
        # bool is an int, and YAML reads `[yes]` as one -- the same trap PRMConfig hits.
        if not all(isinstance(r, int) and not isinstance(r, bool) for r in self.rounds):
            raise ValueError(f"prm_rollout.rounds must be a list of ints, got {self.rounds!r}")

        tags = self.run_tags
        if not isinstance(tags, dict) or not tags:
            raise ValueError(
                f"prm_rollout.run_tags must name at least one run, got {tags!r} -- it is "
                "the run filter as well as the tag, so an empty one builds nothing"
            )
        if not all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in tags.items()):
            raise ValueError(f"prm_rollout.run_tags must map run name -> tag, both non-empty "
                             f"strings, got {tags!r}")
        # The tag opens `list_key`, so two runs under one tag put samples from different
        # generations into the same list -- N2, and nothing downstream could see it.
        if len(set(tags.values())) != len(tags):
            raise ValueError(f"prm_rollout.run_tags must be unique per run, got {tags!r}")

        # Two, not one: a list of one candidate contains no pair, so it trains nothing while
        # costing K evals. The ceiling is what keeps a 25-sample group inside the budget.
        if self.min_list_size < 2:
            raise ValueError(f"prm_rollout.min_list_size must be >= 2, got {self.min_list_size}")
        if self.max_list_size < self.min_list_size:
            raise ValueError(
                f"prm_rollout.max_list_size ({self.max_list_size}) is below min_list_size "
                f"({self.min_list_size}): every list would be capped under the floor and dropped"
            )
        # Strict at both ends: rel_depth is cut_index/n_cuts_total, so it reaches (n-1)/n and
        # never 1.0, and a window of zero width admits no cut at all.
        if not 0 <= self.min_rel_depth < self.max_rel_depth < 1:
            raise ValueError(
                "prm_rollout needs 0 <= min_rel_depth < max_rel_depth < 1, got "
                f"{self.min_rel_depth} .. {self.max_rel_depth}"
            )
        for name in (
            "depths_per_group", "K", "min_rollouts", "max_new_tokens", "max_num_seqs",
            "max_model_len", "prefixes_per_batch", "eval_shards", "num_correct_trials",
            "num_perf_trials", "eval_timeout", "max_length", "depth_buckets", "num_workers",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"prm_rollout.{name} must be >= 1, got {getattr(self, name)}")
        if self.min_rollouts > self.K:
            raise ValueError(
                f"prm_rollout.min_rollouts ({self.min_rollouts}) exceeds K ({self.K}): no "
                "prefix could ever clear it and every list would be dropped after being paid for"
            )

        if self.source not in SOURCES:
            raise ValueError(f"prm_rollout.source must be one of {SOURCES}, got {self.source!r}")
        if self.train_selection not in SELECTIONS:
            raise ValueError(
                f"prm_rollout.train_selection must be one of {SELECTIONS}, "
                f"got {self.train_selection!r}"
            )
        if self.train_selection == PRM_SPREAD and not self.prm_v1_checkpoint:
            raise ValueError(
                "prm_rollout.prm_v1_checkpoint is required when train_selection is "
                f"{PRM_SPREAD}: there is nothing to score the candidates with"
            )

        # The same gate v1 uses, imported at call time so this module stays a leaf.
        from reranker.src.prm.targets import check_knobs

        check_knobs(
            self.label_mode, self.speedup_stat, self.speedup_lo, self.speedup_hi, self.speed_quant
        )
        baseline = _resolve(self.baseline_timing_json)
        if not os.path.isfile(baseline):
            raise ValueError(f"prm_rollout.baseline_timing_json is not a file: {baseline}")


@dataclass
class PRMTrainConfig:
    """PRM listwise training (``reranker.src.prm.rollout.train``); see prm_plan/PLAN_TRAINER.md §6.

    The lists, the prefixes, the v1 parts, the encoder's `max_length` and the depth bands are
    read from `prm_rollout`; the backbone from `model`; the HF arguments from `train`. A second
    copy of `out_dir` here would be a way for the trainer to read a different campaign than the
    one job D wrote.
    """

    sigma: float = 1.0        # logistic slope, straight into lambdarank_loss
    loss_alpha: float = 0.5   # correctness-vs-speed pair weighting, straight into lambdarank_loss

    # How many val lists an eval scores. 0 = every one of them, so adding this knob moved no
    # existing run's number. A campaign's val split is ~26k lists at ~3.2 prefixes each, and
    # the eval runs every `eval_steps` -- uncapped that is ~1h per eval against ~14h of
    # training, i.e. the measurement costs more than the thing measured.
    eval_max_lists: int = 0
    # Seeded because the cap has to draw the SAME lists in every arm of a backbone comparison:
    # two runs scored on different subsets are not comparable, and nothing in the numbers
    # would say so.
    eval_subsample_seed: int = 42

    def validate(self) -> None:
        # softplus(-sigma * (s_i - s_j)) is constant at sigma=0 whatever the model does, so the
        # loss stops depending on the scores and training is a no-op that still logs a curve.
        if self.sigma <= 0:
            raise ValueError(f"prm_train.sigma must be > 0, got {self.sigma}")
        # Outside [0, 1] one of the two pair groups carries a negative weight, and the loss is
        # then minimized by ranking that group wrong.
        if not 0.0 <= self.loss_alpha <= 1.0:
            raise ValueError(f"prm_train.loss_alpha must be in [0, 1], got {self.loss_alpha}")
        if self.eval_max_lists < 0:
            raise ValueError(
                f"prm_train.eval_max_lists must be >= 0 (0 = every list), got "
                f"{self.eval_max_lists}"
            )


TOKENS, CUTS = "tokens", "cuts"
SEL_PRM, SEL_RANDOM = "prm", "random"


@dataclass
class PRMSearchConfig:
    """PRM-steered beam search over half-written generations; the ORM picks the winner.

    Generation knobs are NOT here. `gen_model`, `temperature`, `think_temperature`,
    `max_new_tokens`, `max_length` and `base_model` are read from `prm_rollout`, so a search
    config `_base:`-inherits the rollout config the PRM was trained against and cannot drift
    from it.
    """

    beam_width: int = 4                 # live candidates kept per step
    expand: int = 4                     # children per survivor; decode cost is expand x beam_width
    max_steps: int = 32

    advance: str = TOKENS               # tokens | cuts
    segment_max_tokens: int = 256       # the per-step cap, and the `cuts` policy's generous budget
    # The chunker's granularity for THIS search, independent of the build's -- and under
    # `cuts` also the stride: one step generates exactly one chunk, so these two knobs alone
    # say how big a step is. `tokens` wants them fine (snap-back is then under a line).
    prose_lines_per_chunk: int = 1
    code_steps_per_chunk: int = 1
    score_at_cut: bool = True           # False scores the raw mid-line text instead

    selector: str = SEL_PRM             # prm | random -- random is the ablation control
    min_distinct_parents: int = 2       # 0/1 disables the anti-collapse floor
    select_seed: int = 7

    prm_checkpoint: str = ""
    orm_checkpoint: str = ""
    prm_batch_size: int = 8
    orm_batch_size: int = 8

    # Below vLLM's own 0.92 default: the PRM stays resident for the whole run and the ORM
    # joins it at the end, ~1.2 GB each in bf16, and vLLM measures its budget against TOTAL
    # VRAM rather than free VRAM -- so at 0.92 the two scorers and their activations have to
    # fit in what is left of 80 GB after 73.6.
    gpu_memory_utilization: float = 0.85

    # Which problems to solve. `prm_rollout` has no dataset knobs -- it reads an existing
    # corpus off disk, where this generates a new one -- so these mirror cli.add_dataset_args
    # and are passed straight to sources.load_problems.
    dataset: str = "kernelbench"        # kernelbench | kernelbook
    dataset_name: Optional[str] = None  # None = cli.DATASET_DEFAULTS[dataset]
    level: int = 6
    problems: Optional[str] = None      # '23' | '1-49' | '1,5,10'; None = every row
    ref_dir: Optional[str] = None       # a staged level dir; overrides the row source

    # Which generation rounds the search drives. [0] today because the PRM is trained on round
    # 0 only; the searcher itself is round-agnostic, so this widens without a code change.
    rounds: list[int] = field(default_factory=lambda: [0])
    out_dir: str = "data/prm_search"
    write_pool: bool = True
    keep_pruned: bool = False           # diagnostic: finish pruned candidates to measure regret

    def validate(self) -> None:
        for name in ("beam_width", "expand", "max_steps", "segment_max_tokens",
                     "prose_lines_per_chunk", "code_steps_per_chunk", "prm_batch_size",
                     "orm_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"prm_search.{name} must be >= 1, got {getattr(self, name)}")
        if not 0.0 < self.gpu_memory_utilization < 1.0:
            raise ValueError("prm_search.gpu_memory_utilization must be in (0, 1), got "
                             f"{self.gpu_memory_utilization!r}")
        if self.advance not in (TOKENS, CUTS):
            raise ValueError(
                f"prm_search.advance must be {TOKENS!r} or {CUTS!r}, got {self.advance!r}"
            )
        if self.selector not in (SEL_PRM, SEL_RANDOM):
            raise ValueError(
                f"prm_search.selector must be {SEL_PRM!r} or {SEL_RANDOM!r}, got {self.selector!r}"
            )
        if self.dataset not in ("kernelbench", "kernelbook"):
            raise ValueError(f"prm_search.dataset must be kernelbench|kernelbook, got {self.dataset!r}")
        if self.level < 1:
            raise ValueError(f"prm_search.level must be >= 1, got {self.level}")
        if self.min_distinct_parents > self.beam_width:
            raise ValueError(
                f"prm_search.min_distinct_parents ({self.min_distinct_parents}) exceeds "
                f"beam_width ({self.beam_width}) -- unsatisfiable"
            )
        if self.selector == SEL_PRM and not self.prm_checkpoint:
            raise ValueError("prm_search.selector='prm' needs prm_search.prm_checkpoint")
        # `_coerce` has no list case, so `prm_search.rounds=[0]` from the CLI arrives as the
        # string "[0]" and would iterate as characters. List values belong in a file.
        if not isinstance(self.rounds, list) or not self.rounds:
            raise ValueError(f"prm_search.rounds must be a non-empty list, got {self.rounds!r}")
        if not all(isinstance(r, int) and not isinstance(r, bool) for r in self.rounds):
            raise ValueError(f"prm_search.rounds must be a list of ints, got {self.rounds!r}")


@dataclass
class MLflowConfig:
    db_file: str = "mlflow.db"
    experiment: str = "KernelReranker"
    run_name: Optional[str] = None

    def tracking_uri(self) -> str:
        """Return a SQLite URI; MLflow auto-creates the DB on first use."""
        return "sqlite:///" + _resolve(self.db_file)


@dataclass
class RerankerConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    pairwise: PairwiseConfig = field(default_factory=PairwiseConfig)
    listwise: ListwiseConfig = field(default_factory=ListwiseConfig)
    prm: PRMConfig = field(default_factory=PRMConfig)
    prm_rollout: PRMRolloutConfig = field(default_factory=PRMRolloutConfig)
    prm_train: PRMTrainConfig = field(default_factory=PRMTrainConfig)
    prm_search: PRMSearchConfig = field(default_factory=PRMSearchConfig)


def check_prm_budgets(cfg: RerankerConfig) -> list[str]:
    """Cross the three token budgets against their models, and refuse a crossed backbone.

    A section's own `validate()` cannot do this: it sees only itself, and every one of these
    facts spans two sections (ARCHITECTURE S4). Returns the lines to log rather than printing,
    so a caller decides where they go and a test can read them.
    """
    prm_name, orm_name = cfg.prm_rollout.base_model, cfg.model.base_model
    if prm_name != orm_name:
        raise ValueError(
            f"prm_rollout.base_model ({prm_name!r}) and model.base_model ({orm_name!r}) name "
            "different backbones. They are one model with two names -- the first is "
            "rank_eval's tokenizer fallback, the second is what build_backbone loads -- so "
            "crossed, training and job E would disagree about the tokenizer with every number "
            "still computing. Set model.base_model in the PRM training config."
        )
    return [
        f"gen  max_new_tokens={cfg.prm_rollout.max_new_tokens:<6} model={cfg.prm_rollout.gen_model}",
        f"prm  max_length={cfg.prm_rollout.max_length:<10} model={prm_name}",
        # Printed because it is the budget this job must NOT pick up: it bounds the ORM's
        # SequenceEncoder over ref + kernel, not the PRM's prompt + raw[:cut_char].
        f"orm  max_length={cfg.model.max_length:<10} model={orm_name}  (unused here)",
    ]


def _resolve(path: str) -> str:
    """Resolve a possibly-relative path against the project root."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def _from_dict(cls, data: dict, section: str = "") -> Any:
    """Recursively build a (nested) dataclass from a plain dict.

    Unknown keys raise instead of being silently dropped — a typo in the YAML
    (e.g. ``speedup_qant``) would otherwise leave the default in place with no
    warning, so the config on disk and the config actually used would diverge.
    """
    # `from __future__ import annotations` makes field types strings — resolve them.
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = section or cls.__name__
        raise KeyError(
            f"Unknown config key(s) in '{where}': {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(value, dict):
            child = f"{section}.{f.name}" if section else f.name
            kwargs[f.name] = _from_dict(ftype, value, section=child)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _coerce(raw: str) -> Any:
    """Coerce a CLI override string to bool/int/float/None, falling back to str."""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_override(cfg: RerankerConfig, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"Unknown config key: {dotted_key}")
    setattr(obj, leaf, value)


def _merge(base: dict, over: dict) -> dict:
    """``over`` onto ``base``, recursing into dicts. A list replaces, never appends."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: str, seen: Optional[list[str]] = None) -> dict:
    """Load a config YAML, first applying any ``_base:`` it inherits from.

    The dataset variants differ in a handful of paths but must share every training
    knob -- copied instead, one of six files silently drifts and the runs stop being
    comparable. ``_base`` is resolved relative to the file that names it.
    """
    path = os.path.abspath(path)
    seen = (seen or []) + [path]
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("_base", None)
    if base is None:
        return raw
    base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
    if os.path.abspath(base_path) in seen:
        raise ValueError(f"_base cycle: {' -> '.join(seen + [os.path.abspath(base_path)])}")
    return _merge(_load_yaml(base_path, seen), raw)


def load_config(argv: Optional[list[str]] = None) -> RerankerConfig:
    """Parse `--config path` plus dotted `key=value` overrides into a RerankerConfig."""
    parser = argparse.ArgumentParser(description="Kernel reranker pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, overrides = parser.parse_known_args(argv)

    cfg = _from_dict(RerankerConfig, _load_yaml(args.config))

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not in key=value form")
        key, _, val = item.partition("=")
        _apply_override(cfg, key.strip(), _coerce(val.strip()))

    return cfg


def to_flat_dict(cfg: RerankerConfig, prefix: str = "") -> dict[str, Any]:
    """Flatten the config into dotted keys — handy for mlflow.log_params."""
    out: dict[str, Any] = {}
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        key = f"{prefix}{f.name}"
        if is_dataclass(value):
            out.update(to_flat_dict(value, prefix=f"{key}."))
        elif isinstance(value, list):
            out[key] = ",".join(str(x) for x in value)
        else:
            out[key] = value
    return out
