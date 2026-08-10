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


def load_config(argv: Optional[list[str]] = None) -> RerankerConfig:
    """Parse `--config path` plus dotted `key=value` overrides into a RerankerConfig."""
    parser = argparse.ArgumentParser(description="Kernel reranker pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, overrides = parser.parse_known_args(argv)

    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _from_dict(RerankerConfig, raw)

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
