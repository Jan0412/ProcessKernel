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


def _resolve(path: str) -> str:
    """Resolve a possibly-relative path against the project root."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def _from_dict(cls, data: dict) -> Any:
    """Recursively build a (nested) dataclass from a plain dict."""
    # `from __future__ import annotations` makes field types strings — resolve them.
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(ftype, value)
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
