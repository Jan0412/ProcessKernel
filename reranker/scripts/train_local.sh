#!/bin/bash
# Train the kernel reranker locally (single GPU, via uv).
# Usage:
#   bash reranker/scripts/train_local.sh                              # default config
#   bash reranker/scripts/train_local.sh reranker/configs/default.yaml train.max_steps=20   # smoke test
#   bash reranker/scripts/train_local.sh reranker/configs/default.yaml model.base_model=Qwen/Qwen3-Reranker-0.6B
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # reranker/
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"                      # KernelBench/

CONFIG="${1:-$PROJECT_ROOT/configs/default.yaml}"
shift || true   # remaining args are dotted key=value config overrides

# Memory-friendly defaults for a single consumer GPU (~16 GB). These are applied
# before "$@", so any override you pass on the CLI takes precedence.
LOCAL_DEFAULTS=(
  model.max_length=4096
  train.per_device_train_batch_size=8
  train.per_device_eval_batch_size=8
  train.gradient_accumulation_steps=8
)

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

unset HF_HUB_ENABLE_HF_TRANSFER 2>/dev/null || true

cd "$REPO_ROOT"

echo "============================================"
echo "  Project root : $PROJECT_ROOT"
echo "  Repo root    : $REPO_ROOT"
echo "  Config       : $CONFIG"
echo "  Overrides    : $*"
echo "  Start time   : $(date)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
echo "============================================"

# build_dataset / splits run automatically inside train.py if artifacts are missing.
uv run python -m reranker.src.train --config "$CONFIG" "$@"

echo "============================================"
echo "  End time   : $(date)"
echo "  MLflow     : mlflow ui --backend-store-uri sqlite:///$PROJECT_ROOT/mlflow.db --host 0.0.0.0 --allowed-hosts '*' --cors-allowed-origins '*'"
echo "  Checkpoint : $PROJECT_ROOT/data/checkpoints/final/"
echo "============================================"
