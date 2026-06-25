#!/bin/bash
# Train the kernel reranker with the LISTWISE (LambdaRank) loss locally (single GPU, via uv).
# Usage:
#   bash reranker/scripts/train_listwise_local.sh                                            # default listwise config
#   bash reranker/scripts/train_listwise_local.sh reranker/configs/listwise_config.yaml train.max_steps=20   # smoke test
#   bash reranker/scripts/train_listwise_local.sh reranker/configs/listwise_config.yaml listwise.sigma=2.0
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # reranker/
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"                      # repo root

CONFIG="${1:-$PROJECT_ROOT/configs/listwise_config.yaml}"
shift || true   # remaining args are dotted key=value config overrides

# Memory-friendly defaults for a single consumer GPU. A list = up to list_size
# forward passes, so the per-device batch is kept at 1 list. Applied before "$@"
# so CLI wins.
LOCAL_DEFAULTS=(
  model.max_length=4096
  train.per_device_train_batch_size=1
  train.per_device_eval_batch_size=8
  train.gradient_accumulation_steps=16
  listwise.list_size=16
)

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

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

# build_dataset / lists run automatically inside listwise/train.py if artifacts are missing.
uv run python -m reranker.src.listwise.train --config "$CONFIG" "${LOCAL_DEFAULTS[@]}" "$@"

echo "============================================"
echo "  End time   : $(date)"
echo "  MLflow     : mlflow ui --backend-store-uri sqlite:///$PROJECT_ROOT/mlflow.db --host 0.0.0.0 --allowed-hosts '*' --cors-allowed-origins '*'"
echo "  Checkpoint : $PROJECT_ROOT/data/checkpoints_listwise/final/"
echo "============================================"
