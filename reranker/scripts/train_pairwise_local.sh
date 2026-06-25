#!/bin/bash
# Train the kernel reranker with the PAIRWISE loss locally (single GPU, via uv).
# Usage:
#   bash reranker/scripts/train_pairwise_local.sh                                        # default pairwise config
#   bash reranker/scripts/train_pairwise_local.sh reranker/configs/pairwise_config.yaml train.max_steps=20   # smoke test
#   bash reranker/scripts/train_pairwise_local.sh reranker/configs/pairwise_config.yaml pairwise.loss_type=margin
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # reranker/
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"                      # repo root

CONFIG="${1:-$PROJECT_ROOT/configs/pairwise_config.yaml}"
shift || true   # remaining args are dotted key=value config overrides

# Memory-friendly defaults for a single consumer GPU. A pair = 2 forward passes,
# so the per-device batch is kept modest. Applied before "$@" so CLI wins.

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

# build_dataset / pairs run automatically inside pairwise/train.py if artifacts are missing.
uv run python -m reranker.src.pairwise.train --config "$CONFIG" "$@"

echo "============================================"
echo "  End time   : $(date)"
echo "  MLflow     : mlflow ui --backend-store-uri sqlite:///$PROJECT_ROOT/mlflow.db --host 0.0.0.0 --allowed-hosts '*' --cors-allowed-origins '*'"
echo "  Checkpoint : $PROJECT_ROOT/data/checkpoints_pairwise/final/"
echo "============================================"
