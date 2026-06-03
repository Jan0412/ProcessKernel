#!/bin/bash
# Evaluate a reranker checkpoint on all splits (train / val / test).
# Usage:
#   bash reranker/scripts/eval_local.sh
#   bash reranker/scripts/eval_local.sh reranker/configs/default.yaml
#   bash reranker/scripts/eval_local.sh reranker/configs/default.yaml --checkpoint data/checkpoints/final
#   bash reranker/scripts/eval_local.sh reranker/configs/default.yaml --checkpoint data/checkpoints/checkpoint-200
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # reranker/
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"                      # repo root

CONFIG="${1:-$PROJECT_ROOT/configs/default.yaml}"
shift || true   # remaining args forwarded to eval.py (e.g. --checkpoint ...)

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

cd "$REPO_ROOT"

echo "============================================"
echo "  Project root : $PROJECT_ROOT"
echo "  Repo root    : $REPO_ROOT"
echo "  Config       : $CONFIG"
echo "  Args         : $*"
echo "  Start time   : $(date)"
echo "============================================"

uv run python -m reranker.src.eval --config "$CONFIG" "$@"

echo "============================================"
echo "  End time : $(date)"
echo "============================================"
