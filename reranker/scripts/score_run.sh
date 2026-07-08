#!/bin/bash
# Stage 1b — score an eval table with one reranker checkpoint (single GPU, via uv).
# Writes scores/<name>.jsonl keyed by (run_name, kernel_file). Run once per
# reranker you want to compare; the eval table (Stage 1a) is built only once.
#
# Usage:
#   bash reranker/scripts/score_run.sh \
#       --eval-table reranker/data/eval/eval_table.jsonl \
#       --runs /path/runs/RUN1 /path/runs/RUN2 \
#       --kernelbench-dir /path/KernelBench \
#       --checkpoint /path/listwise_model_06B \
#       --name listwise_l56 \
#       --out reranker/data/eval/scores/listwise_l56.jsonl \
#       --max-length 6144 --batch-size 8
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # reranker/
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"                      # repo root

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd "$REPO_ROOT"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

uv run python -m reranker.src.eval_pipeline.score_run "$@"
