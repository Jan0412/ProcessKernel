#!/bin/bash
# Stage 1a — build the reranker-independent eval table (CPU only).
# Walks the KernelBench run dirs, joins eval outcomes + baseline timings, and
# writes eval_table.jsonl. Built once, reused by every reranker (Stage 1b).
#
# Usage:
#   bash reranker/scripts/build_eval_table.sh \
#       --runs /path/runs/RUN1 /path/runs/RUN2 \
#       --timing /path/results/timing/A100/baseline_time_torch.json \
#       --kernelbench-dir /path/KernelBench \
#       --out reranker/data/eval/eval_table.jsonl
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # reranker/
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"                      # repo root

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO_ROOT"

uv run python -m reranker.src.eval_pipeline.build_eval_table "$@"
