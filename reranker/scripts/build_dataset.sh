#!/bin/bash
# Build the labeled reranker dataset + problem-level splits from KernelBench eval runs.
# Local, CPU-only — no GPU needed. Run from anywhere.
set -euo pipefail

# Resolve paths: reranker project root (this script lives in reranker/scripts/)
# and repo root (parent of reranker/, contains pyproject.toml and src/).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

CONFIG="${1:-$PROJECT_ROOT/configs/default.yaml}"

# Put the repo root on PYTHONPATH so `reranker.src.*` (namespace package) resolves.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Run from the repo root so `uv` finds pyproject.toml.
cd "$REPO_ROOT"

echo "[build_dataset] project root : $PROJECT_ROOT"
echo "[build_dataset] repo root    : $REPO_ROOT"
echo "[build_dataset] config       : $CONFIG"

uv run python -m reranker.src.data.build_dataset --config "$CONFIG"
uv run python -m reranker.src.data.splits        --config "$CONFIG"

echo "[build_dataset] done."
