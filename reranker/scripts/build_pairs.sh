#!/bin/bash
# Build the pairwise reranker dataset (pairs + fresh train/val split) from the
# labeled source dataset. Local, CPU-only — no GPU needed. Run from anywhere.
set -euo pipefail

# reranker project root (this script lives in reranker/scripts/) and repo root.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

CONFIG="${1:-$PROJECT_ROOT/configs/pairwise_config.yaml}"
shift || true   # remaining args are dotted key=value config overrides

# Put the repo root on PYTHONPATH so `reranker.src.*` resolves; src/ for `kernelbench`.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"

echo "[build_pairs] project root : $PROJECT_ROOT"
echo "[build_pairs] repo root    : $REPO_ROOT"
echo "[build_pairs] config       : $CONFIG"
echo "[build_pairs] overrides    : $*"

# Ensure the labeled source dataset exists, then build the pairs + pairwise split.
uv run python -m reranker.src.data.build_dataset --config "$CONFIG" "$@"
uv run python -m reranker.src.pairwise.pairs     --config "$CONFIG" "$@"

echo "[build_pairs] done."
