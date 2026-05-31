#!/bin/bash
#SBATCH --job-name=kernel_reranker
#SBATCH --output=kernel_reranker_%j.out
#SBATCH --error=kernel_reranker_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
# reranker project root = directory containing this script's parent.
# repo root = parent of reranker/ (contains pyproject.toml, src/, KernelBench/).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-$PROJECT_ROOT/configs/default.yaml}"
shift || true                          # remaining args are dotted config overrides
# PYTHONPATH: repo root for `reranker.src.*`; repo root/src for `kernelbench`.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python3.10 -m ensurepip --user 2>/dev/null || curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.10 - --user
python3.10 -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
python3.10 -m pip install -q transformers datasets accelerate huggingface_hub
python3.10 -m pip install -q -r "${PROJECT_ROOT}/requirements.txt"
# KernelBench is imported (dataset utils) by the dataset builder.
if [ -d "${REPO_ROOT}" ]; then
    python3.10 -m pip install -q --no-deps "${REPO_ROOT}" --ignore-requires-python || true
fi

echo "============================================"
echo "  Job ID      : ${SLURM_JOB_ID:-local}"
echo "  Node        : ${SLURM_NODELIST:-local}"
echo "  GPUs        : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Project root: $PROJECT_ROOT"
echo "  Config      : $CONFIG"
echo "  Overrides   : $*"
echo "  Start time  : $(date)"
echo "============================================"
nvidia-smi || true
echo "============================================"

# ── Train ─────────────────────────────────────────────────────────────────────
# build_dataset / splits run automatically inside train.py if artifacts are missing.
python3.10 -m reranker.src.train --config "$CONFIG" "$@"

echo "============================================"
echo "  End time    : $(date)"
echo "  MLflow store: ${PROJECT_ROOT}/mlruns"
echo "  Inspect     : mlflow ui --backend-store-uri ${PROJECT_ROOT}/mlruns"
echo "============================================"
