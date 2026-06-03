#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../kernel_gen"

MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
OUTPUT_DIR="$SCRIPT_DIR/../runs/Qwen3-Coder-30B-A3B-Instruct_reranked_level1_triton"
RERANKER_CHECKPOINT="$SCRIPT_DIR/../reranker/data/checkpoints/final"

echo "============================================"
echo "  Start time  : $(date)"
echo "============================================"
nvidia-smi
echo "============================================"

python generate_kernels_reranked.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --level 1 \
    --all \
    --num-samples 10 \
    --backend triton \
    --option one_shot \
    --gpu-name A100 \
    --max-new-tokens 16384 \
    --reranker-checkpoint "$RERANKER_CHECKPOINT" \
    --reranker-device cuda:0 \
    --skip-existing

echo "============================================"
echo "  End time    : $(date)"
echo "  Exit code   : $?"
echo "============================================"
