#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../kernel_gen"

MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
OUTPUT_DIR="$SCRIPT_DIR/../runs/Qwen3-Coder-30B-A3B-Instruct-FP8_kernelbook_reranked_pair_fastp_think_level2_triton"
RERANKER_CHECKPOINT="/path/to/ProcessKernel/mlruns/3/3160efcda6304b909af4ab29b87f7384/artifacts/model"

echo "============================================"
echo "  Start time  : $(date)"
echo "============================================"
nvidia-smi
echo "============================================"

python generate_kernels_reranked.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --level 2 \
    --all \
    --num-samples 10 \
    --backend triton \
    --option one_shot \
    --gpu-name L40S \
    --max-new-tokens 16384 \
    --temperature 0.3 \
    --think-temperature 1.0 \
    --reranker-checkpoint "$RERANKER_CHECKPOINT" \
    --reranker-device cuda:0 \
    --skip-existing

echo "============================================"
echo "  End time    : $(date)"
echo "  Exit code   : $?"
echo "============================================"
