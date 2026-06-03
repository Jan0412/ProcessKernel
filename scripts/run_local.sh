#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../kernel_gen"

MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
OUTPUT_DIR="Qwen3-Coder-30B-A3B-Instruct_generated_kernels_samples"

echo "============================================"
echo "  Start time  : $(date)"
echo "============================================"
nvidia-smi
echo "============================================"

python generate_kernels_samples.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --level 1 \
    --all \
    --num-samples 10 \
    --backend triton \
    --option one_shot \
    --gpu-name A100 \
    --max-new-tokens 16384 \
    --skip-existing

echo "============================================"
echo "  End time    : $(date)"
echo "  Exit code   : $?"
echo "============================================"
