#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../kernel_gen"

MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
OUTPUT_DIR="/path/to/ProcessKernel/runs/Qwen3-Coder-30B-A3B-Instruct-FP8_level2_triton_one_shot_plan_mode"

echo "============================================"
echo "  Start time  : $(date)"
echo "============================================"
nvidia-smi
echo "============================================"

python generate_kernels_samples.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --level 2 \
    --all \
    --num-samples 1 \
    --backend triton \
    --option one_shot \
    --gpu-name L40S \
    --max-new-tokens 16384 \
    --skip-existing

echo "============================================"
echo "  End time    : $(date)"
echo "  Exit code   : $?"
echo "============================================"
