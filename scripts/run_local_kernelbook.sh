#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../kernel_gen"

MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
# Pseudo-level used in output filenames; must match what you pass to
# convert_kernelbook.py and to eval_from_generations.py (level=$PSEUDO_LEVEL).
PSEUDO_LEVEL=5
# First 1000 KernelBook rows (the dataset is a flat 18k-row train split, no levels).
ROWS="0-999"
NUM_SAMPLES=10
OUTPUT_DIR="/path/to/ProcessKernel/runs/Qwen3-Coder-30B-A3B-Instruct-FP8_kernelbook_level${PSEUDO_LEVEL}_triton"

echo "============================================"
echo "  Start time  : $(date)"
echo "  Rows        : $ROWS"
echo "  Num samples : $NUM_SAMPLES"
echo "  Output dir  : $OUTPUT_DIR"
echo "============================================"
nvidia-smi
echo "============================================"

python generate_kernelbook_samples.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --rows "$ROWS" \
    --pseudo-level "$PSEUDO_LEVEL" \
    --num-samples "$NUM_SAMPLES" \
    --backend triton \
    --option one_shot \
    --gpu-name A100 \
    --max-new-tokens 16384 \
    --skip-existing

echo "============================================"
echo "  End time    : $(date)"
echo "  Exit code   : $?"
echo "============================================"
