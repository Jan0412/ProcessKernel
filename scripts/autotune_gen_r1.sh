#!/bin/bash
#SBATCH --job-name=autotune_gen_r1
#SBATCH --output=autotune_gen_r1_%j.out
#SBATCH --error=autotune_gen_r1_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#
# Round-1 generation. Feeds BOTH A1 (these kernels as written) and A2 (the same kernels,
# swept). Nothing here is arm-specific.
#
# Qwen3-Coder-30B-A3B is a MoE with 3B active params; bf16 weights are ~60 GB, so it fits on
# one 80 GB H100 with room for the KV cache. If it OOMs, switch to the FP8 variant before
# reaching for a second GPU -- generation is not the bottleneck in this experiment.
#
# Usage:  sbatch scripts/autotune_gen_r1.sh <level>       # level = 1 or 2
#         SMOKE=1 sbatch scripts/autotune_gen_r1.sh 1     # 5 problems x 2 samples

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

LEVEL="${1:?usage: sbatch scripts/autotune_gen_r1.sh <level>}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
SMOKE="${SMOKE:-0}"

MODEL_SLUG=$(basename "$MODEL")
OUTPUT_DIR="$SLURM_SUBMIT_DIR/runs/${MODEL_SLUG}_level${LEVEL}_r1_triton"

if [ "$SMOKE" = "1" ]; then
    SCOPE=(--problems 0-4 --num-samples 2)
    OUTPUT_DIR="${OUTPUT_DIR}_smoke"
else
    SCOPE=(--all --num-samples 10)
fi

echo "============================================================"
echo "  round-1 generation | level $LEVEL | smoke=$SMOKE"
echo "  model  : $MODEL"
echo "  out    : $OUTPUT_DIR"
echo "  node   : $(hostname)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

uv run --no-sync python kernel_gen/generate_kernels_samples.py \
    --model "$MODEL" \
    --level "$LEVEL" \
    "${SCOPE[@]}" \
    --backend triton \
    --option one_shot \
    --temperature 0.3 \
    --think-temperature 1.0 \
    --max-new-tokens 16384 \
    --output-dir "$OUTPUT_DIR" \
    --skip-existing

echo
echo "Generated $(find "$OUTPUT_DIR" -name '*_kernel.py' | wc -l) kernels."
echo "Next: sbatch scripts/autotune_eval.sh $OUTPUT_DIR $LEVEL"
