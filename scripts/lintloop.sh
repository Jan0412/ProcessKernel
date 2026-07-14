#!/bin/bash
#SBATCH --job-name=lintloop
#SBATCH --output=lintloop_%j.out
#SBATCH --error=lintloop_%j.err
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
# Arm A5: the lint-feedback loop. Generate, lint, repair, up to --rounds times, with each
# sample stopping as soon as it is clean.
#
# This ONE job produces both sides of the comparison. rounds/round_0/ is an unrefined
# generation -- same prompt, same sampler, no feedback existed yet -- so it is the
# baseline, and it is paired with the refined run slot for slot. Do not generate a
# separate baseline; an independently-drawn one would confound "the feedback helped"
# with "the sampler changed".
#
# Usage:  sbatch scripts/lintloop.sh <level>                    # KernelBench, 3 rounds x 10 samples
#         SMOKE=1 sbatch scripts/lintloop.sh 1                  # 5 problems x 2 samples x 2 rounds
#         DATASET=kernelbook sbatch scripts/lintloop.sh 5       # KernelBook at pseudo-level 5
#
# Then, and this is where the result actually lives:
#         sbatch scripts/autotune_eval.sh <OUTPUT_DIR> <level>
#         sbatch scripts/autotune_eval.sh <OUTPUT_DIR>/rounds/round_0 <level>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

LEVEL="${1:?usage: sbatch scripts/lintloop.sh <level>}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
DATASET="${DATASET:-kernelbench}"
ROUNDS="${ROUNDS:-3}"
POLICY="${POLICY:-severity}"
SMOKE="${SMOKE:-0}"

MODEL_SLUG=$(basename "$MODEL")
TAG=$([ "$DATASET" = "kernelbook" ] && echo "kb" || echo "level")
OUTPUT_DIR="$SLURM_SUBMIT_DIR/runs/${MODEL_SLUG}_${TAG}${LEVEL}_lintloop_triton"

if [ "$SMOKE" = "1" ]; then
    SCOPE=(--problems 0-4 --num-samples 2)
    ROUNDS=2
    OUTPUT_DIR="${OUTPUT_DIR}_smoke"
else
    SCOPE=(--all --num-samples 10)
fi

echo "============================================================"
echo "  lint-feedback loop (A5) | $DATASET level $LEVEL | smoke=$SMOKE"
echo "  model  : $MODEL"
echo "  rounds : $ROUNDS (policy=$POLICY)"
echo "  out    : $OUTPUT_DIR"
echo "  node   : $(hostname)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

uv run --no-sync python -m kernel_gen.arms.lintloop \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --level "$LEVEL" \
    "${SCOPE[@]}" \
    --rounds "$ROUNDS" \
    --feedback-policy "$POLICY" \
    --backend triton \
    --option one_shot \
    --temperature 0.3 \
    --think-temperature 1.0 \
    --max-new-tokens 16384 \
    --max-model-len 32768 \
    --output-dir "$OUTPUT_DIR" \
    --skip-existing

echo
echo "Refined kernels : $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_kernel.py' | wc -l)"
echo "Baseline (r0)   : $(find "$OUTPUT_DIR/rounds/round_0" -maxdepth 1 -name '*_kernel.py' | wc -l)"
echo
echo "Next -- eval BOTH, and compare those, not the linter's own numbers:"
echo "  sbatch scripts/autotune_eval.sh $OUTPUT_DIR $LEVEL"
echo "  sbatch scripts/autotune_eval.sh $OUTPUT_DIR/rounds/round_0 $LEVEL"
