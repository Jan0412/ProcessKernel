#!/bin/bash
#SBATCH --job-name=autotune_gen_r2
#SBATCH --output=autotune_gen_r2_%j.out
#SBATCH --error=autotune_gen_r2_%j.err
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
# Round-2 generation: arm A3 (timing feedback) or arm A4 (tuning feedback).
#
# Both arms are seeded with the SAME kernel -- the A2 champion for each problem -- and given
# the SAME sampling budget. The only thing that differs is the feedback text. That is the
# whole design: it is what makes any A4-vs-A3 difference attributable to the feedback and
# nothing else.
#
# Run the tuning arm FIRST, then the timing arm with --assert-seeds-match pointed at it. The
# assert compares seed file hashes across the arms and refuses to proceed if they differ,
# because a seed mismatch would make the headline comparison uninterpretable.
#
# Usage:  sbatch scripts/autotune_gen_r2.sh <arm> <round1_dir> <level>
#           arm = tuning (A4)  |  timing (A3)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

ARM="${1:?usage: sbatch scripts/autotune_gen_r2.sh <tuning|timing> <round1_dir> <level>}"
ROUND1_DIR="${2:?usage: sbatch scripts/autotune_gen_r2.sh <tuning|timing> <round1_dir> <level>}"
LEVEL="${3:?usage: sbatch scripts/autotune_gen_r2.sh <tuning|timing> <round1_dir> <level>}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"

MODEL_SLUG=$(basename "$MODEL")
OUTPUT_DIR="$SLURM_SUBMIT_DIR/runs/${MODEL_SLUG}_level${LEVEL}_r2${ARM}_triton"

# The control arm must be seeded identically to the proposal arm.
FAIRNESS=()
OTHER_ARM=$([ "$ARM" = "timing" ] && echo tuning || echo timing)
OTHER_SEEDS="$SLURM_SUBMIT_DIR/runs/${MODEL_SLUG}_level${LEVEL}_r2${OTHER_ARM}_triton/seeds.json"
if [ -f "$OTHER_SEEDS" ]; then
    FAIRNESS=(--assert-seeds-match "$OTHER_SEEDS")
    echo "will assert seeds match the $OTHER_ARM arm"
fi

echo "============================================================"
echo "  round-2 generation | arm $ARM | level $LEVEL"
echo "  seeds from : $ROUND1_DIR/sweep/sweep_summary.json"
echo "  out        : $OUTPUT_DIR"
echo "  node       : $(hostname)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

uv run --no-sync python kernel_gen/generate_kernels_feedback.py \
    --model "$MODEL" \
    --level "$LEVEL" \
    --arm "$ARM" \
    --round1-dir "$ROUND1_DIR" \
    --baseline-file "timing/H100/baseline_time_torch.json" \
    --output-dir "$OUTPUT_DIR" \
    --all \
    --num-samples "$NUM_SAMPLES" \
    --temperature 0.3 \
    --think-temperature 1.0 \
    --max-new-tokens 16384 \
    --max-model-len 32768 \
    --skip-existing \
    "${FAIRNESS[@]}"

echo
echo "Next: sbatch scripts/autotune_eval.sh $OUTPUT_DIR $LEVEL"
echo "then: sbatch scripts/autotune_sweep.sh $OUTPUT_DIR $LEVEL"
