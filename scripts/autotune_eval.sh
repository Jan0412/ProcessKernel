#!/bin/bash
#SBATCH --job-name=autotune_eval
#SBATCH --output=autotune_eval_%j.out
#SBATCH --error=autotune_eval_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#
# Evaluate every generated sample in a run -> eval_results.json.
#
# On round 1 this produces arm A1. On the round-2 runs it is what the sweep needs before it
# can pick which kernels to tune, so all four arms pass through here.
#
# Deliberately NOT KernelBench's scripts/eval_from_generations.py: that script resumes per
# *problem*, so once any sample of a problem is written it skips the rest. These jobs get
# requeued, and that granularity would silently drop samples. autotune.eval_run resumes per
# sample, and calls the identical eval_kernel_against_ref with the same 5/100 trials.
#
# Usage:  sbatch scripts/autotune_eval.sh <run_dir> <level>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

RUN_DIR="${1:?usage: sbatch scripts/autotune_eval.sh <run_dir> <level>}"
LEVEL="${2:?usage: sbatch scripts/autotune_eval.sh <run_dir> <level>}"

echo "============================================================"
echo "  eval | $RUN_DIR | level $LEVEL"
echo "  kernels: $(find "$RUN_DIR" -name '*_kernel.py' | wc -l)"
echo "  node   : $(hostname)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

# Resumable: re-running after a requeue skips the samples already recorded.
uv run --no-sync python -m autotune.eval_run \
    --run-dir "$RUN_DIR" \
    --level "$LEVEL" \
    --dataset-dir "KernelBench/level${LEVEL}" \
    --gpus 0,1 \
    --timeout 300 \
    --num-correct-trials 5 \
    --num-perf-trials 100

echo
echo "Next: sbatch scripts/autotune_sweep.sh $RUN_DIR $LEVEL"
