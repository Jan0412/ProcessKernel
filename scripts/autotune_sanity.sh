#!/bin/bash
#SBATCH --job-name=autotune_sanity
#SBATCH --output=autotune_sanity_%j.out
#SBATCH --error=autotune_sanity_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# THE check that has to pass before any number this project produces means anything.
#
# The sweep reports a "tuning gain" = (runtime at the kernel's own constants) / (runtime at
# the best config found). If we took the numerator from eval_results.json and the denominator
# from the sweep, then ANY systematic difference between the two harnesses -- a different
# process, a different warmup, a different cache state -- would show up as tuning gain. We
# would get a beautiful result that was pure measurement artifact.
#
# So the sweep re-measures the kernel's own constants itself, as config 0. This job verifies
# that config 0 reproduces what the evaluator recorded for the same kernel. If it does not,
# the two are not measuring the same thing and the whole experiment is invalid.
#
# Usage:  sbatch scripts/autotune_sanity.sh <run_dir> <level>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

RUN_DIR="${1:?usage: sbatch scripts/autotune_sanity.sh <run_dir> <level>}"
LEVEL="${2:?usage: sbatch scripts/autotune_sanity.sh <run_dir> <level>}"
N_PROBLEMS="${3:-20}"

echo "=== identity-config sanity check: $RUN_DIR (level $LEVEL, $N_PROBLEMS problems) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

# Sweep a handful of kernels. We only need config 0 re-measured at full fidelity, so run the
# search phase (which includes config 0) and then finalize.
uv run --no-sync python -m autotune.sweep \
    --run-dir "$RUN_DIR" --level "$LEVEL" \
    --top-k 1 --gpus 0,1 --timeout 300 \
    --limit-problems "$N_PROBLEMS"

uv run --no-sync python -m autotune.sweep \
    --run-dir "$RUN_DIR" --level "$LEVEL" \
    --top-k 1 --gpus 0,1 --timeout 300 \
    --limit-problems "$N_PROBLEMS" --finalize

echo
uv run --no-sync python -m autotune.sanity --run-dir "$RUN_DIR"
