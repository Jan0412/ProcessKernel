#!/bin/bash
#SBATCH --job-name=autotune_sweep
#SBATCH --output=autotune_sweep_%j.out
#SBATCH --error=autotune_sweep_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#
# Sweep launch configs over a run's correct kernels.
#
# On round 1 this IS arm A2 -- and the go/no-go gate. On the round-2 runs it is how A3 and A4
# get scored, so that every arm is compared with its constants held at their best and what we
# are actually comparing is the quality of the algorithm the model wrote.
#
# Two phases. Search ranks the grid cheaply (3 correctness trials, 20 timed runs). Finalize
# re-measures only the winner AND the kernel's own constants at the full 5/100, so the
# reported tuning gain is a ratio of two full-fidelity numbers taken by the same harness.
#
# Safe to requeue: results.jsonl is append-only and resume skips what is already recorded.
#
# Usage:  sbatch scripts/autotune_sweep.sh <run_dir> <level> [top_k]

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

RUN_DIR="${1:?usage: sbatch scripts/autotune_sweep.sh <run_dir> <level> [top_k]}"
LEVEL="${2:?usage: sbatch scripts/autotune_sweep.sh <run_dir> <level> [top_k]}"
TOP_K="${3:-2}"

echo "============================================================"
echo "  sweep | $RUN_DIR | level $LEVEL | top-k $TOP_K"
echo "  node  : $(hostname)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

COMMON=(--run-dir "$RUN_DIR" --level "$LEVEL"
        --dataset-dir "KernelBench/level${LEVEL}"
        --top-k "$TOP_K" --gpus 0,1 --timeout 240)

echo; echo "--- phase 1: search the grid ---"
uv run --no-sync python -m autotune.sweep "${COMMON[@]}"

echo; echo "--- phase 2: re-time the winner and the identity config at full fidelity ---"
uv run --no-sync python -m autotune.sweep "${COMMON[@]}" --finalize

echo; echo "--- is the sweep measuring the same thing the evaluator measured? ---"
uv run --no-sync python -m autotune.sanity --run-dir "$RUN_DIR" || {
    echo
    echo "SANITY FAILED — stop here. Every tuning gain below would be a measurement"
    echo "artifact rather than a real speedup. Do not run the gate on this."
    exit 1
}

echo; echo "--- A2 / go-no-go gate ---"
uv run --no-sync python -m autotune.gate --run-dirs "$RUN_DIR"
