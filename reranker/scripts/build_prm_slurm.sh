#!/bin/bash
#SBATCH --job-name=prm_build
#SBATCH --output=prm_build_%j.out
#SBATCH --error=prm_build_%j.err
# YOUR_PARTITION, though the build is pure CPU and cpu-batch is its natural home: cpu-batch
# allows only QOS normal/high, and the only association carrying `normal` is the `default`
# account, which is capped at MaxSubmit=0. YOUR_PARTITION accepts the QOS. No --gres, so
# this schedules against free CPUs and never competes with the GPU jobs queued here.
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
# Each worker streams one attempts.jsonl (<=117 MB) and holds a tokenizer; 64G is slack.
#SBATCH --mem=64G
# Measured 3m03s for 4 units / 13,072 attempts at 4 workers, so 144 units at 32 is ~12 min.
# 4h absorbs a slow node or a cold GPFS read rather than losing the job at the wall.
#SBATCH --time=04:00:00
#
#   sbatch reranker/scripts/build_prm_slurm.sh                          # prm_config.yaml
#   sbatch reranker/scripts/build_prm_slurm.sh reranker/configs/prm_smoke.yaml
# Trailing dotted key=value args override the config; list values cannot (config._coerce
# has no list case), so rounds/run_dirs stay in the file. stats.py is NOT run here -- it is
# a read-only report you will re-run by hand, and it does not need 32 cores.
set -euo pipefail

# NOT dirname "${BASH_SOURCE[0]}": sbatch runs a COPY out of /var/spool/slurmd/job<N>, so
# under SLURM that resolves the repo to /var/spool. Submit from the repo root.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [ ! -d "$REPO_ROOT/reranker/configs" ]; then
    echo "[FATAL] $REPO_ROOT is not the repo root (no reranker/configs). Submit from the repo root." >&2
    exit 1
fi

CONFIG="${1:-reranker/configs/prm_config.yaml}"
[ $# -gt 0 ] && shift
# Absolute before the cd, or a relative config resolves against the wrong directory.
[ -f "$CONFIG" ] && CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
[ -f "$CONFIG" ] || CONFIG="$REPO_ROOT/$CONFIG"
[ -f "$CONFIG" ] || { echo "[FATAL] config not found: $CONFIG" >&2; exit 1; }

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# Off, not merely unset: each of the 32 forked workers builds a tokenizer, and HF's Rust
# parallelism deadlocks across fork unless it is disabled before the first one is made.
export TOKENIZERS_PARALLELISM=false
# The home cache, like train_listwise_slurm.sh -- the Qwen3-Reranker checkpoints are only
# there, not in the scratch cache holding the generation models.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# .venv, not .venv-cu129: the tests and the smoke build ran here. A different transformers
# pins a different tokenizer, which would move n_raw_tokens and so which rows hit max_length.
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || { echo "[FATAL] no python at $PY" >&2; exit 1; }

# From SLURM rather than the config, so the pool and the allocation cannot drift apart.
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

echo "============================================"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Node       : ${SLURM_NODELIST:-local}"
echo "  Python     : $PY"
echo "  Config     : $CONFIG"
echo "  Workers    : $WORKERS"
echo "  Overrides  : $*"
echo "  Start time : $(date)"
echo "============================================"

"$PY" -m reranker.src.prm.build --config "$CONFIG" "prm.num_workers=$WORKERS" "$@"
"$PY" -m reranker.src.prm.splits --config "$CONFIG" "$@"

echo "============================================"
echo "  End time   : $(date)"
echo "  Report     : $PY -m reranker.src.prm.stats --config $CONFIG"
echo "============================================"
