#!/bin/bash
#SBATCH --job-name=prm-train
#SBATCH --output=prm_train_%j.out
#SBATCH --error=prm_train_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
# 4 on one node. Data-parallel over lists: the corpus is ~121,800 train lists at ~4,000
# tokens each, which is ~14h of forward+backward on one H100 and ~4h on four.
#SBATCH --gres=gpu:h100:4
#
# Excluded (2026-08-13): node01 is the only node whose driver (590.48.01) satisfies torch
# 2.11+cu130, which is why lintloop.sh pins itself there -- but one of its four H100s answers
# cudaErrorDevicesUnavailable to every context created on it, and Slurm hands out the lowest
# free index first, so job 2450377 lost all three of its attempts to that GPU in 2:58.
#SBATCH --exclude=node01
# 1 task: torchrun forks the four ranks itself, so this is NOT --ntasks=4.
#SBATCH --ntasks=1
# 18 per rank, the ORM's ratio, against dataloader_num_workers=16.
#SBATCH --cpus-per-task=72
# 4 ranks x 16 workers, each forking the dataset's Source map -- a memory multiplier, not
# only a throughput knob. 4x the ORM's 64G.
#SBATCH --mem=256G
# ~2 days: the estimate is ~16h (13.7h train + 2.5h eval), so this is ~3x margin. The
# script passes no --resume, so a wall-clock kill costs the whole run, not the tail.
#SBATCH --time=2-00:00:00
#
# Train the PRM on a finished campaign's lists (prm_plan/PLAN_TRAINER.md).
#
# Reads {prm_rollout.out_dir}/lists_{train,val}.jsonl, the campaign's prefixes.jsonl and v1's
# parts; writes {train.output_dir}/final/ plus reranker_head.json, and logs to the same
# mlflow.db the ORM runs use.
#
# NOT resumable the way job B is: HF checkpoints under train.output_dir let `--resume` pick a
# run back up, but this script does not pass it -- a resubmit starts a fresh run rather than
# silently continuing one whose config may have moved.
#
# Usage, after a campaign has produced its lists. One backbone arm per submission:
#
#   CONFIG=reranker/configs/prm_train_qwen3base06b.yaml   sbatch scripts/prm_train.sh
#   CONFIG=reranker/configs/prm_train_qwen25coder05b.yaml sbatch scripts/prm_train.sh
#   # a smoke run first -- five steps, same code path:
#   CONFIG=reranker/configs/prm_train_qwen3base06b.yaml sbatch scripts/prm_train.sh \
#       train.max_steps=5
#
# Then score the checkpoint with job E and compare its numbers against the run's own final
# eval_prm_* metrics; they are computed by the same functions and must agree (§12 G1):
#
#   python -m reranker.src.prm.rollout.rank_eval \
#       --config reranker/configs/prm_rollout.yaml --checkpoint <output_dir>/final

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Pinned, not inherited. ~/.bashrc exports this only in an interactive shell, so a job
# submitted non-interactively runs with HF_HOME unset and re-downloads the backbone into
# $HOME against a 200 GB quota (job 2450388).
export HF_HOME="${HF_HOME:-/path/to/.cache/huggingface}"
# Offline, so a cache that does not hold the backbone fails in seconds instead of pulling it.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Guard on a cached model, never on the directory: $HOME/.cache/huggingface exists and is
# empty, so `[ -d "$HF_HOME/hub" ]` passes on exactly the cache that would download again.
if ! ls -d "$HF_HOME"/hub/models--* >/dev/null 2>&1; then
    echo "ABORT: no models cached under HF_HOME=$HF_HOME" >&2
    exit 1
fi

CONFIG="${CONFIG:-reranker/configs/prm_train_qwen3base06b.yaml}"

# The cu129 venv, NOT `uv run --no-sync` (which resolves .venv). .venv has neither mlflow nor
# accelerate; .venv-cu129 is the one the ORM trainers run from and has both, plus the
# cu129 torch build these H100s want.
PY="${PY:-$PWD/.venv-cu129/bin}"
[ -x "$PY/torchrun" ] || { echo "ABORT: no torchrun at $PY" >&2; exit 1; }

NPROC="${NPROC:-4}"

echo "============================================================"
echo "  PRM listwise training | config $CONFIG"
echo "  node   : $(hostname)"
echo "  gpus   : $NPROC"
echo "  start  : $(date)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

# torchrun, not python: DDP data-parallel over the four ranks. HF's Trainer picks the
# distributed env up on its own; only rank zero writes to MLflow and saves (train.py).
# "$@" so scalar overrides reach the trainer; list-valued knobs still belong in a config
# file, since config._coerce has no list case.
"$PY/torchrun" --nproc_per_node="$NPROC" --standalone \
    -m reranker.src.prm.rollout.train --config "$CONFIG" "$@"

echo "============================================================"
echo "  end    : $(date)"
echo "  mlflow : mlflow ui --backend-store-uri sqlite:///$PWD/reranker/mlflow.db"
echo "============================================================"
