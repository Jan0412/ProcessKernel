#!/bin/bash
#SBATCH --job-name=prm-orm-score
#SBATCH --output=prm_orm_score_%A_%a.out
#SBATCH --error=prm_orm_score_%A_%a.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=node01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#
# Job B2: score a campaign's rollouts with the ORM (PLAN_v3, one unit per array task).
#
# Usage:
#   CONFIG=reranker/configs/prm_rollout_l1.yaml sbatch --array=0 scripts/prm_orm_score.sh
#   CONFIG=... ANCHORS=1 sbatch --array=0 scripts/prm_orm_score.sh     # anchors part only
#
# The array index space comes from prefixes.jsonl, not a disk glob, so a partial resubmit
# re-runs the units it names. A task whose unit has no rollout part exits before the model
# loads -- the QoS caps this account at 16 GPUs and an idle task must not hold one.
set -euo pipefail

CONFIG="${CONFIG:?set CONFIG to a prm_rollout campaign yaml}"
# $0 is Slurm's spooled copy, not the repo -- same idiom as prm_rollout.sh.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

if [ -n "${ANCHORS:-}" ]; then
    exec uv run --no-sync python -m reranker.src.prm.rollout.orm_score --anchors --config "$CONFIG"
fi
exec uv run --no-sync python -m reranker.src.prm.rollout.orm_score --config "$CONFIG"
