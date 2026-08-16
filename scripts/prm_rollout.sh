#!/bin/bash
#SBATCH --job-name=prm-rollout
#SBATCH --output=prm_rollout_%A_%a.out
#SBATCH --error=prm_rollout_%A_%a.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#
# Excluded (2026-08-13): node01 is the only node whose driver (590.48.01) satisfies torch
# 2.11+cu130, which is why lintloop.sh pins itself there -- but one of its four H100s answers
# cudaErrorDevicesUnavailable to every context created on it, and Slurm hands out the lowest
# free index first, so job 2450377 lost all three of its attempts to that GPU in 2:58.
#SBATCH --exclude=node01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#
# Generate the rollouts of a PRM campaign (PLAN_v2 §6, job B).
#
# Reads job A's prefixes.jsonl and writes one gzipped part per (run, shard, round) unit,
# each with the counters beside it. A unit whose part exists is skipped: `code_sha1` hashes
# text sampled at temperature 0.6 and never repeats, so resume is by part and by nothing
# else. Re-submitting after a wall-clock kill therefore continues the campaign.
#
# Usage, in order. Staging is a separate CPU pass and is NOT run here -- it is campaign-wide
# while this job is per unit and resumable -- and job C cannot be chained onto this one with
# --dependency, because `prm_eval.sh --shards` reads a manifest that staging has not written
# yet and the array size is evaluated at submit time.
#
#   CFG=reranker/configs/prm_rollout_l6_r0.yaml
#   python -m reranker.src.prm.rollout.prefixes --config $CFG                       # A
#   CONFIG=$CFG sbatch --array=0-$(($(CONFIG=$CFG scripts/prm_rollout.sh --units) - 1)) \
#       scripts/prm_rollout.sh                                                      # B, this script
#   # ... once it has finished:
#   python -m reranker.src.prm.rollout.stage --config $CFG           # rollouts -> kernels
#   OUT_DIR=$PWD/reranker/data/<out_dir> \
#       sbatch --array=0-$(($(scripts/prm_eval.sh --shards) - 1)) scripts/prm_eval.sh   # C

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Pinned, not inherited. ~/.bashrc exports this only in an interactive shell, so a job
# submitted non-interactively runs with HF_HOME unset and re-downloads the model into
# $HOME -- 61 GB of gpt-oss-120b against a 200 GB home quota, job 2450388.
export HF_HOME="${HF_HOME:-/path/to/.cache/huggingface}"
# And offline, so a cache that does not hold the model fails in seconds instead of pulling it
# down again: 61 GB for gpt-oss, 149 GB for DeepSeek-V4-Flash. lintloop.sh sets it for the
# same reason, and backend.py's DeepSeek path already assumes callers run offline.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Guard on a cached model, never on the directory: $HOME/.cache/huggingface exists and is
# empty, so `[ -d "$HF_HOME/hub" ]` passes on exactly the cache that would download again.
if ! ls -d "$HF_HOME"/hub/models--* >/dev/null 2>&1; then
    echo "ABORT: no models cached under HF_HOME=$HF_HOME" >&2
    exit 1
fi

CONFIG="${CONFIG:-reranker/configs/prm_rollout.yaml}"

if [ "${1:-}" = "--units" ]; then
    uv run --no-sync python -m reranker.src.prm.rollout.rollout --config "$CONFIG" --units
    exit 0
fi

echo "============================================================"
echo "  PRM rollouts (job B) | config $CONFIG"
echo "  node   : $(hostname)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

# vLLM's sampler JIT-compiles flashinfer's top-k/top-p kernel during memory profiling and
# needs nvcc, which these nodes do not have. See the script for the whole story.
source scripts/cuda_jit_env.sh

# A CUDA fault kills the context, so the process cannot recover in-band -- only restart.
# Each attempt resumes from the parts already on disk, exactly as a resubmitted job would.
ATTEMPTS="${ATTEMPTS:-3}"
for attempt in $(seq 1 "$ATTEMPTS"); do
    echo
    echo "--- rollout attempt $attempt/$ATTEMPTS"
    # "$@" so scalar overrides reach the driver the way §12 invokes them; list-valued knobs
    # still belong in a config file, since config._coerce has no list case.
    if uv run --no-sync python -m reranker.src.prm.rollout.rollout --config "$CONFIG" "$@"; then
        echo "--- rollouts finished on attempt $attempt"
        exit 0
    fi
    if [ "$attempt" -ge "$ATTEMPTS" ]; then
        echo "!! $ATTEMPTS attempts all died -- giving up. Finished units are intact;" >&2
        echo "!! resubmit to continue from the parts already written." >&2
        exit 1
    fi
    echo "--- attempt $attempt died; retrying, resuming from the finished units" >&2
done
