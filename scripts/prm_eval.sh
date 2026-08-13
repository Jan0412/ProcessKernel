#!/bin/bash
#SBATCH --job-name=prm-eval
#SBATCH --output=prm_eval_%A_%a.out
#SBATCH --error=prm_eval_%A_%a.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#
# Excluded, not pinned -- the opposite of lintloop.sh, and for its reason. node01 is the only
# node with driver 590.48.01, which is the only one torch 2.11+cu130 runs on, so generation can
# go nowhere else. KernelBench pins torch cu128 and runs on every H100 node, so eval keeps off
# node01 instead of queueing on top of the job that has no alternative.
#SBATCH --exclude=node01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#
# Evaluate one shard of a staged PRM rollout campaign (PLAN_v2 §6, job C).
#
# Each array task reads its own shard out of stage_manifest.json: its run dir, its problem
# range, and how many samples per problem the harness has to walk. One run dir per task is
# required, not tidy -- add_to_eval_results_file rewrites the whole eval_results.json, so two
# tasks pointed at one file would each drop the other's results.
#
# Usage:
#   python -m reranker.src.prm.rollout.stage --config reranker/configs/prm_rollout.yaml
#   sbatch --array=0-$(($(scripts/prm_eval.sh --shards) - 1)) scripts/prm_eval.sh
#
# --shards prints the shard count and exits, so the array is sized off what was actually
# staged rather than off eval_shards -- a campaign that staged fewer problems than shards
# gets fewer shards, and a task beyond the last one exits 0 rather than evaluating nothing.
#
# The day below is sized for a full shard. A small campaign should ask for less at submit
# time (`sbatch -t 00:30:00 ...`): the partition backfills short jobs, and a 20-minute run
# asking for a day waits behind everything.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO}/data/prm_rollout}"
KERNELBENCH_DIR="${KERNELBENCH_DIR:-/path/to/workdir/KernelBench}"
MANIFEST="${OUT_DIR}/stage_manifest.json"
# The source runs are all *_triton_*; a wrong backend compiles every kernel the wrong way and
# grades the whole campaign at zero. Overridable, but never silently.
BACKEND="${BACKEND:-triton}"
GPU_ARCH="${GPU_ARCH:-Hopper}"
NUM_GPU_DEVICES="${NUM_GPU_DEVICES:-1}"

if [ ! -f "$MANIFEST" ]; then
    echo "ABORT: no ${MANIFEST} -- run reranker.src.prm.rollout.stage first" >&2
    exit 1
fi

N_SHARDS=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["shards"]))' "$MANIFEST")

if [ "${1:-}" = "--shards" ]; then
    echo "$N_SHARDS"
    exit 0
fi

TASK="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK" -ge "$N_SHARDS" ]; then
    echo "task ${TASK}: only ${N_SHARDS} shards were staged, nothing to do"
    exit 0
fi

read -r RUN_NAME LEVEL LO HI NSAMPLES RUNS_DIR CORRECT PERF TIMEOUT < <(
    python3 - "$MANIFEST" "$TASK" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
s = m["shards"][int(sys.argv[2])]
e = m["eval"]
print(s["run_name"], s["level"], s["subset"][0], s["subset"][1],
      s["num_samples_per_problem"], m["eval_runs_dir"],
      e["num_correct_trials"], e["num_perf_trials"], e["timeout"])
PY
)

echo "== Job info =="
echo "Job ID:        ${SLURM_JOB_ID:-N/A} (array task ${TASK} of ${N_SHARDS})"
echo "Node:          $(hostname)"
echo "Run name:      ${RUN_NAME}"
echo "Level:         ${LEVEL}   problems ${LO}-${HI}   samples/problem ${NSAMPLES}"
echo "Backend:       ${BACKEND}   arch ${GPU_ARCH}"
echo "Runs dir:      ${RUNS_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
echo "==============="

STAGED=$(find "${RUNS_DIR}/${RUN_NAME}" -maxdepth 1 -name '*_kernel.py' 2>/dev/null | wc -l)
if [ "$STAGED" -eq 0 ]; then
    echo "ABORT: no staged kernels in ${RUNS_DIR}/${RUN_NAME}" >&2
    exit 1
fi
echo "Evaluating ${STAGED} staged kernels"

cd "$KERNELBENCH_DIR"
uv run --no-sync python scripts/eval_from_generations.py \
    run_name="${RUN_NAME}" \
    dataset_src=local \
    level="${LEVEL}" \
    subset="[${LO},${HI}]" \
    num_samples_per_problem="${NSAMPLES}" \
    runs_dir="${RUNS_DIR}" \
    backend="${BACKEND}" \
    gpu_arch="[\"${GPU_ARCH}\"]" \
    num_gpu_devices="${NUM_GPU_DEVICES}" \
    timeout="${TIMEOUT}" \
    num_correct_trials="${CORRECT}" \
    num_perf_trials="${PERF}"
