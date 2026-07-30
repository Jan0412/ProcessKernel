#!/bin/bash
#SBATCH --job-name=lintloop
#SBATCH --output=lintloop_%j.out
#SBATCH --error=lintloop_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#
# The pin is required, not a preference: node01 has driver 590.48.01, every other H100
# node (node02-08) has 570.211.01, and torch 2.11+cu130 needs r580+. On those nodes
# torch only warns and reports no CUDA, so vLLM dies at init. Lifting the pin means
# building against cu128, not deleting this comment.
#SBATCH --nodelist=node01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
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
# TRACE=1 additionally records per-token model internals to $OUTPUT_DIR/traces/ -- token
# ids, the top-20 alternatives at every step, the plan prose and the linter's line
# numbers. That is PRM training data; it changes nothing about what is generated, and it
# writes to its own output dir so a traced run is never confused with an untraced one.
#
# REF_DIR points generation at the STAGED level dir -- the same files eval scores. Without
# it a KernelBook row is re-converted in-process UNSCALED, so the model is prompted about a
# 4x4 problem and graded on the 2048x2048 one. See kernel_gen/core/sources.py.
#
# Usage:  sbatch scripts/lintloop.sh <level>                    # KernelBench, 3 rounds x 10 samples
#         SMOKE=1 sbatch scripts/lintloop.sh 1                  # 5 problems x 2 samples x 2 rounds
#         TRACE=1 sbatch scripts/lintloop.sh 1                  # the same, capturing traces
#         THINK_TEMP=0 sbatch scripts/lintloop.sh 1             # single-pass, no plan (own _nothink dir)
#         DATASET=kernelbook sbatch scripts/lintloop.sh 6       # KernelBook at pseudo-level 6
#
# KernelBook is ~17k problems and does not fit one job. Shard it over an array; each task
# takes a balanced slice of the staged ids and writes its OWN run dir under
# <OUTPUT_DIR>/shard_NN -- the jsonl journals are appended without locking, so tasks
# sharing a dir would corrupt them. %14 stays inside the 16-GPU QOS cap.
#
#         DATASET=kernelbook TRACE=1 NUM_SAMPLES=4 \
#             sbatch --array=0-31%14 scripts/lintloop.sh 6
#
# Then, and this is where the result actually lives, from the KernelBench checkout:
#         cd /path/to/workdir/KernelBench
#         sbatch --export=ALL,RUN_NAME=<run>,LEVEL=<level> slum_scripts/eval_from_generations.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

LEVEL="${1:?usage: sbatch scripts/lintloop.sh <level>}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
DATASET="${DATASET:-kernelbench}"
ROUNDS="${ROUNDS:-3}"
POLICY="${POLICY:-severity}"
SMOKE="${SMOKE:-0}"
TRACE="${TRACE:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
# KernelBook has no usable HF path any more (see the header); KernelBench levels do --
# their HF `code` column is byte-identical to the staged files -- so REF_DIR stays opt-in
# there, not least because level1-4 are not staged in this repo.
if [ "$DATASET" = "kernelbook" ]; then
    REF_DIR="${REF_DIR:-$SLURM_SUBMIT_DIR/KernelBench/level${LEVEL}}"
else
    REF_DIR="${REF_DIR:-}"
fi

REF_ARGS=()
if [ -n "$REF_DIR" ]; then
    if [ ! -d "$REF_DIR" ]; then
        echo "!! REF_DIR does not exist: $REF_DIR" >&2
        echo "!! stage it first:  sbatch scripts/create_kernelbook.sh" >&2
        exit 1
    fi
    REF_ARGS=(--ref-dir "$REF_DIR")
fi
# The plan/code split's temperature. >0 enables the two-pass "think" path (plan at this
# temperature, code at --temperature); THINK_TEMP=0 turns it off entirely -- single-pass
# generation, no plan prose. The two are different experiments, so they get different
# output dirs below (--skip-existing keys on lint_loop.jsonl, and pointing one at the
# other's dir would skip every done slot and capture nothing).
THINK_TEMP="${THINK_TEMP:-1.0}"

MODEL_SLUG=$(basename "$MODEL")
TAG=$([ "$DATASET" = "kernelbook" ] && echo "kb" || echo "level")
# OUTPUT_DIR="$SLURM_SUBMIT_DIR/runs/${MODEL_SLUG}_${TAG}${LEVEL}_lintloop_triton_v2"
OUTPUT_DIR="/path/to/workdir/KernelBench/runs/${MODEL_SLUG}_${TAG}${LEVEL}_lintloop_triton_v3"

if [ "$THINK_TEMP" = "0" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_nothink"
fi

TRACE_ARGS=()
if [ "$TRACE" = "1" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_traced"
    TRACE_ARGS=(--trace --trace-topk "${TRACE_TOPK:-20}" --trace-window "${TRACE_WINDOW:-512}")
fi

if [ "$SMOKE" = "1" ]; then
    SCOPE=(--problems 0-4 --num-samples 2)
    ROUNDS=2
    OUTPUT_DIR="${OUTPUT_DIR}_smoke"
else
    SCOPE=(--all --num-samples "$NUM_SAMPLES")
fi

# Array mode. Staged ids are sparse, so shard_ids.py cuts the sorted id LIST (not the id
# range) into equal chunks and returns the range spanning each -- balanced and disjoint.
# Each task gets its own run dir; see the header for why they must not share one.
SHARD_LABEL=""
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ -z "$REF_DIR" ]; then
        echo "!! array mode needs REF_DIR (the staged level dir) to shard the ids" >&2
        exit 1
    fi
    NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:?set NSHARDS: Slurm did not export SLURM_ARRAY_TASK_COUNT}}"
    SHARD_RANGE=$(uv run --no-sync python -m scripts.shard_ids \
        "$REF_DIR" "$SLURM_ARRAY_TASK_ID" "$NSHARDS")
    if [ -z "$SHARD_RANGE" ]; then
        echo "!! could not compute a shard range for task $SLURM_ARRAY_TASK_ID" >&2
        exit 1
    fi
    SHARD_LABEL=$(printf "shard_%02d" "$SLURM_ARRAY_TASK_ID")
    SCOPE=(--problems "$SHARD_RANGE" --num-samples "$NUM_SAMPLES")
    OUTPUT_DIR="$OUTPUT_DIR/$SHARD_LABEL"
fi

echo "============================================================"
echo "  lint-feedback loop (A5) | $DATASET level $LEVEL | smoke=$SMOKE trace=$TRACE think=$THINK_TEMP"
echo "  model  : $MODEL"
echo "  rounds : $ROUNDS (policy=$POLICY)"
echo "  refs   : ${REF_DIR:-<HF dataset, converted in-process>}"
echo "  scope  : ${SCOPE[*]}${SHARD_LABEL:+   ($SHARD_LABEL of $NSHARDS)}"
echo "  out    : $OUTPUT_DIR"
echo "  node   : $(hostname)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

# vLLM's sampler JIT-compiles flashinfer's top-k/top-p kernel during memory profiling and
# needs nvcc, which these nodes do not have. See the script for the whole story.
source "$SLURM_SUBMIT_DIR/scripts/cuda_jit_env.sh"

# A CUDA fault kills the context, so the process cannot recover in-band -- only restart.
# Each attempt resumes via --skip-existing, which reads the slots lint_loop.jsonl already
# journaled, so a retry continues the run instead of redoing it. Bounded at 3: past that
# the failure is not the transient this is here to absorb, and a wrong config should not
# burn the whole 2-day allocation discovering that.
ATTEMPTS="${ATTEMPTS:-3}"
for attempt in $(seq 1 "$ATTEMPTS"); do
    echo
    echo "--- generation attempt $attempt/$ATTEMPTS"
    if uv run --no-sync python -m kernel_gen.arms.lintloop \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --level "$LEVEL" \
        "${REF_ARGS[@]}" \
        "${SCOPE[@]}" \
        --rounds "$ROUNDS" \
        --feedback-policy "$POLICY" \
        --backend triton \
        --option one_shot \
        --temperature 0.6 \
        --think-temperature "$THINK_TEMP" \
        --max-num-seqs "${MAX_NUM_SEQS:-64}" \
        --max-new-tokens 16384 \
        --max-model-len 40960 \
        --output-dir "$OUTPUT_DIR" \
        "${TRACE_ARGS[@]}" \
        --skip-existing; then
        echo "--- generation finished on attempt $attempt"
        break
    fi
    if [ "$attempt" -ge "$ATTEMPTS" ]; then
        echo "!! $ATTEMPTS attempts all died -- giving up. The journal is intact:" >&2
        echo "!! resubmit to resume from $OUTPUT_DIR/lint_loop.jsonl" >&2
        exit 1
    fi
    echo "--- attempt $attempt died; retrying, resuming from the journal" >&2
done

echo
echo "Refined kernels : $(find "$OUTPUT_DIR" -maxdepth 1 -name '*_kernel.py' | wc -l)"
echo "Baseline (r0)   : $(find "$OUTPUT_DIR/rounds/round_0" -maxdepth 1 -name '*_kernel.py' | wc -l)"
if [ "$TRACE" = "1" ]; then
    echo "Traces          : $(find "$OUTPUT_DIR/traces" -name '*.npz' | wc -l) npz, \
$(du -sh "$OUTPUT_DIR/traces" 2>/dev/null | cut -f1)"
    echo "  uv run python -m kernel_gen.inspect_trace --run-dir $OUTPUT_DIR"
fi
echo
# Relative to runs/, not basename: an array task writes to <run>/shard_NN and eval
# resolves runs/$RUN_NAME, so basename would name a directory that does not exist.
RUN_NAME="${OUTPUT_DIR##*/runs/}"
echo "Next -- eval BOTH, and compare those, not the linter's own numbers:"
echo "  cd /path/to/workdir/KernelBench"
echo "  sbatch --export=ALL,RUN_NAME=$RUN_NAME,LEVEL=$LEVEL,NUM_SAMPLES_PER_PROBLEM=$NUM_SAMPLES \\"
echo "      slum_scripts/eval_from_generations.sh"
echo "  sbatch --export=ALL,RUN_NAME=$RUN_NAME/rounds/round_0,LEVEL=$LEVEL,NUM_SAMPLES_PER_PROBLEM=$NUM_SAMPLES \\"
echo "      slum_scripts/eval_from_generations.sh"
