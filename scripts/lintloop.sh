#!/bin/bash
#SBATCH --job-name=lintloop
#SBATCH --output=lintloop_%j.out
#SBATCH --error=lintloop_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
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
# Usage:  sbatch scripts/lintloop.sh <level>                    # KernelBench, 3 rounds x 10 samples
#         SMOKE=1 sbatch scripts/lintloop.sh 1                  # 5 problems x 2 samples x 2 rounds
#         TRACE=1 sbatch scripts/lintloop.sh 1                  # the same, capturing traces
#         DATASET=kernelbook sbatch scripts/lintloop.sh 5       # KernelBook at pseudo-level 5
#
# Then, and this is where the result actually lives:
#         sbatch scripts/autotune_eval.sh <OUTPUT_DIR> <level>
#         sbatch scripts/autotune_eval.sh <OUTPUT_DIR>/rounds/round_0 <level>

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

MODEL_SLUG=$(basename "$MODEL")
TAG=$([ "$DATASET" = "kernelbook" ] && echo "kb" || echo "level")
# OUTPUT_DIR="$SLURM_SUBMIT_DIR/runs/${MODEL_SLUG}_${TAG}${LEVEL}_lintloop_triton_v2"
OUTPUT_DIR="/path/to/workdir/KernelBench/runs/${MODEL_SLUG}_${TAG}${LEVEL}_lintloop_triton_v2"

TRACE_ARGS=()
if [ "$TRACE" = "1" ]; then
    # Its own dir: --skip-existing keys on lint_loop.jsonl, so pointing a traced run at
    # an untraced run's dir would skip every slot already done and capture nothing.
    OUTPUT_DIR="${OUTPUT_DIR}_traced"
    TRACE_ARGS=(--trace --trace-topk "${TRACE_TOPK:-20}" --trace-window "${TRACE_WINDOW:-512}")
fi

if [ "$SMOKE" = "1" ]; then
    SCOPE=(--problems 0-4 --num-samples 2)
    ROUNDS=2
    OUTPUT_DIR="${OUTPUT_DIR}_smoke"
else
    SCOPE=(--all --num-samples 10)
fi

echo "============================================================"
echo "  lint-feedback loop (A5) | $DATASET level $LEVEL | smoke=$SMOKE trace=$TRACE"
echo "  model  : $MODEL"
echo "  rounds : $ROUNDS (policy=$POLICY)"
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
        "${SCOPE[@]}" \
        --rounds "$ROUNDS" \
        --feedback-policy "$POLICY" \
        --backend triton \
        --option one_shot \
        --temperature 0.6 \
        --think-temperature 1.0 \
        --max-num-seqs "${MAX_NUM_SEQS:-10}" \
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
echo "Next -- eval BOTH, and compare those, not the linter's own numbers:"
echo "  sbatch scripts/autotune_eval.sh $OUTPUT_DIR $LEVEL"
echo "  sbatch scripts/autotune_eval.sh $OUTPUT_DIR/rounds/round_0 $LEVEL"
