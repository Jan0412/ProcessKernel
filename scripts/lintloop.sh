#!/bin/bash
#SBATCH --job-name=lintloop
#SBATCH --output=lintloop_%j.out
#SBATCH --error=lintloop_%j.err
# Listed, not single: shortrun is PriorityTier=1000 against YOUR_PARTITION's 100 and tier
# beats age outright, but sbatch rejects shortrun as the *only* partition. The 24h below
# is load-bearing -- ask for more and shortrun is ineligible and the job silently falls
# back to batch, which is the queue this is here to escape.
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#
# Four GPUs because DeepSeek-V4-Flash is 149 GB of weights (FP8 block-quant attention
# and dense layers, MXFP4 experts) and does not fit two cards. backend.py derives
# tensor_parallel_size from CUDA_VISIBLE_DEVICES, so :4 here IS the TP=4 -- there is no
# separate flag. Each rank holds ~37 GB, leaving ~36 GB per card for KV cache.
#
# The 16-GPU-per-user QOS cap therefore throttles an array at %4, not %8.
#SBATCH --gres=gpu:h100:4
#
# The node01 pin is gone. It existed because that node has driver 590.48.01 while every
# other H100 node (node02-08) has 570.211.01, and the cu130 project venv needs r580+.
# This script now runs the cu12 venv (see PY below), which r570 satisfies, so all seven
# nodes are eligible. Re-pin only if you point PY back at .venv.
#
# Excluded rather than merely unpinned: two of node01's four GPUs answer
# cudaErrorDevicesUnavailable to every context (confirmed 2026-08-03, job 2402571 -- both
# GPUs it was handed failed). The quarantine below survives landing there, but each time
# costs a job and parks a GPU for the sleep. Nothing needs that node now.
#SBATCH --exclude=node01
#SBATCH --ntasks=1
# Both scaled with the GPU count, not raised on their own: 8 CPUs and 96 GB of host
# memory per TP rank, the same per-rank share the 2-GPU gpt-oss config had. The 149 GB
# checkpoint is read by all four ranks, and its page cache counts against the cgroup.
# The nodes carry 224 CPUs and 2 TB, so neither number narrows what we can land on.
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=24:00:00
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
# sharing a dir would corrupt them. At 4 GPUs per task, %4 stays inside the 16-GPU QOS cap.
#
#         DATASET=kernelbook TRACE=1 NUM_SAMPLES=4 \
#             sbatch --array=0-31%4 scripts/lintloop.sh 6
#
# Then, and this is where the result actually lives, from the KernelBench checkout:
#         cd /path/to/workdir/KernelBench
#         sbatch --export=ALL,RUN_NAME=<run>,LEVEL=<level> slum_scripts/eval_from_generations.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# The cu12 venv: torch 2.11.0+cu129 / vLLM 0.26.0+cu129, the newest pair NVIDIA and vLLM
# both publish a CUDA-12 build of. It exists so this script is not confined to node01 --
# see the header. `uv run` is deliberately not used: it resolves the project venv (.venv,
# cu130) and would silently put us back on the driver that only node01 has.
PY="${PY:-$SLURM_SUBMIT_DIR/.venv-cu129/bin/python}"
if [ ! -x "$PY" ]; then
    echo "!! no interpreter at $PY -- build it, or set PY to a venv that exists" >&2
    exit 1
fi

# The model cache lives on scratch. $HOME is a 200 G quota with ~19 G free, which
# DeepSeek-V4-Flash's 149 GB does not fit in; HF's default is ~/.cache/huggingface.
export HF_HOME="${HF_HOME:-/path/to/hf}"
# Resolve from cache only. 32 array tasks all revalidating the same repo against the Hub
# is rate-limit bait, and a task that starts while the network is flaky should fail loudly
# rather than re-download 149 GB onto a compute node.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Humming (the DSv4 MoE backend) JIT-compiles its kernels and caches them under $HOME by
# default. Its own footprint is small -- ~124 MB -- but $HOME is the shared 200 G quota,
# so when anything else fills it these are the writes that fail, and the whole task dies
# at model load with "OSError: [Errno 122] Disk quota exceeded". That cost shards 8-31 of
# run 2405135 while 0-7 had already finished. Scratch has 330 T.
export HUMMING_CACHE_DIR="${HUMMING_CACHE_DIR:-/path/to/jit/humming/cache}"
export HUMMING_TMP_DIR="${HUMMING_TMP_DIR:-/path/to/jit/humming/tmp}"
mkdir -p "$HUMMING_CACHE_DIR" "$HUMMING_TMP_DIR"

LEVEL="${1:?usage: sbatch scripts/lintloop.sh <level>}"
# Repo id, not the snapshot path: OUTPUT_DIR is keyed on `basename $MODEL`, and a snapshot
# path would name the run dir after a commit hash.
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
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
# v5: pack() no longer keeps vLLM's duplicated sampled token (KGEN-25), so a traced run
# records K genuinely distinct alternatives per row. v4's traces were repaired offline to
# a clean K=19 rather than regenerated, and repaired-19 and native-19 are the same thing --
# but only when TRACE_TOPK=19. Sharing a dir across the fix would put rows of different K
# in one journal, and entropy/deepconf_c/tail_mass are not comparable across K.
# v6: prompts.py now routes through apply_deltas, and sampling gained TOP_P/TOP_K so each
# model runs at its own model-card setting instead of one shared temperature. Both change
# what a row means, so v5 and v6 do not share a dir.
OUTPUT_DIR="/path/to/workdir/KernelBench/runs/${MODEL_SLUG}_${TAG}${LEVEL}_lintloop_triton_v6"

# A/B arms whose flags alone would collide with an existing run need their own dir:
# --skip-existing keys on that run's journal and would skip every slot instead of
# generating. RUN_SUFFIX is how a matched control gets one.
if [ -n "${RUN_SUFFIX:-}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_${RUN_SUFFIX}"
fi

# A/B arms must not share a run dir: the jsonl journals are appended without locking and
# --skip-existing keys on them, so two arms in one dir corrupt each other.
PROMPT_DELTAS="${PROMPT_DELTAS:-}"
if [ -n "$PROMPT_DELTAS" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_deltas-$(echo "$PROMPT_DELTAS" | tr ',' '-')"
fi

if [ "$THINK_TEMP" = "0" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_nothink"
fi

# Native reasoning instead of the "## Plan" prefill. Only valid with THINK_TEMP=0 --
# lintloop.py rejects the pair at startup. Its own dir: a thinking corpus and a
# plan corpus are different experiments and --skip-existing keys on the journal.
ENABLE_THINKING="${ENABLE_THINKING:-0}"
THINK_ARGS=()
if [ "$ENABLE_THINKING" = "1" ]; then
    THINK_ARGS=(--enable-thinking)
    OUTPUT_DIR="${OUTPUT_DIR}_thinking"
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
elif [ -n "${PROBLEMS:-}" ]; then
    # An explicit id range, for A/B arms that must cover the same problems as an
    # existing run rather than the whole level.
    SCOPE=(--problems "$PROBLEMS" --num-samples "$NUM_SAMPLES")
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
    SHARD_RANGE=$("$PY" -m scripts.shard_ids \
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
echo "  lint-feedback loop (A5) | $DATASET level $LEVEL | smoke=$SMOKE trace=$TRACE think=$THINK_TEMP native_think=$ENABLE_THINKING"
echo "  model  : $MODEL"
echo "  rounds : $ROUNDS (policy=$POLICY)"
echo "  sample : temp=${TEMPERATURE:-0.6} top_p=${TOP_P:-1.0} top_k=${TOP_K:-0}"
echo "  refs   : ${REF_DIR:-<HF dataset, converted in-process>}"
echo "  scope  : ${SCOPE[*]}${SHARD_LABEL:+   ($SHARD_LABEL of $NSHARDS)}"
echo "  out    : $OUTPUT_DIR"
echo "  node   : $(hostname)"
echo "  python : $PY"
echo "============================================================"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

# node01 was losing GPUs -- two of four answered cudaErrorDevicesUnavailable to every
# context created on them. Slurm hands out the lowest free index and does not retry a
# failed array task, so tasks funnel onto a dead one and drop their shard 90s later. The
# pin is gone, so this is now insurance against any node rather than that one node.
#
# Probe before the model load. On a good GPU fall through to the run; on a dead one hand
# this shard to a fresh task and then sit on the GPU for the duration, so the scheduler
# stops offering it. Each bad GPU therefore costs one job once, not every task forever.
# Depth-bounded: a chain longer than four is a different fault and should stop rather
# than resubmit itself indefinitely.
if ! "$PY" -c "import torch; torch.zeros(1, device='cuda')" 2>/dev/null; then
    depth="${QUARANTINE_DEPTH:-0}"
    echo "!! GPU unusable -- quarantining it (depth $depth)" >&2
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "$depth" -lt 4 ]; then
        sbatch --array="$SLURM_ARRAY_TASK_ID" \
            --export=ALL,QUARANTINE_DEPTH=$((depth + 1)) \
            "$SLURM_SUBMIT_DIR/scripts/lintloop.sh" "$LEVEL"
    else
        echo "!! not resubmitting -- depth $depth or no array id" >&2
    fi
    echo "!! holding this GPU; release with: scancel $SLURM_JOB_ID" >&2
    sleep 23h
    exit 0
fi

# vLLM's sampler JIT-compiles flashinfer's top-k/top-p kernel during memory profiling and
# needs nvcc, which these nodes do not have. See the script for the whole story.
source "$SLURM_SUBMIT_DIR/scripts/cuda_jit_env.sh"

# A CUDA fault kills the context, so the process cannot recover in-band -- only restart.
# Each attempt resumes via --skip-existing, which reads the slots lint_loop.jsonl already
# journaled, so a retry continues the run instead of redoing it. Bounded at 3: past that
# the failure is not the transient this is here to absorb, and a wrong config should not
# burn the whole 2-day allocation discovering that.
# MAX_MODEL_LEN/MAX_NEW_TOKENS are overridable for short-context models: drkernel-14b
# caps at 32768, below the 40960 default, and vLLM refuses to start above the cap.
# TEMPERATURE/TOP_P/TOP_K are overridable because each model card specifies its own
# (Qwen coding 0.6/0.95/20, MiniMax 1.0/0.95/40, Nemotron 1.0/0.95, DeepSeek 1.0/1.0).
ATTEMPTS="${ATTEMPTS:-3}"
for attempt in $(seq 1 "$ATTEMPTS"); do
    echo
    echo "--- generation attempt $attempt/$ATTEMPTS"
    if "$PY" -m kernel_gen.arms.lintloop \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --level "$LEVEL" \
        "${REF_ARGS[@]}" \
        "${SCOPE[@]}" \
        --rounds "$ROUNDS" \
        --feedback-policy "$POLICY" \
        --backend triton \
        --option one_shot \
        --temperature "${TEMPERATURE:-0.6}" \
        --top-p "${TOP_P:-1.0}" \
        --top-k "${TOP_K:-0}" \
        --think-temperature "$THINK_TEMP" \
        --max-num-seqs "${MAX_NUM_SEQS:-64}" \
        --max-new-tokens "${MAX_NEW_TOKENS:-16384}" \
        --max-model-len "${MAX_MODEL_LEN:-40960}" \
        --prompt-deltas "$PROMPT_DELTAS" \
        "${THINK_ARGS[@]}" \
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
    echo "  $PY -m kernel_gen.inspect_trace --run-dir $OUTPUT_DIR"
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
