#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Paths ─────────────────────────────────────────────────────────────────────
# Absolute path to the full KernelBench repo (contains scripts/, runs/, results/, src/).
# Adjust this to wherever the repo is cloned on the target machine.
KERNELBENCH_DIR="${HOME}/KernelBench"

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_SLUG="Qwen3-Coder-Next"
LEVEL=1
BACKEND="triton"
RUN_NAME="${MODEL_SLUG}_level${LEVEL}_${BACKEND}"
RUNS_DIR="${KERNELBENCH_DIR}/runs"
HARDWARE="A100"
GPU_ARCH='["Ampere"]'
NUM_GPU_DEVICES=1
NUM_SAMPLES=10
PASS_AT_K="[1,5,10]"

BASELINE_FILE="${KERNELBENCH_DIR}/results/timing/${HARDWARE}/baseline_time_torch.json"

export PYTHONPATH="${KERNELBENCH_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"

# ── Print run info ────────────────────────────────────────────────────────────
echo "============================================"
echo "  Run name    : $RUN_NAME"
echo "  Level       : $LEVEL"
echo "  KernelBench : $KERNELBENCH_DIR"
echo "  Start time  : $(date)"
echo "============================================"
nvidia-smi
echo "============================================"

# ── Verify kernels are present ────────────────────────────────────────────────
STAGED_COUNT=$(ls "${RUNS_DIR}/${RUN_NAME}" 2>/dev/null | wc -l)
if [ "$STAGED_COUNT" -eq 0 ]; then
    echo "ABORT: No kernel files found in ${RUNS_DIR}/${RUN_NAME}"
    exit 1
fi
echo "Found ${STAGED_COUNT} kernel files in ${RUNS_DIR}/${RUN_NAME}:"
ls "${RUNS_DIR}/${RUN_NAME}" | head -5
echo "============================================"

# ── Generate PyTorch eager baseline (skip if already present) ─────────────────
# generate_baseline_time.py has no CLI interface; it reads hardware_name from
# source. Run it from the repo root so REPO_TOP_PATH resolves correctly, and
# pipe "yes" to auto-confirm its interactive prompts.
if [ -f "$BASELINE_FILE" ]; then
    echo "Baseline already exists at ${BASELINE_FILE}, skipping generation"
else
    ( cd "$KERNELBENCH_DIR" && yes | python3 scripts/generate_baseline_time.py )
fi

echo "============================================"

# ── Evaluate generated kernels ────────────────────────────────────────────────
python3 "${KERNELBENCH_DIR}/scripts/eval_from_generations.py" \
    run_name="${RUN_NAME}" \
    runs_dir="${RUNS_DIR}" \
    dataset_src=local \
    level="${LEVEL}" \
    backend="${BACKEND}" \
    gpu_arch="${GPU_ARCH}" \
    num_gpu_devices="${NUM_GPU_DEVICES}" \
    timeout=300 \
    num_correct_trials=5 \
    num_perf_trials=100 \
    num_samples_per_problem="${NUM_SAMPLES}" \
    pass_at_k_values="${PASS_AT_K}"

echo "============================================"

# ── Compute fast_p benchmark score ────────────────────────────────────────────
python3 "${KERNELBENCH_DIR}/scripts/benchmark_eval_analysis.py" \
    run_name="${RUN_NAME}" \
    level="${LEVEL}" \
    hardware="${HARDWARE}" \
    baseline=baseline_time_torch \
    eval_results_dir="${RUNS_DIR}" \
    baseline_file="${BASELINE_FILE}"

echo "============================================"
echo "  End time    : $(date)"
echo "  Exit code   : $?"
echo "============================================"
