#!/bin/bash
#SBATCH --job-name=kernelbook_sample
#SBATCH --output=kernelbook_sample_%j.out
#SBATCH --error=kernelbook_sample_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc

# Activate your virtual environment
# source vllm-venv/.vllm-venv/bin/activate
# export UV_PROJECT_ENVIRONMENT=.vllm-venv

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- Set the CUDA compatibility layers permanently for this job ---
# export VLLM_ENABLE_CUDA_COMPATIBILITY=1
# export VLLM_CUDA_COMPATIBILITY_PATH=/path/to/workdir/ProcessKernel-1/vllm-venv/my-compat-env/cuda-compat

# ── Config ───────────────────────────────────────────────────────────────────
# FP8 requires Hopper (H100+). Use the bf16 model on A100 unless you know
# the job landed on an H100 node.
# MODEL="Qwen/Qwen3-Coder-Next"
MODEL="openai/gpt-oss-120b" # H100 only
# MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
# MODEL="deepseek-ai/DeepSeek-V4-Flash"
# MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"   # H100 only

PSEUDO_LEVEL=5
# ROWS="0-3639"
# ROWS="3640-7279"
# ROWS="7280-10919"
# ROWS="10920-14559"
ROWS="14560-18200"

NUM_SAMPLES=10

MODEL_SLUG=$(basename "$MODEL")
OUTPUT_DIR="$SLURM_SUBMIT_DIR/runs/${MODEL_SLUG}_kernelbook_level${PSEUDO_LEVEL}_real6_triton"

echo "============================================"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Job name    : $SLURM_JOB_NAME"
echo "  Node        : $SLURM_NODELIST"
echo "  GPUs        : $CUDA_VISIBLE_DEVICES"
echo "  Model       : $MODEL"
echo "  Rows        : $ROWS"
echo "  Num samples : $NUM_SAMPLES"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Start time  : $(date)"
echo "============================================"
nvidia-smi
echo "============================================"

uv sync --frozen

uv run python kernel_gen/generate_kernelbook_samples.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --rows "$ROWS" \
    --pseudo-level "$PSEUDO_LEVEL" \
    --num-samples "$NUM_SAMPLES" \
    --backend triton \
    --temperature 0.3 \
    --think-temperature 1.0 \
    --option one_shot \
    --gpu-name A100 \
    --max-new-tokens 16384 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --skip-existing
RC=$?

echo "============================================"
echo "  End time    : $(date)"
echo "  Exit code   : $RC"
echo "============================================"
exit $RC
