#!/bin/bash
#SBATCH --job-name=create_kernelbook
#SBATCH --output=create_kernelbook_%j.out
#SBATCH --error=create_kernelbook_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
# Conversion runs on the cu128 venv (see CONVERT_PY), so the 570.211.01 nodes work and the
# r580 pin is gone. node01 is EXCLUDED: its assigned GPU-xxxxxxxx reports available but
# every kernel launch returns cudaErrorDevicesUnavailable (verified on 4 allocations).
#SBATCH --exclude=node01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Host RAM, NOT guarded by --smoke-mem-gb (that caps CUDA only). get_inputs() builds on
# CPU before moving to the GPU, so a large row peaks here first. 32G died after 250 rows,
# 192G after 6092. Kept small deliberately: the QOS caps a user at 2056G total and this
# account is shared, so a request bigger than the gaps that open never gets scheduled.
# Bound growth with 500-row ROWS chunks instead -- each chunk is a fresh process.
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#
# Stage KernelBook as a KernelBench local level. The output dir is the pipeline's single
# source of truth: KernelBench's eval scores against it, lintloop.sh prompts from it via
# --ref-dir. Measured rate ~0.27 s/row, so a full run is ~82 min.
#
# --smoke-mem-gb is a ceiling for the eval budget, not the card size: eval holds the
# reference and the candidate model live at once, so a row must fit in ~half the H100.
# Re-running the same --out resumes (existing files count as skipped_exists); never point
# two concurrent jobs at one --out, they clobber each other's manifest.
#
# Usage:  sbatch scripts/create_kernelbook.sh                        # -> KernelBench/level6
#         OUT=KernelBench/level6_new ROWS=0-199 sbatch scripts/create_kernelbook.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Separate cu128 venv: the repo .venv is torch cu130 for vLLM and must stay that way, but
# cu130 only runs on node01 whose GPU is faulty. cu128 also matches the KernelBench eval
# venv (torch 2.10.0+cu128), so rows are validated in the environment that grades them.
CONVERT_PY="${CONVERT_PY:-/path/to/workdir/venv-convert-cu128/bin/python}"
OUT="${OUT:-KernelBench/level6}"
ROWS="${ROWS:-}"
SMOKE_MEM_GB="${SMOKE_MEM_GB:-32}"
MAX_NUMEL_RATIO="${MAX_NUMEL_RATIO:-50}"
SCALE_FALLBACK="${SCALE_FALLBACK:-1.0,0.5,0.25}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-300}"

if [ -n "$ROWS" ]; then
    SCOPE=(--rows "$ROWS")
else
    SCOPE=(--all)
fi

echo "============================================================"
echo "  stage KernelBook -> $OUT"
echo "  scope  : ${SCOPE[*]}"
echo "  guards : mem<=${SMOKE_MEM_GB}GiB/row, output<=${MAX_NUMEL_RATIO}x input elems, ${SMOKE_TIMEOUT}s/row"
echo "  ladder : $SCALE_FALLBACK (then unscaled)"
echo "  node   : $(hostname)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# The budget is applied as a fraction of the visible card, so a smaller card disables it.
TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$TOTAL_MIB" -lt 70000 ]; then
    echo "!! card has only ${TOTAL_MIB} MiB; --smoke-mem-gb ${SMOKE_MEM_GB} would not bind." >&2
    exit 1
fi

# is_available() is not enough: on the faulty node01 device it returned True while every
# launch failed, and the converter charged that to 6092 rows instead of aborting. Require
# a real launch plus a sync, so a bad GPU fails the job here rather than staging 0 rows.
if ! "$CONVERT_PY" -c "import torch; torch.zeros(1, device='cuda').sum().item()" 2>/dev/null; then
    echo "!! GPU on $(hostname) cannot launch a kernel (driver $(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version | head -1)); refusing to stage an unguarded dir." >&2
    exit 1
fi
echo "  torch CUDA  : launch OK (driver $(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version | head -1))"

"$CONVERT_PY" kernel_gen/convert_kernelbook.py \
    "${SCOPE[@]}" \
    --out "$OUT" \
    --smoke-test \
    --smoke-device cuda \
    --smoke-timeout "$SMOKE_TIMEOUT" \
    --smoke-mem-gb "$SMOKE_MEM_GB" \
    --max-numel-ratio "$MAX_NUMEL_RATIO" \
    --scale uniform \
    --scale-fallback "$SCALE_FALLBACK"

echo
echo "Staged: $(find "$OUT" -maxdepth 1 -name '*.py' | wc -l) references"
echo
echo "Verify:  uv run --no-sync python -m scripts.check_level_dir $OUT"
echo "Then  :  DATASET=kernelbook TRACE=1 NUM_SAMPLES=4 REF_DIR=\$PWD/$OUT \\"
echo "             sbatch --array=0-31%14 scripts/lintloop.sh 6"
