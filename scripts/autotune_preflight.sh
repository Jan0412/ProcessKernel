#!/bin/bash
#SBATCH --job-name=autotune_preflight
#SBATCH --output=autotune_preflight_%j.out
#SBATCH --error=autotune_preflight_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#
# Answers the one question blocking everything else: can torch actually use the GPU on this
# node? The login node cannot -- its driver is CUDA 12.8 while the venv's torch is built for
# CUDA 13.0, so torch.cuda.is_available() is False there regardless of the GPUs present.
#
# Then runs the sweep's own smoke test: patch a real kernel, evaluate it, get a number back.
# If this job is green, the harness works and Stage 2 can proceed.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "  node        : $(hostname)"
echo "  job         : $SLURM_JOB_ID"
echo "============================================================"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
echo

echo "--- can torch see the GPU? ---"
uv run --no-sync python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"torch            : {torch.__version__}")
print(f"built for CUDA   : {torch.version.cuda}")
print(f"cuda available   : {ok}")
if ok:
    print(f"device 0         : {torch.cuda.get_device_name(0)}")
    print(f"capability       : {torch.cuda.get_device_capability(0)}")
else:
    raise SystemExit(
        "\nBLOCKED: torch cannot use CUDA on this node either.\n"
        "The venv's torch is built for CUDA 13.0; this node's driver is older.\n"
        "Fix by installing a torch built for the node's driver, or load a newer driver module."
    )
PY
rc=$?
[ $rc -ne 0 ] && exit $rc

echo
echo "--- does triton compile and run here? ---"
uv run --no-sync python - <<'PY'
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

n = 1 << 20
x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
o = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK_SIZE=1024)
torch.cuda.synchronize()
assert torch.allclose(o, x + y), "triton produced the wrong answer"
print(f"triton {triton.__version__}: compiled and ran correctly")
PY
rc=$?
[ $rc -ne 0 ] && exit $rc

echo
echo "============================================================"
echo "  PREFLIGHT PASSED — the GPU path works on this node."
echo "  Next: scripts/autotune_sanity.sh (identity-config check)"
echo "============================================================"
