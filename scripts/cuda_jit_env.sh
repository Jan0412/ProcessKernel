#!/bin/bash
# Make nvcc available to vLLM's JIT compilers. Source this, do not execute it.
#
# vLLM's sampler routes top-k/top-p through flashinfer, which JIT-compiles its CUDA
# kernels on first use -- during memory profiling, before a single token is generated.
# `module` is not available in the batch shell, so `module load cuda` fails and there is
# no /usr/local/cuda on the compute nodes, so that build dies with:
#
#     RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
#
# Use the nvcc bundled in the venv by the nvidia-cuda-nvcc-cu13 wheel (it matches torch's
# cu130 build) and point CUDA_HOME at it -- no system CUDA, no module system.
#
# Safe under `set -euo pipefail`: every inherited path variable is expanded with :-.

CUDA_HOME=$(uv run --no-sync python -c "import nvidia.cu13 as m; print(list(m.__path__)[0])")
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

# The bundled nvcc wheel is 13.2 while the rest of the CUDA stack (cudart, torch+cu130)
# is 13.0, so CCCL's header-vs-compiler version check aborts the JIT builds. 13.0 and
# 13.2 are ABI-compatible minor versions, so disable that check for every nvcc call.
export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

# The JIT link step passes `-lcudart -lnvrtc`, but the cu13 wheels ship only versioned
# libs (libcudart.so.13) under lib/ -- no unversioned symlink, and no lib64/ (which is
# where flashinfer's -L points). Materialize unversioned symlinks in a stable dir and
# expose it via LIBRARY_PATH so ld can resolve -l*.
CUDA_LINK_DIR="${SLURM_SUBMIT_DIR:-$PWD}/.cuda_link"
mkdir -p "$CUDA_LINK_DIR"
for so in "$CUDA_HOME"/lib/lib*.so.*; do
    [ -e "$so" ] || continue
    stem=$(basename "$so"); stem="${stem%%.so.*}"
    ln -sf "$so" "$CUDA_LINK_DIR/${stem}.so"
done
export LIBRARY_PATH="$CUDA_LINK_DIR:$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

echo "  CUDA_HOME   : $CUDA_HOME"
nvcc --version | tail -2
