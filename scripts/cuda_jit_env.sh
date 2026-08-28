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
# Works for both venvs. Set PY to the interpreter of the one in use; it defaults to the
# cu130 project venv so existing callers are unchanged.
#
# Safe under `set -euo pipefail`: every inherited path variable is expanded with :-.

PY="${PY:-uv run --no-sync python}"

# flashinfer's JIT shells out to bare `ninja`, which ships in the venv's bin -- and
# calling $PY directly (rather than via `uv run`) leaves that dir off PATH.
VENV_BIN=$($PY -c "import sys, os; print(os.path.dirname(sys.executable))" 2>/dev/null || true)
if [ -n "$VENV_BIN" ]; then
    export PATH="$VENV_BIN:$PATH"
fi

# cu13 ships the whole toolkit as one nvidia.cu13 package, nvcc included. The cu12 wheels
# split it across nvidia/<component>/lib AND ship no nvcc -- PyPI's nvidia-cuda-nvcc-cu12
# is ptxas only -- so that stack takes nvcc from NVIDIA's redistributable archive on
# scratch and collects its libs from the venv instead.
#
# Probe for the nvcc binary, not the package: the cu12 venv also grows an nvidia/cu13
# directory, because nvidia-cuda-nvdisasm publishes cu13-only wheels. It holds no nvcc.
CU13_HOME=$($PY -c "import nvidia.cu13 as m; print(list(m.__path__)[0])" 2>/dev/null || true)
if [ -n "$CU13_HOME" ] && [ -x "$CU13_HOME/bin/nvcc" ]; then
    CUDA_STACK=cu13
    CUDA_HOME="$CU13_HOME"
    CUDA_LIB_DIRS=("$CUDA_HOME/lib")
else
    CUDA_STACK=cu12
    NVCC_ARCHIVE="${CUDA12_NVCC_HOME:-/path/to/cuda-redist/cuda_nvcc-linux-x86_64-12.9.86-archive}"
    if [ ! -x "$NVCC_ARCHIVE/bin/nvcc" ]; then
        echo "!! no nvcc at $NVCC_ARCHIVE -- re-fetch the cuda_nvcc redistributable" >&2
        return 1 2>/dev/null || exit 1
    fi
    # Assemble a cu13-shaped root out of the two halves. flashinfer builds against
    # $CUDA_HOME as if it were a real toolkit -- -I$CUDA_HOME/include, -L$CUDA_HOME/lib64
    # -- but on cu12 nvcc lives in the redistributable and the headers/libs are scattered
    # over nvidia/<component>/{include,lib}, and the nvcc archive carries no
    # cuda_runtime.h at all. Symlinks only, rebuilt each run, so it tracks the venv.
    NV_ROOT=$($PY -c "import nvidia, os.path; print(os.path.dirname(nvidia.__file__))")
    # Per job: array tasks share SLURM_SUBMIT_DIR, and the rm -rf below would otherwise
    # wipe the toolchain a concurrently-running sibling is pointing CUDA_HOME at.
    LINK_ROOT="${SLURM_SUBMIT_DIR:-$PWD}/.cuda_link"
    find "$LINK_ROOT" -maxdepth 1 -name 'cu12root-*' -mtime +2 -exec rm -rf {} + 2>/dev/null || true
    CUDA_HOME="$LINK_ROOT/cu12root${SLURM_JOB_ID:+-$SLURM_JOB_ID}"
    rm -rf "$CUDA_HOME"
    mkdir -p "$CUDA_HOME/bin" "$CUDA_HOME/include" "$CUDA_HOME/lib"
    ln -sfn "$NVCC_ARCHIVE/nvvm" "$CUDA_HOME/nvvm"
    ln -sfn lib "$CUDA_HOME/lib64"
    for f in "$NVCC_ARCHIVE"/bin/*; do ln -sf "$f" "$CUDA_HOME/bin/"; done
    for d in "$NVCC_ARCHIVE"/include "$NV_ROOT"/*/include; do
        [ -d "$d" ] || continue
        for h in "$d"/*; do [ -e "$h" ] && ln -sf "$h" "$CUDA_HOME/include/"; done
    done
    for d in "$NV_ROOT"/*/lib; do
        [ -d "$d" ] || continue
        for so in "$d"/lib*; do [ -e "$so" ] && ln -sf "$so" "$CUDA_HOME/lib/"; done
    done
    # nvcc resolves its own realpath to find crt/ and cicc, which lands it back in the
    # archive's include -- so the merged headers have to be on the search path explicitly.
    export CPATH="$CUDA_HOME/include:${CPATH:-}"
    CUDA_LIB_DIRS=("$CUDA_HOME/lib")
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

# On cu13 the bundled nvcc is 13.2 against a 13.0 cudart, so CCCL's header-vs-compiler
# check aborts the JIT builds; the two are ABI-compatible minor versions. Harmless on
# cu12, where nvcc and cudart are both 12.9.
export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

# The JIT link step passes `-lcudart -lnvrtc`, but the wheels ship only versioned libs
# (libcudart.so.13) under lib/ -- no unversioned symlink, and no lib64/ (which is where
# flashinfer's -L points). Materialize unversioned symlinks in a stable dir and expose it
# via LIBRARY_PATH so ld can resolve -l*. Keyed by stack: the two venvs' libs have the
# same names and a shared dir would leave whichever ran last pointing at the wrong CUDA.
CUDA_LINK_DIR="${SLURM_SUBMIT_DIR:-$PWD}/.cuda_link/$CUDA_STACK"
mkdir -p "$CUDA_LINK_DIR"
for dir in "${CUDA_LIB_DIRS[@]}"; do
    for so in "$dir"/lib*.so.*; do
        [ -e "$so" ] || continue
        stem=$(basename "$so"); stem="${stem%%.so.*}"
        ln -sf "$so" "$CUDA_LINK_DIR/${stem}.so"
    done
done
LIB_PATH=$(IFS=:; echo "${CUDA_LIB_DIRS[*]}")
export LIBRARY_PATH="$CUDA_LINK_DIR:$LIB_PATH:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LIB_PATH:${LD_LIBRARY_PATH:-}"

# Fail here, not 90s later inside a vLLM worker where the retry loop burns every attempt.
if ! command -v ninja >/dev/null; then
    echo "!! no ninja on PATH -- pip install ninja into the venv at $VENV_BIN" >&2
    return 1 2>/dev/null || exit 1
fi

echo "  CUDA stack  : $CUDA_STACK"
echo "  CUDA_HOME   : $CUDA_HOME"
echo "  ninja       : $(command -v ninja)"
nvcc --version | tail -2
