#!/bin/bash
#SBATCH --job-name=prm-search
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
set -euo pipefail
cd /path/to/workdir/ProcessKernel-prm-rollout-v3
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Pinned, not inherited. ~/.bashrc exports this only in an interactive shell, so a job
# submitted non-interactively runs with HF_HOME unset and re-downloads the model into
# $HOME -- 61 GB of gpt-oss-120b against a 200 GB home quota, job 2450388.
export HF_HOME="${HF_HOME:-/path/to/.cache/huggingface}"
# And offline, so a cache that does not hold the model fails in seconds instead of pulling it
# down again: 61 GB for gpt-oss, 149 GB for DeepSeek-V4-Flash. lintloop.sh sets it for the
# same reason, and backend.py's DeepSeek path already assumes callers run offline.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Guard on a cached model, never on the directory: $HOME/.cache/huggingface exists and is
# empty, so `[ -d "$HF_HOME/hub" ]` passes on exactly the cache that would download again.
if ! ls -d "$HF_HOME"/hub/models--* >/dev/null 2>&1; then
    echo "ABORT: no models cached under HF_HOME=$HF_HOME" >&2
    exit 1
fi

# vLLM's sampler JIT-compiles flashinfer's top-k/top-p kernel during memory profiling and
# needs nvcc, which these nodes do not have. See the script for the whole story.
source scripts/cuda_jit_env.sh

CFG="${CFG:-reranker/configs/prm_search_l6.yaml}"
# The PRM and the ORM are ~1.2 GB each in bf16 and are loaded before vLLM. vLLM measures its
# budget as a fraction of TOTAL VRAM, so its utilization has to stay under the free fraction.
uv run --no-sync python -m reranker.src.prm.search.run --config "$CFG" "$@"
