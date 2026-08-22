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

# The generation-model cache, NOT the reranker one ~/.bashrc names. That cache's hub/ was
# recreated on 2026-08-20 and lost the gpt-oss symlink, so vLLM died offline three minutes
# in (job 2474382); this one has held gpt-oss-120b untouched since Aug 3. The two scorers
# load from reranker/data/checkpoints_*/final, which carry their own tokenizer.json, so
# nothing in this job needs the reranker cache.
#
# Assigned, not defaulted with :-. ~/.bashrc exports the other path in an interactive
# shell, and sbatch --export=ALL would carry it in and silently win.
export HF_HOME=/path/to/hf
# And offline, so a cache that does not hold the model fails in seconds instead of pulling it
# down again: 61 GB for gpt-oss, 149 GB for DeepSeek-V4-Flash. lintloop.sh sets it for the
# same reason, and backend.py's DeepSeek path already assumes callers run offline.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Guard on THIS model, never on the directory and never on `models--*`: the wiped cache
# still held an unrelated reranker, so the glob passed on exactly the cache that lacked
# the model. Read the name out of the config rather than hardcoding it.
GEN_MODEL="$(sed -n 's/^  gen_model: *//p' reranker/configs/prm_rollout_l6_r0.yaml | head -1)"
if ! ls -d "$HF_HOME/hub/models--${GEN_MODEL//\//--}" >/dev/null 2>&1; then
    echo "ABORT: $GEN_MODEL not cached under HF_HOME=$HF_HOME" >&2
    exit 1
fi

# vLLM's sampler JIT-compiles flashinfer's top-k/top-p kernel during memory profiling and
# needs nvcc, which these nodes do not have. See the script for the whole story.
source scripts/cuda_jit_env.sh

CFG="${CFG:-reranker/configs/prm_search_l6.yaml}"
# The PRM and the ORM are ~1.2 GB each in bf16 and are loaded before vLLM. vLLM measures its
# budget as a fraction of TOTAL VRAM, so its utilization has to stay under the free fraction.
uv run --no-sync python -m reranker.src.prm.search.run --config "$CFG" "$@"
