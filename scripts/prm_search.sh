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
CFG="${CFG:-reranker/configs/prm_search_l6.yaml}"
# The PRM and the ORM are ~1.2 GB each in bf16 and are loaded before vLLM. vLLM measures its
# budget as a fraction of TOTAL VRAM, so its utilization has to stay under the free fraction.
uv run --no-sync python -m reranker.src.prm.search.run --config "$CFG" "$@"
