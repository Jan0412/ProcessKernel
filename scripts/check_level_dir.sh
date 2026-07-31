#!/bin/bash
#SBATCH --job-name=check_level_dir
#SBATCH --output=check_level_dir_%j.out
#SBATCH --error=check_level_dir_%j.err
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#
# Gate a staged level dir before generation runs against it. No GPU on purpose: the check
# never loads torch, so this schedules outside the 16-GPU QOS cap.
#
# Usage:  sbatch --dependency=afterok:<convert_jobid> scripts/check_level_dir.sh
#         OUT=KernelBench/level6_new sbatch scripts/check_level_dir.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

OUT="${OUT:-KernelBench/level6}"
MIN_REFS="${MIN_REFS:-16000}"
echo "gating $OUT on $(hostname)"

# A short dir is self-consistent, so check_level_dir alone would pass it: rows lost to a
# dead GPU look identical to rows the guards legitimately rejected. level6_old staged
# 17071 of 18162 rows, so anything under 16000 means rows were lost.
n=$(find "$OUT" -maxdepth 1 -name '*.py' 2>/dev/null | wc -l)
echo "  references: $n (floor $MIN_REFS)"
if [ "$n" -lt "$MIN_REFS" ]; then
    echo "FAIL only $n references, expected >= $MIN_REFS -- conversion lost rows" >&2
    exit 1
fi

# exec so python's exit status becomes the job's -- that is what afterok reads.
exec uv run --no-sync python -m scripts.check_level_dir "$OUT"
