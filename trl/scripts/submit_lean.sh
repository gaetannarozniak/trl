#!/bin/bash
# Launch sft_lean.py on one 8-GPU node, from the login node.
#
#   ./submit_lean.sh --trainer opsft --lr 1e-5 --n-train 600 --score-norm-probes 8 --score-projection-steps 3
#
# Every argument is forwarded verbatim to sft_lean.py. The slurm job name, the wandb run and the
# checkpoint directory all take the name sft_lean.py derives from those arguments, so a job is
# identifiable from `squeue` alone. Override the resources through the environment:
#
#   TIME=4:00:00 QOS=h200_dev ./submit_lean.sh --trainer opsft ...
#
# Re-executes itself through sbatch: the first pass (on the login node) only resolves the run name.
set -euo pipefail

REPO=/storage/home/gaetan/projects/trl
LOGS=${LOGS:-$HOME/slurm_logs}
PARTITION=${PARTITION:-h200}
ACCOUNT=${ACCOUNT:-scientific-reasoning}
# Not h200_dev: it caps a job at 24h of wall and a user at 16 GPUs, and a run needs neither cap.
# The project QOS has no wall of its own, so the ceiling is the partition's 7 days.
QOS=${QOS:-h200_scientific-reasoning_high}
# 13900 steps at ~60 s each is out of reach at any wall clock, so TIME is what actually decides how
# long a run trains and every healthy run ends as TIMEOUT. 96h buys ~5700 steps, comfortably past
# the ~3000 where the eval curves have separated.
TIME=${TIME:-96:00:00}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    NAME=$("$REPO/.venv/bin/python" "$REPO/trl/scripts/sft_lean.py" "$@" --print-run-name)
    mkdir -p "$LOGS"
    echo "submitting $NAME  (logs: $LOGS/$NAME-<jobid>.out)"
    exec sbatch \
        --job-name="$NAME" \
        --output="$LOGS/$NAME-%j.out" \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --qos="$QOS" \
        --time="$TIME" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gpus-per-node=8 \
        --cpus-per-task=96 \
        --exclusive \
        "$0" "$@"
fi

# --- inside the allocation. accelerate spawns the 8 ranks itself, so this is a single srun-less task.
cd "$REPO"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Every rank calls load_dataset(), which takes a FileLock inside the NFS-backed HF cache. Slurm
# releases a whole sweep in the same second, and the create/unlink race on that lock file hands
# ESTALE to whoever loses -- it killed 13 of 20 jobs two minutes in. $RANDOM rather than anything
# derived from SLURM_JOB_ID: the ids of a sweep are consecutive, so a modulus of them stays
# consecutive and spreads nothing.
sleep $((RANDOM % ${STAGGER:-240}))

"$REPO/.venv/bin/accelerate" launch \
    --config_file "$REPO/examples/accelerate_configs/deepspeed_zero2.yaml" \
    "$REPO/trl/scripts/sft_lean.py" "$@"
