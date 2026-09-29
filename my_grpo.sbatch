#!/bin/bash
#SBATCH --job-name=my_grpo
#SBATCH --partition=g3
#SBATCH --qos=g3_scientific-reasoning_high
#SBATCH --account=scientific-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=36
#SBATCH --time=48:00:00
#SBATCH --output=/storage/home/gaetan/slurm-logs/%x-%j.out
#SBATCH --error=/storage/home/gaetan/slurm-logs/%x-%j.out

set -euo pipefail

# All args are forwarded untouched; --expe_name is also reused as the Slurm job name.
ARGS=("$@")
NAME=""
for i in "${!ARGS[@]}"; do
    case "${ARGS[i]}" in
        --expe_name | --expe-name) NAME="${ARGS[i + 1]:-}" ;;
        --expe_name=* | --expe-name=*) NAME="${ARGS[i]#*=}" ;;
    esac
done
[ -n "$NAME" ] && scontrol update JobId="$SLURM_JOB_ID" JobName="$NAME"

cd /storage/home/gaetan/projects/trl
source .venv/bin/activate

# Every rank resolves the same model and dataset at startup. Submitting a sweep at once puts dozens of them on
# the Hub API within seconds, which answers HTTP 429 and takes the job down before training starts. Both are in
# the local cache already, so resolve from there and never reach for the network.
export HF_HUB_OFFLINE=1

# The fp32 logits are one 37 GiB block (batch x completion x 152k vocab). With a second vLLM engine resident
# there is no contiguous room for it, and the allocator falls back to cudaMalloc/cudaFree every step.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Triton defaults its cache to ~/.triton, which is NFS: the four vLLM engines compile the same kernels at the
# same instant and one rank's atomic rename replaces a file another already has open, which surfaces as
# `OSError: [Errno 116] Stale file handle`. /tmp is node-local, and one tree per job cannot collide at all.
export TRITON_CACHE_DIR=/tmp/triton-$USER-$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-$USER-$SLURM_JOB_ID

echo "job $SLURM_JOB_ID (${NAME:-unnamed}) on $(hostname) | gpus=$SLURM_GPUS_ON_NODE cpus=$SLURM_CPUS_ON_NODE"

accelerate launch --num_processes 4 trl/scripts/my_grpo.py "$@"
