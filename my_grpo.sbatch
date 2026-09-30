#!/bin/bash
#SBATCH --job-name=my_grpo
#SBATCH --partition=g3
#SBATCH --qos=g3_scientific-reasoning_high
#SBATCH --account=scientific-reasoning
#SBATCH --nodes=1
# One node unless the submitter asks for more: `sbatch --nodes=N` overrides the directive above, and
# everything below sizes itself from SLURM_NNODES. Twice the nodes is twice the generation batch at the same
# work per rank, since generation_batch_size = per_device x num_processes x steps_per_generation.
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

GPUS=4
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
PORT=$((29500 + SLURM_JOB_ID % 1000))   # two jobs on one node must not share a rendezvous port

echo "job $SLURM_JOB_ID (${NAME:-unnamed}) on $SLURM_NNODES node(s), master $MASTER port $PORT | $((SLURM_NNODES * GPUS)) processes"

# One accelerate per node, each owning that node's 4 GPUs. `machine_rank` has to expand inside the task, so
# it is single-quoted here and evaluated by the remote shell; `_ "$@"` forwards the sweep's arguments.
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c '
accelerate launch \
    --num_machines '"$SLURM_NNODES"' \
    --num_processes '"$((SLURM_NNODES * GPUS))"' \
    --machine_rank $SLURM_PROCID \
    --main_process_ip '"$MASTER"' \
    --main_process_port '"$PORT"' \
    trl/scripts/my_grpo.py "$@"' _ "$@"
