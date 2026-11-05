#!/bin/bash
#SBATCH --job-name=value-vf4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=120
#SBATCH --time=48:00:00
#SBATCH --partition=g3
#SBATCH --account=scientific-reasoning
#SBATCH --qos=g3_scientific-reasoning_high
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

# Usage: sbatch [--nodes=N] trl/value_function/run_value.sbatch --run_name my-run [--any_train_value_field value ...]
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# Gradient accumulation keeps 512 completions per step (micro-batch 8) whatever the node count; passing it explicitly
# overrides it. The vLLM compile cache is node-local: concurrent compiles in the shared home race on NFS
srun bash -c 'VLLM_CACHE_ROOT=/tmp/vllm-cache-$USER accelerate launch \
    --num_machines $SLURM_NNODES --num_processes $((SLURM_NNODES * 4)) \
    --machine_rank $SLURM_NODEID --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    trl/value_function/train_value.py --gradient_accumulation_steps $((16 / SLURM_NNODES)) "$@"' _ "$@"
