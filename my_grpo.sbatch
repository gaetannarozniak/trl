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

# All args are forwarded untouched; --run_name is also reused as the Slurm job name.
ARGS=("$@")
NAME=""
for i in "${!ARGS[@]}"; do
    case "${ARGS[i]}" in
        --run_name | --run-name) NAME="${ARGS[i + 1]:-}" ;;
        --run_name=* | --run-name=*) NAME="${ARGS[i]#*=}" ;;
    esac
done
[ -n "$NAME" ] && scontrol update JobId="$SLURM_JOB_ID" JobName="$NAME"

cd /storage/home/gaetan/projects/trl
source .venv/bin/activate

echo "job $SLURM_JOB_ID (${NAME:-unnamed}) on $(hostname) | gpus=$SLURM_GPUS_ON_NODE cpus=$SLURM_CPUS_ON_NODE"

accelerate launch --num_processes 4 trl/scripts/my_grpo.py "$@"
