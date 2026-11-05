#!/bin/bash
#SBATCH --job-name=value-mc-estimation
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=30
#SBATCH --time=48:00:00
#SBATCH --partition=g3
#SBATCH --account=scientific-reasoning
#SBATCH --output=logs/%x-%j.out
#SBATCH --qos=g3_scientific-reasoning_high 
set -euo pipefail
export OUT=mc_value_results/$SLURM_JOB_ID
mkdir -p "$OUT" logs

srun --gpu-bind=single:1 --cpu-bind=cores --output=logs/%x-%j-rank%t.out \
    bash -c 'VLLM_CACHE_ROOT=/tmp/vllm-cache-$SLURM_PROCID python trl/value_function/value_mc_estimate.py --dataset_name zhuzilin/dapo-math-17k --n_samples_mc 512 \
    --model_id Qwen/Qwen3-4B-Base --out "$OUT/shard_${SLURM_PROCID}.jsonl" --shard_index $SLURM_PROCID --shard_count $SLURM_NTASKS \
    --max_prompts 1024 --eval_interval 64 --max_model_len 4096 --max_completion_length 2048 "$@"' _ "$@"

cat "$OUT"/shard_*.jsonl > "$OUT/all.jsonl"

echo "Finished"
