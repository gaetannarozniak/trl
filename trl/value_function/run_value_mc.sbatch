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

# Usage: sbatch [--nodes=N] [--array=0-<J-1>] trl/value_function/run_value_mc.sbatch [--any_value_mc_estimate_arg value ...]
# The J jobs of an array, which wait less in the queue than one big job, split the shards (one per GPU) between them,
# and the last one to finish merges them into all.jsonl
RUN=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
export OUT=mc_value_results/$RUN
export FIRST_SHARD=$(( ${SLURM_ARRAY_TASK_ID:-0} * SLURM_NTASKS ))
export SHARD_COUNT=$(( ${SLURM_ARRAY_TASK_COUNT:-1} * SLURM_NTASKS ))
mkdir -p "$OUT" logs

srun --gpu-bind=single:1 --cpu-bind=cores --output=logs/%x-$RUN-shard$FIRST_SHARD-rank%t.out \
    bash -c 'SHARD=$((FIRST_SHARD + SLURM_PROCID)); VLLM_CACHE_ROOT=/tmp/vllm-cache-$SLURM_PROCID python trl/value_function/value_mc_estimate.py --dataset_name zhuzilin/dapo-math-17k --n_samples_mc 512 \
    --model_id Qwen/Qwen3-4B-Base --out "$OUT/shard_${SHARD}.jsonl" --shard_index $SHARD --shard_count $SHARD_COUNT \
    --max_prompts 1024 --eval_interval 64 --max_model_len 4096 --max_completion_length 2048 "$@"' _ "$@"

# Jobs finishing together both merge, into their own temporary file renamed to the same result
if [ "$(ls "$OUT"/shard_*.jsonl | wc -l)" -eq "$SHARD_COUNT" ]; then
    cat "$OUT"/shard_*.jsonl > "$OUT/all.jsonl.$SLURM_JOB_ID"
    mv "$OUT/all.jsonl.$SLURM_JOB_ID" "$OUT/all.jsonl"
    echo "Merged the $SHARD_COUNT shards into $OUT/all.jsonl"
fi

echo "Finished"
