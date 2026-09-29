#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."   # sbatch paths below are relative to the repo root

TPL_SHORT=$'Ground truth solution:\n{solution}'
# 32768 tokens is Qwen3-8B's own trajectory length. It fits because lr 0 freezes the policy, which drops its
# gradients, Adam states, saved activations and backward: measured peak 66.5 GiB instead of 113.6 at bs 1.
# 16 generations per gpu need 72 GiB of KV at 144 KiB/token, hence 0.35.
QWEN3=(--model_name "Qwen/Qwen3-8B"
      --model_init_kwargs '{"dtype": "bfloat16"}'
      --per_device_train_batch_size 1
      --gradient_accumulation_steps 16
      --max_completion_length 32768
      --gradient_checkpointing True
      --vllm_gpu_memory_utilization 0.35
      --vllm_max_model_length 33024)
DELAY=${DELAY:-30}
n=0
sub() { n=$((n+1)); [ $n -gt 1 ] && ${DRY:+echo }sleep "$DELAY"; ${DRY:+echo }sbatch --qos="${QOS:-g3_core_shared}" my_grpo.sbatch "$@"; }

for s in 1 2; do
  sub --expe_name "9_plr0|residual|aime|qwen3|s$s"            --learning_rate 0 --dataset "Pandores/aime-1983-2026" --seed $s --target residual "${QWEN3[@]}"
  sub --expe_name "9_plr0|residual|aime|privileged|qwen3|s$s" --learning_rate 0 --dataset "Pandores/aime-1983-2026" --seed $s --privileged_template "$TPL_SHORT" "${QWEN3[@]}"
done
