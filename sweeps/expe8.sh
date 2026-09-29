#!/bin/bash
# Experiment 8: the two reference runs of expe_0_to_3 - plain residual critic (expe 0) and
# privileged (expe 3) - on Olmo-3-7B-Think. Frozen policy, full train set, whole backbone.
# DRY=1 to preview, DELAY=<s> to change the gap between submissions (default 30).
#
# bf16 because the critic deepcopies the backbone: ~14B trainable params per rank, 227 GB in fp32.
# Its Adam states go bf16 too, so at critic_lr 1e-5 the updates round away - first suspect if the
# value loss stays flat, raise it to ~1e-4. batch 4 x accum 8 keeps the 128 completions per step of
# the runs this mirrors. 8192 tokens because Olmo thinks: watch completions/clipped_ratio, and note
# that accuracy_reward grades the *first* \boxed{}, which can be one floated mid-<think>.
set -euo pipefail
cd "$(dirname "$0")/.."   # sbatch paths below are relative to the repo root

TPL_SHORT=$'Ground truth solution:\n{solution}'
OLMO=(--model_name "allenai/Olmo-3-7B-Think"
      --model_init_kwargs '{"dtype": "bfloat16"}'
      --per_device_train_batch_size 4
      --gradient_accumulation_steps 8
      --max_completion_length 8192
      --gradient_checkpointing True
      --vllm_gpu_memory_utilization 0.2
      --vllm_max_model_length 8960)
DELAY=${DELAY:-30}
n=0
sub() { n=$((n+1)); [ $n -gt 1 ] && ${DRY:+echo }sleep "$DELAY"; ${DRY:+echo }sbatch --qos="${QOS:-g3_core_shared}" my_grpo.sbatch "$@"; }

for s in 1 2; do
  sub --expe_name "8_plr0|residual|olmo|s$s"            --learning_rate 0 --seed $s --target residual "${OLMO[@]}"
  sub --expe_name "8_plr0|residual|privileged|olmo|s$s" --learning_rate 0 --seed $s --privileged_template "$TPL_SHORT" "${OLMO[@]}"
done
