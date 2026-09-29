#!/bin/bash
# Experiment 10: expe 9 on Qwen3.5-9B instead of Qwen3-8B.
# DRY=1 to preview, DELAY=<s> to change the gap between submissions (default 30).
#
# Qwen3.5 is multimodal (Qwen3_5ForConditionalGeneration), which costs two things: the critic reads
# its width from config.get_text_config(), and my_grpo hands the trainer an AutoTokenizer so the
# batched processor path is skipped. The vision tower is dead weight here - the critic deepcopies it
# and trains it on nothing - but it is only 0.5B of the 8.8B. batch 2 x accum 16 keeps the usual 128
# completions per step: the 248320 vocab makes the fp32 logits block 1.6x Qwen3-8B's, so bs 4 would
# leave only ~20 GiB of headroom.
set -euo pipefail
cd "$(dirname "$0")/.."   # sbatch paths below are relative to the repo root

TPL_SHORT=$'Ground truth solution:\n{solution}'
QWEN35=(--model_name "Qwen/Qwen3.5-9B"
      --model_init_kwargs '{"dtype": "bfloat16"}'
      --per_device_train_batch_size 2
      --gradient_accumulation_steps 16
      --max_completion_length 8192
      --gradient_checkpointing True
      --ddp_find_unused_parameters True
      --vllm_gpu_memory_utilization 0.2
      --vllm_max_model_length 8960)
DELAY=${DELAY:-30}
n=0
sub() { n=$((n+1)); [ $n -gt 1 ] && ${DRY:+echo }sleep "$DELAY"; ${DRY:+echo }sbatch --qos="${QOS:-g3_core_shared}" my_grpo.sbatch "$@"; }

for s in 1 2; do
  sub --expe_name "10_plr0|residual|aime|qwen3.5|s$s"            --learning_rate 0 --dataset "Pandores/aime-1983-2026" --seed $s --target residual "${QWEN35[@]}"
  sub --expe_name "10_plr0|residual|aime|privileged|qwen3.5|s$s" --learning_rate 0 --dataset "Pandores/aime-1983-2026" --seed $s --privileged_template "$TPL_SHORT" "${QWEN35[@]}"
done
