#!/bin/bash
# Experiment 11: expe 9 on Qwen3-8B-Base instead of Qwen3-8B, privileged against not.
# DRY=1 to preview, DELAY=<s> to change the gap between submissions (default 30).
#
# Same Qwen3ForCausalLM, same vocab, and the chat template renders byte-identical down to the privileged
# user turn, so the critic, vLLM and TPL_SHORT all carry over untouched. Two things do not. The base
# checkpoint's eos is <|endoftext|> while the template closes a turn with <|im_end|>, so vLLM would run
# every rollout to the cap and TRL would then read them all as truncated: the stop id halts the engine,
# --eos_token fixes clipped_ratio and value_ev_nontruncated. And it has 32768 positions against 40960,
# so the budget drops from 33024.
set -euo pipefail
cd "$(dirname "$0")/.."   # sbatch paths below are relative to the repo root

TPL_SHORT=$'Ground truth solution:\n{solution}'
QWEN3BASE=(--model_name "Qwen/Qwen3-8B-Base"
      --model_init_kwargs '{"dtype": "bfloat16"}'
      --eos_token '<|im_end|>'
      --generation_kwargs '{"stop_token_ids": [151645]}'
      --per_device_train_batch_size 2
      --gradient_accumulation_steps 4
      --max_completion_length 32512
      --gradient_checkpointing True
      --vllm_importance_sampling_correction False
      --vllm_gpu_memory_utilization 0.15
      --vllm_max_model_length 32768)
DS="Pandores/aime-1983-2026"
DELAY=${DELAY:-30}
n=0
NODES=${NODES:-2}
# 0.15 not 0.30: the base model does not think, so completions measure ~1.4k tokens and need far less KV.
# ONLY=<glob> submits just the matching arms, e.g. ONLY='*privileged*' to redo one.
sub() { [[ "$2" == ${ONLY:-*} ]] || return 0; n=$((n+1));[ $n -gt 1 ] && ${DRY:+echo }sleep "$DELAY"; ${DRY:+echo }sbatch --qos="${QOS:-g3_core_shared}" --nodes="$NODES" my_grpo.sbatch "$@"; }

for s in 1; do
  sub --expe_name "11_plr0|residual|aime|qwen3base|s$s"                 --learning_rate 0 --dataset "$DS" --seed $s --target residual "${QWEN3BASE[@]}"
  sub --expe_name "11_plr0|residual|aime|privileged|qwen3base|s$s"      --learning_rate 0 --dataset "$DS" --seed $s --privileged_template "$TPL_SHORT" "${QWEN3BASE[@]}"
done
