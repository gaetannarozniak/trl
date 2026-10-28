#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."   # sbatch paths below are relative to the repo root

TPL_SHORT=$'Ground truth solution:\n{solution}'
# The dataset ships human-written solutions to its own problems: a median of 321 tokens, worst case 3209,
# against 8192 of positional headroom, so none of them has to be cut.
TPL_PROOF=$'Ground truth solution:\n{solution}\n\nA worked solution:\n{example_proof}'
# 32768 tokens is Qwen3-8B's own trajectory length. It fits because lr 0 freezes the policy, which drops its
# gradients, Adam states, saved activations and backward: measured peak 66.5 GiB instead of 113.6 at bs 1.
# 16 generations per gpu need 72 GiB of KV at 144 KiB/token, hence 0.35.
QWEN3=(--model_name "Qwen/Qwen3-8B"
      --model_init_kwargs '{"dtype": "bfloat16"}'
      --per_device_train_batch_size 2
      --gradient_accumulation_steps 4
      --max_completion_length 32768
      --gradient_checkpointing True
      # lr 0 discards the policy gradient, and the correction only reweights it: the critic's loss reads
      # rewards and values, never logprobs. Skipping it drops a forward over the whole generation batch.
      --vllm_importance_sampling_correction False
      --vllm_gpu_memory_utilization 0.30
      --vllm_max_model_length 33024)
DELAY=${DELAY:-30}
n=0
# NODES=<n> to change how many nodes each job gets. Two nodes doubles the generation batch to 128 completions
# per step at unchanged work per rank, so it buys throughput rather than a shorter step.
NODES=${NODES:-2}
# ONLY=<glob> submits only the matching arms, e.g. ONLY='9.1[01]_*'. [[ ]] not case: names contain `|`.
sub() { [[ "$2" == ${ONLY:-*} ]] || return 0; n=$((n+1)); [ $n -gt 1 ] && ${DRY:+echo }sleep "$DELAY"; ${DRY:+echo }sbatch --qos="${QOS:-g3_core_shared}" --nodes="$NODES" my_grpo.sbatch "$@"; }

# 9.x reproduces experiment x of expe_0_to_3 .. expe7 on this model. The probe variants of expe 3 and 3.1 are
# absent on purpose: _probe_hidden interleaves a probe after every completion token and builds a dense
# (B, 1, L, L) mask with L = prompt + 2*32768, which is 8.6 GB per sequence for the mask alone.
DS="Pandores/aime-1983-2026-hard"
# 1044/1054 index the full 1065-row set, not the 355-row hard subset, so the overfit runs override DS.
DS_FULL="Pandores/aime-1983-2026"
run() { sub --expe_name "$1" --learning_rate 0 --dataset "$DS" --seed "$2" "${@:3}" "${QWEN3[@]}"; }

# TEST=1 submits the reference alone, to check a new node count or model actually trains before committing
# the other 23 jobs to the queue.
if [ -n "${TEST:-}" ]; then
  run "9.0_plr0|residual|aime|qwen3|s1" 1 --target residual
  exit 0
fi

for s in 1 2; do
  run "9.0_plr0|residual|aime|qwen3|s$s"                            $s --target residual
  run "9.1_plr0|residual|aime|layer12|qwen3|s$s"                    $s --critic_layer 12
  run "9.1_plr0|residual|aime|layer24|qwen3|s$s"                    $s --critic_layer 24
  # for l in 0.999 0.995 0.99; do
  #   run "9.2_plr0|residual|aime|vlambda$l|qwen3|s$s"                $s --value_lambda $l
  # done
  run "9.3_plr0|residual|aime|privileged|qwen3|s$s"                 $s --privileged_template "$TPL_SHORT"
  for l in 0.999 0.995 0.99; do
    run "9.4_plr0|residual|aime|privileged|vlambda$l|qwen3|s$s"     $s --privileged_template "$TPL_SHORT" --value_lambda $l
  done
  run "9.5_plr0|residual|aime|privileged|skipunanimous|qwen3|s$s"   $s --privileged_template "$TPL_SHORT" --value_loss_skip_unanimous True
  # 1044 is AIME 2026 I problem 10 (answer 156): recent and hard, unlike the easy bulk of the set
  run "9.7_plr0|residual|aime|privileged|overfit1044|qwen3|s$s"     $s --privileged_template "$TPL_SHORT" --overfit_sample 1044 --dataset "$DS_FULL"
  run "9.7_plr0|residual|aime|privileged|overfit1054|qwen3|s$s"     $s --privileged_template "$TPL_SHORT" --overfit_sample 1054 --dataset "$DS_FULL"
  run "9.8_plr0|residual|aime|privileged|proof|qwen3|s$s"           $s --privileged_template "$TPL_PROOF" --example_proof True
  # Only the value head trains, on a frozen backbone: a linear probe on the policy's own representation.
  run "9.9_plr0|residual|aime|privileged|headonly|qwen3|s$s"        $s --privileged_template "$TPL_SHORT" --critic_head_only True
  # 9.3 never separated from 9.0 because bfloat16 rounds the critic's 1e-5 Adam step away: take a larger step.
  run "9.11_plr0|residual|aime|privileged|clr1e-4|qwen3|s$s"        $s --privileged_template "$TPL_SHORT" --critic_lr 1e-4
done
