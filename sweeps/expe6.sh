#!/bin/bash
# LLM-generated privileged information: a frozen analyser reviews the group's rollouts against the
# reference and its structured feedback is handed to the critic.
# DRY=1 to preview, DELAY=<s> to change the gap between submissions (default 30).
set -euo pipefail
cd "$(dirname "$0")/.."   # sbatch paths below are relative to the repo root

TPL=$'Ground truth solution:\n{solution}\n\nFeedback on attempts at this problem:\n{analysis}'
DELAY=${DELAY:-30}
n=0
sub() { n=$((n+1)); [ $n -gt 1 ] && ${DRY:+echo }sleep "$DELAY"; ${DRY:+echo }sbatch --qos="${QOS:-g3_core_shared}" my_grpo.sbatch "$@"; }

for s in 1 2 3; do
  sub --expe_name "6_plr0|residual|analysis|s$s" \
      --learning_rate 0 --seed $s \
      --privileged_template "$TPL" \
      --value_loss_skip_unanimous True \
      --analysis_model "Qwen/Qwen2.5-0.5B-Instruct" \
      --vllm_gpu_memory_utilization 0.3
done
