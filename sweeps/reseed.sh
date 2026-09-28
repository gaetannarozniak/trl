#!/bin/bash
# Superseded by expe_0_to_3.sh: this is the pre-renaming version, before the 0_-4_ prefixes.
# DRY=1 to preview.
set -euo pipefail
cd "$(dirname "$0")/.."   # sbatch paths below are relative to the repo root

TPL_SHORT=$'Ground truth solution:\n{solution}'
TPL_LONG=$'Hint: the ground truth solution is\n{solution}. Output after [[ your estimate of the probability that you will output the correct solution at the end of your answer.'
SUB="${DRY:+echo }sbatch --qos="${QOS:-g3_core_shared}" my_grpo.sbatch"

for s in 1 2; do
  $SUB --expe_name "plr0|residual|s$s"                                    --learning_rate 0 --seed $s --target residual
  $SUB --expe_name "1_plr0|residual|layer8|s$s"                           --learning_rate 0 --seed $s --critic_layer 8
  $SUB --expe_name "1_plr0|residual|layer12|s$s"                          --learning_rate 0 --seed $s --critic_layer 12
  $SUB --expe_name "1_plr0|residual|layer16|s$s"                          --learning_rate 0 --seed $s --critic_layer 16
  $SUB --expe_name "1_plr0|residual|layer24|s$s"                          --learning_rate 0 --seed $s --critic_layer 24
  $SUB --expe_name "plr0|residual|privileged|s$s"                         --learning_rate 0 --seed $s --privileged_template "$TPL_SHORT"
  $SUB --expe_name "plr0|residual|privileged|probe|s$s"                   --learning_rate 0 --seed $s --privileged_template "$TPL_LONG" --probe_token "[["
  $SUB --expe_name "plr0|residual|vlambda0.999|s$s"                       --learning_rate 0 --seed $s --value_lambda 0.999
  $SUB --expe_name "plr0|residual|vlambda0.995|s$s"                       --learning_rate 0 --seed $s --value_lambda 0.995
  $SUB --expe_name "plr0|residual|vlambda0.99|s$s"                        --learning_rate 0 --seed $s --value_lambda 0.99
  $SUB --expe_name "plr0|residual|privileged|probe|critic_layer12|s$s"    --learning_rate 0 --seed $s --privileged_template "$TPL_LONG" --probe_token "[[" --critic_layer 12
done
