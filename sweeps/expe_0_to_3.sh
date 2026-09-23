#!/bin/bash
# DRY=1 to preview, DELAY=<s> to change the gap between submissions (default 30).
set -euo pipefail
cd "$(dirname "$0")/.."   # sbatch paths below are relative to the repo root

TPL_SHORT=$'Ground truth solution:\n{solution}'
TPL_LONG=$'Hint: the ground truth solution is\n{solution}. Output after [[ your estimate of the probability that you will output the correct solution at the end of your answer.'
DELAY=${DELAY:-30}
n=0
sub() { n=$((n+1)); [ $n -gt 1 ] && ${DRY:+echo }sleep "$DELAY"; ${DRY:+echo }sbatch --qos=g3_core_shared my_grpo.sbatch "$@"; }

for s in 1 2; do
  sub --expe_name "0_plr0|residual|s$s"                          --learning_rate 0 --seed $s --target residual
  sub --expe_name "1_plr0|residual|layer8|s$s"                   --learning_rate 0 --seed $s --critic_layer 8
  sub --expe_name "1_plr0|residual|layer12|s$s"                  --learning_rate 0 --seed $s --critic_layer 12
  sub --expe_name "1_plr0|residual|layer16|s$s"                  --learning_rate 0 --seed $s --critic_layer 16
  sub --expe_name "1_plr0|residual|layer24|s$s"                  --learning_rate 0 --seed $s --critic_layer 24
  sub --expe_name "2_plr0|residual|vlambda0.999|s$s"             --learning_rate 0 --seed $s --value_lambda 0.999
  sub --expe_name "2_plr0|residual|vlambda0.995|s$s"             --learning_rate 0 --seed $s --value_lambda 0.995
  sub --expe_name "2_plr0|residual|vlambda0.99|s$s"              --learning_rate 0 --seed $s --value_lambda 0.99
  sub --expe_name "3_plr0|residual|privileged|s$s"               --learning_rate 0 --seed $s --privileged_template "$TPL_SHORT"
  sub --expe_name "3_plr0|residual|privileged|probe|s$s"         --learning_rate 0 --seed $s --privileged_template "$TPL_LONG" --probe_token "[["
  sub --expe_name "3_plr0|residual|privileged|probe|layer12|s$s" --learning_rate 0 --seed $s --privileged_template "$TPL_LONG" --probe_token "[[" --critic_layer 12
done
