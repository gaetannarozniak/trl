#!/bin/bash
# Experiment 7: the expe 3 privileged run, but training on a single sample alone to test overfitting:
# hard 75468 (1/16) and easy 72308 (triple integral, 1/960).
# DRY=1 to preview, DELAY=<s> to change the gap between submissions (default 30).
set -euo pipefail
cd "$(dirname "$0")/.."   # sbatch paths below are relative to the repo root

TPL_SHORT=$'Ground truth solution:\n{solution}'
DELAY=${DELAY:-30}
n=0
sub() { n=$((n+1)); [ $n -gt 1 ] && ${DRY:+echo }sleep "$DELAY"; ${DRY:+echo }sbatch --qos="${QOS:-g3_core_shared}" my_grpo.sbatch "$@"; }

for s in 1 2 3; do
  sub --expe_name "7_plr0|residual|privileged|overfit75468|hard|s$s" --learning_rate 0 --seed $s --privileged_template "$TPL_SHORT" --overfit_sample 75468
  sub --expe_name "7_plr0|residual|privileged|overfit72308|easy|s$s" --learning_rate 0 --seed $s --privileged_template "$TPL_SHORT" --overfit_sample 72308
done
