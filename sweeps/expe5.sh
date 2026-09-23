#!/bin/bash
# DRY=1 to preview, DELAY=<s> to change the gap between submissions (default 30).
set -euo pipefail
cd "$(dirname "$0")/.."   # sbatch paths below are relative to the repo root

TPL_SHORT=$'Ground truth solution:\n{solution}'
DELAY=${DELAY:-30}
n=0
sub() { n=$((n+1)); [ $n -gt 1 ] && ${DRY:+echo }sleep "$DELAY"; ${DRY:+echo }sbatch --qos=g3_core_shared my_grpo.sbatch "$@"; }

for s in 1 2 3; do
  sub --expe_name "5_plr0|residual|privileged|skipunanimous|s$s" \
      --learning_rate 0 --seed $s --privileged_template "$TPL_SHORT" --value_loss_skip_unanimous True
done
