#!/bin/bash
# Experiment 3 with the yes/no value readout: same probe settings as expe 3, but the value is
# p(yes) / (p(yes) + p(no)) at the probe slots, so the privileged prompt asks for a Yes/No verdict.
# DRY=1 to preview, DELAY=<s> to change the gap between submissions (default 30).
set -euo pipefail
cd "$(dirname "$0")/.."   # sbatch paths below are relative to the repo root

TPL_YESNO=$'Hint: the ground truth solution is\n{solution}. After [[ answer Yes if the attempt below is correct, No otherwise:'
DELAY=${DELAY:-30}
n=0
sub() { n=$((n+1)); [ $n -gt 1 ] && ${DRY:+echo }sleep "$DELAY"; ${DRY:+echo }sbatch --qos="${QOS:-g3_core_shared}" my_grpo.sbatch "$@"; }

for s in 1 2; do
  sub --expe_name "3.1_plr0|residual|privileged|probe|yesno|s$s"         --learning_rate 0 --seed $s --privileged_template "$TPL_YESNO" --probe_token "[[" --yes_no_tokens_as_value True
done
