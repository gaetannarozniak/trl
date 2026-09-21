#!/bin/bash
# First large experiment: does a critic-derived per-token advantage beat GRPO's uniform one?
#
# 5 configurations x 5 seeds = 25 jobs, 4 GPUs each. The critic is trained in all of them (it is built
# unconditionally), so the baseline still reports `value_r2` and `advantage_std` and stays comparable.
#
#   ./sweep_1.sh           submit
#   ./sweep_1.sh --dry-run print the sbatch commands without submitting

set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=""
[ "${1:-}" = "--dry-run" ] && DRY_RUN="echo"

POLICY_LR=1e-6
CRITIC_LR=1e-5
PREFIX=1  # first large experiment

# name-suffix | extra args
CONFIGS=(
    "gaenone|no_v_adv;--use_value_advantages False"
    "gaenone|v_adv;--use_value_advantages True"
    "gae0.99|v_adv;--use_value_advantages True --gae_lambda 0.99"
    # Raw TD residuals: no per-rollout constant, so the sum is the critic's own V(s_T) - V(s_0)
    "gaenone|v_adv|nosum;--use_value_advantages True --value_advantages_preserve_sum False"
    # Critic regressed on the raw reward R instead of the leave-one-out advantage R - R_bar
    "gaenone|v_adv|targetR;--use_value_advantages True --center_value_target False"
)

for config in "${CONFIGS[@]}"; do
    suffix="${config%%;*}"
    extra="${config#*;}"
    for seed in 0 1 2 3 4; do
        name="${PREFIX}_plr${POLICY_LR}|vlr${CRITIC_LR}|${suffix}|s${seed}"
        $DRY_RUN sbatch my_grpo.sbatch \
            --run_name "$name" \
            --output_dir "trainer_output/${name//|/_}" \
            --seed "$seed" \
            --learning_rate "$POLICY_LR" \
            --critic_learning_rate "$CRITIC_LR" \
            $extra
    done
done
