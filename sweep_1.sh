#!/bin/bash
# First large experiment: does a critic-derived per-token advantage beat GRPO's uniform one?
#
# 5 configurations x 3 seeds x 2 policy learning rates = 30 jobs, 4 GPUs each. The critic is trained in all of
# them (it is built unconditionally), so every run reports `value_r2` and `advantage_std` and stays comparable.
# At `plr=0` the policy is frozen and the run is pure policy evaluation: only the critic learns.
#
#   ./sweep_1.sh           submit
#   ./sweep_1.sh --dry-run print the sbatch commands without submitting

set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=""
[ "${1:-}" = "--dry-run" ] && DRY_RUN="echo"

# my_grpo.sbatch renames the job to its --run_name, so what is already queued or running is exactly the set of
# names to leave alone. Re-running this script therefore submits only what is missing, which is how a job that
# died gets relaunched: no separate script, and no risk of a duplicate.
QUEUED="$(squeue -u "$USER" -h -o '%j' 2>/dev/null || true)"

CRITIC_LR=1e-5
POLICY_LRS=(1e-6 0)
SEEDS=(0 1 2)
PREFIX=1 # first large experiment

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

# Seed outermost, so the first pass submits one seed of every configuration: a queue that only drains part of
# the sweep still covers the whole grid rather than all the seeds of the first few arms.
for seed in "${SEEDS[@]}"; do
    for policy_lr in "${POLICY_LRS[@]}"; do
        for config in "${CONFIGS[@]}"; do
            suffix="${config%%;*}"
            extra="${config#*;}"
            name="${PREFIX}_plr${policy_lr}|vlr${CRITIC_LR}|${suffix}|s${seed}"
            if grep -qxF -- "$name" <<<"$QUEUED"; then
                echo "already queued, skipping: $name"
                continue
            fi
            $DRY_RUN sbatch my_grpo.sbatch \
                --run_name "$name" \
                --output_dir "trainer_output/${name//|/_}" \
                --seed "$seed" \
                --learning_rate "$policy_lr" \
                --critic_learning_rate "$CRITIC_LR" \
                $extra
        done
    done
done
