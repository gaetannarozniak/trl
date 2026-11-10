#!/bin/bash
# 6 (pilot): the value function is trained and evaluated on the same N problems (train=test), drawn by the seed among
# the 1024 training problems of 6_scaling_law_mc.sh, for N in {1, 16, 256} and 2 seeds, 500 steps: how noisy is the
# angle across seeds and steps, before choosing the grid of the scaling law M(t, N)? The angles of REINFORCE and on the
# 128 held-out problems are logged as references.
# Run from the repo root once 6_scaling_law_mc.sh is done:
#   bash trl/value_function/expe/6_scaling_law_pilot.sh mc_value_results/<jobid>/all.jsonl

MC=${1:?"usage: bash trl/value_function/expe/6_scaling_law_pilot.sh mc_value_results/<jobid>/all.jsonl"}
CKPT=/checkpoint/fort/gaetan/projects/value_function4/first_run_qwen4b_ckpts/checkpoint-300
for N in 1 16 256; do
    for SEED in 0 1; do
        NAME=6_scaling_law_pilot_N${N}_seed${SEED}
        sbatch --nodes=2 --job-name=$NAME trl/value_function/run_value.sbatch --run_name $NAME \
            --policy_model_name_or_path $CKPT --value_model_name_or_path $CKPT --value_mc_path $MC \
            --value_num_train_problems $N --seed $SEED --max_steps 500 --eval_steps 10
    done
done
