#!/bin/bash
# 4: same as experiment 3 (intermediate credit, K=128, 4 MC samples, 8 nodes), plus 4 extra value updates per step on
# steps drawn from a replay buffer of the last 4 steps: do cheap reused updates help the value function, or does it
# memorize (compare value_replay_loss with value_loss)?
# Needs the value_function4_intermediate_credit branch checked out. From the repo root:
#   bash trl/value_function/expe/4_value_ckpt300_mc1_8nodes_intermediate_credit_replay.sh

CKPT=/checkpoint/fort/gaetan/projects/value_function4/first_run_qwen4b_ckpts/checkpoint-300
sbatch --nodes=8 --job-name=4_value_ckpt300_mc1_8nodes_intermediate_credit_replay trl/value_function/run_value.sbatch \
    --run_name 4_value_ckpt300_mc1_8nodes_intermediate_credit_replay \
    --policy_model_name_or_path $CKPT --value_model_name_or_path $CKPT \
    --value_mc_path mc_value_results/mc1_256trajs_128samples/all.jsonl --eval_on_start true --eval_steps 5 \
    --per_device_eval_batch_size 8 --eval_steps 20 --value_mc_targets_every 128 --value_mc_targets_samples 4 \
    --value_replay_updates 8 --value_replay_buffer_steps 64
