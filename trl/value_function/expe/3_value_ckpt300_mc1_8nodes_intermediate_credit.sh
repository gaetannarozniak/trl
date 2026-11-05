#!/bin/bash
# 3: same as experiment 1 (value training on 8 nodes, frozen checkpoint-300 policy, angle on the held-out mc1
# trajectories), with intermediate credit: the value model sees the final reward as a hint, and the tokens in
# [c - 128, c) are regressed on a 4-sample MC estimate of the value after c tokens: does it match the MC shape better?
# Needs the code of the value_function4_intermediate_credit branch. From the repo root:
#   git checkout value_function4_intermediate_credit
#   git show value_function4:trl/value_function/expe/3_value_ckpt300_mc1_8nodes_intermediate_credit.sh | bash

CKPT=/checkpoint/fort/gaetan/projects/value_function4/first_run_qwen4b_ckpts/checkpoint-300
sbatch --nodes=8 --job-name=3_value_ckpt300_mc1_8nodes_intermediate_credit trl/value_function/run_value.sbatch \
    --run_name 3_value_ckpt300_mc1_8nodes_intermediate_credit \
    --policy_model_name_or_path $CKPT --value_model_name_or_path $CKPT \
    --value_mc_path mc_value_results/mc1_256trajs_128samples/all.jsonl --eval_on_start true --eval_steps 5 \
    --per_device_eval_batch_size 8 --eval_steps 20 --value_mc_targets_every 128 --value_mc_targets_samples 4
