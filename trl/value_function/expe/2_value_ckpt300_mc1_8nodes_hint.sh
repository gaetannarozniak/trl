#!/bin/bash
# 2: same as experiment 1 (value training on 8 nodes, frozen checkpoint-300 policy, angle on the held-out mc1
# trajectories), but the value model sees "Hint: the solution to the problem is {ground_truth}" after the prompt:
# does knowing the answer let the value function generalize to unseen problems?
# Run from the repo root: bash trl/value_function/expe/2_value_ckpt300_mc1_8nodes_hint.sh

CKPT=/checkpoint/fort/gaetan/projects/value_function4/first_run_qwen4b_ckpts/checkpoint-300
sbatch --nodes=8 --job-name=2_value_ckpt300_mc1_8nodes_hint trl/value_function/run_value.sbatch \
    --run_name 2_value_ckpt300_mc1_8nodes_hint \
    --policy_model_name_or_path $CKPT --value_model_name_or_path $CKPT \
    --value_mc_path mc_value_results/mc1_256trajs_128samples/all.jsonl --eval_on_start true --eval_steps 5 \
    --per_device_eval_batch_size 8 --eval_steps 20 --value_hint true
