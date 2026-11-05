#!/bin/bash
# 1: same value training as value_ckpt300_mc1jsonl (frozen checkpoint-300 policy, value model from checkpoint-300,
# angle on the held-out mc1 trajectories), but on 8 nodes with the same 512 completions per step: does a larger
# budget let the value function generalize to unseen problems?
# Run from the repo root: bash trl/value_function/expe/1_value_ckpt300_mc1_8nodes.sh

CKPT=/checkpoint/fort/gaetan/projects/value_function4/first_run_qwen4b_ckpts/checkpoint-300
sbatch --nodes=8 --job-name=1_value_ckpt300_mc1_8nodes trl/value_function/run_value.sbatch \
    --run_name 1_value_ckpt300_mc1_8nodes \
    --policy_model_name_or_path $CKPT --value_model_name_or_path $CKPT \
    --value_mc_path mc_value_results/mc1_256trajs_128samples/all.jsonl --eval_on_start true --eval_steps 5 \
    --per_device_eval_batch_size 8 --eval_steps 20
