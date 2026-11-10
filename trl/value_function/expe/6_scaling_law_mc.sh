#!/bin/bash
# 6 (MC estimates): the first 2560 problems of the shuffled DAPO are screened with 64 rollouts of checkpoint-300 each,
# and the eligible ones (0.1 <= V0 <= 0.9, about half) get 4 trajectories whose values are estimated with 256 rollouts
# every 64 tokens. Those of index < 128 are the held-out problems, the first 1024 after them the pool of the training
# problems of 6_scaling_law_pilot.sh. 32 jobs of 1 node, each with 4 of the 128 shards (about 1.5 h each), the last one
# to finish merges them into mc_value_results/<array job id>/all.jsonl.
# Run from the repo root: bash trl/value_function/expe/6_scaling_law_mc.sh
# Progress of each GPU: for f in logs/6_scaling_law_mc-<array job id>-*.out; do grep "^shard" $f | tail -1; done

CKPT=/checkpoint/fort/gaetan/projects/value_function4/first_run_qwen4b_ckpts/checkpoint-300
sbatch --array=0-31 --nodes=1 --job-name=6_scaling_law_mc trl/value_function/run_value_mc.sbatch \
    --model_id $CKPT --max_prompts 2560 --screen_samples 64 --num_trajectories 4 --n_samples_mc 256
