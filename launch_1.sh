#!/bin/bash
# Experiment batch "1": OPSFT variants on the SDFT science task, run for two models.
# Each run is launched on its own node/tmux session via launch_tmux.sh.
#
# Usage: ./launch_1.sh [--qos <qos>]     (--qos is forwarded to every run)
#
# 6 variants x 2 models = 12 runs. Run names are prefixed "1_" and read as 1_<model>_<variant>:
#   sft      plain SFT baseline            (--sft_baseline True)
#   frozen   lr=0 control, weights frozen  (--learning_rate 0)
#   opsft    OPSFT, no whitening           (--whitening_probes 0)
#   whiten3/10/25   whitening_probes sweep (--whitening_probes N)

cd "$(dirname "$0")"

LAUNCH=./launch_tmux.sh
SCRIPT=trl/scripts/opsft_sdft.py
CONFIG=trl/scripts/configs/opsft_sdft.yaml
NPROC=8

QOS_ARGS=("$@")  # forwarded verbatim to launch_tmux.sh (e.g. --qos h200_dev)

# tag|hf-model-id
MODELS=(
    # "qwen|Qwen/Qwen2.5-7B-Instruct"
    "mistral|mistralai/Mistral-7B-Instruct-v0.3"
)

# tag|extra overrides appended to the training command
VARIANTS=(
    "sft|--sft_baseline True"
    "frozen|--learning_rate 0"
    "opsft|--whitening_probes 0"
    "whiten3|--whitening_probes 3"
    "whiten10|--whitening_probes 10"
    "whiten25|--whitening_probes 25"
)

for m in "${MODELS[@]}"; do
    mtag="${m%%|*}"; model="${m#*|}"
    for v in "${VARIANTS[@]}"; do
        vtag="${v%%|*}"; overrides="${v#*|}"
        name="1_${mtag}_${vtag}"
        cmd="torchrun --nproc_per_node=$NPROC $SCRIPT --config $CONFIG \
--run_name '$name' --output_dir /checkpoint/scientific-reasoning/gaetan/ckpt/opsft/$name --model_name $model $overrides"
        echo "==> launching $name"
        "$LAUNCH" "${QOS_ARGS[@]}" --name "$name" "$cmd"
    done
done
