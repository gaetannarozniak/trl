#!/bin/bash
# SFT and OPSFT on each of a list of base models, at the best hyperparameters found so far.
# Same contract as sweep_lean.sh: names derived by sft_lean.py, live or finished configs skipped,
# so rerunning submits exactly what died.
#
#   DRY=1 ./sweep_models.sh
#   TIME=48:00:00 ./sweep_models.sh
set -euo pipefail

REPO=/storage/home/gaetan/projects/trl
BASE="--name-prefix ${PREFIX:-2_models}"

# All three are already in the HF cache, which HF_HUB_OFFLINE=1 requires. Olmo-3-1025-7B is the
# control every earlier result was measured on; the Instruct share its pretraining and tokenizer, so
# the pair isolates post-training alone; Qwen3.5-4B-Base changes both family and scale.
MODELS=(
    "allenai/Olmo-3-1025-7B"
    "allenai/Olmo-3-7B-Instruct"
    "Qwen/Qwen3.5-4B-Base"
)

# Left without --lr so each config keeps its own default, as every earlier sft run did.
SFT="--trainer sft"
OPSFT="--trainer opsft --use-vllm --vllm-gpu-mem 0.2 --beta 1e-8 --lr 3e-6 \
       --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 1e-1"

CONFIGS=()
for m in "${MODELS[@]}"; do
    CONFIGS+=("$SFT --model $m" "$OPSFT --model $m")
done

declare -A STATE
while IFS='|' read -r name st; do
    case "$st" in
        PENDING | RUNNING | COMPLETING | SUSPENDED | REQUEUED | COMPLETED | TIMEOUT) STATE[$name]=$st ;;
    esac
done < <(sacct -u "$USER" -S "${SINCE:-now-3days}" -X -n -P -o JobName,State)

for cfg in "${CONFIGS[@]}"; do
    name=$("$REPO/.venv/bin/python" "$REPO/trl/scripts/sft_lean.py" $BASE $cfg --print-run-name)
    if [ -n "${STATE[$name]:-}" ] && [ -z "${FORCE:-}" ]; then
        echo "skip    $name (${STATE[$name]})"
    elif [ -n "${DRY:-}" ]; then
        echo "submit  $name"
    else
        "$REPO/trl/scripts/submit_lean.sh" $BASE $cfg
    fi
done
