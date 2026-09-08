#!/bin/bash
# Natural gradient against the usual projection, two seeds each, at the centre of the first sweep.
# By the Woodbury identity the two updates partition g*, so this is the whole comparison: 4 runs.
#
#   DRY=1 ./sweep_natgrad.sh
#   TIME=48:00:00 ./sweep_natgrad.sh
set -euo pipefail

REPO=/storage/home/gaetan/projects/trl
BASE="--name-prefix ${PREFIX:-3_natgrad}"
CENTRE="--trainer opsft --use-vllm --vllm-gpu-mem 0.2 --beta 1e-8 --lr 3e-6 \
        --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 1e-1"

CONFIGS=()
for s in 43 44; do
    CONFIGS+=("$CENTRE --seed $s" "$CENTRE --seed $s --natural-gradient")
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
