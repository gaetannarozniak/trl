#!/bin/bash
# 20-run OPSFT hyperparameter sweep, one 8-GPU job per config, from the login node.
#
#   ./sweep_lean.sh          # submit all 20
#   DRY=1 ./sweep_lean.sh    # print the run names, submit nothing
#   BETA=1e-8 ./sweep_lean.sh
#
# No --expe-name anywhere: sft_lean.py derives the wandb run, the slurm job name and the
# checkpoint directory from the arguments, so a name can never disagree with what actually ran.
#
# Resources come from submit_lean.sh's own defaults (project QOS, 96h) -- one source of truth. A
# TIME= or QOS= in this script's environment still reaches sbatch, since it is inherited straight
# through: TIME=48:00:00 ./sweep_lean.sh works.
set -euo pipefail

REPO=/storage/home/gaetan/projects/trl

# beta 1e-8 rather than 0: the penalty it puts on the rewards is numerically nothing, but a
# non-zero beta is what keeps the reference model alive and the `kl` metric logged, and that metric
# is the distance-from-the-base-model axis every run in the sweep has to be read against.
BASE="--trainer opsft --use-vllm --vllm-gpu-mem 0.2 --beta ${BETA:-1e-8}"

CONFIGS=(
    # --- A: learning rate x whitening strength (8 runs) ---------------------------------------
    # weak = damping 1e-2 with 1 CG step, strong = damping 1e-4 with 5. Whitening conditions the
    # update, so the best learning rate should sit higher in the strong column than in the weak
    # one; that interaction is what the block is for, and it is why lr is not swept on its own.
    "--lr 1e-6 --score-norm-probes 8 --score-projection-steps 1 --score-norm-damping 1e-2"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 1 --score-norm-damping 1e-2"
    "--lr 1e-5 --score-norm-probes 8 --score-projection-steps 1 --score-norm-damping 1e-2"
    "--lr 3e-5 --score-norm-probes 8 --score-projection-steps 1 --score-norm-damping 1e-2"
    "--lr 1e-6 --score-norm-probes 8 --score-projection-steps 5 --score-norm-damping 1e-4"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 5 --score-norm-damping 1e-4"
    "--lr 1e-5 --score-norm-probes 8 --score-projection-steps 5 --score-norm-damping 1e-4"
    "--lr 3e-5 --score-norm-probes 8 --score-projection-steps 5 --score-norm-damping 1e-4"

    # --- B: damping ladder at a fixed, known-stable lr (5 runs) -------------------------------
    # 8 CG steps throughout so the step count is never the binding constraint and damping is the
    # only thing varying. Read advantages/cg_residual_ratio to confirm 8 was enough at 1e-6.
    # The 1e-2 cell also prices the "1e-2 saturates immediately" claim against A's 1-step cell.
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 1e-1"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 1e-2"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 1e-3"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 1e-5"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 1e-6"

    # --- C: does any of the machinery pay for itself (4 runs) ---------------------------------
    # The first two are the null hypotheses and the cheapest runs in the sweep: raw inner products,
    # then diagonal normalisation with no CG at all. If the second ties A's 5-step cells, the
    # conjugate gradient solve is dead weight. The last two bracket the probe count around 8.
    "--lr 3e-6 --score-norm-probes 0"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 0"
    "--lr 3e-6 --score-norm-probes 2  --score-projection-steps 5 --score-norm-damping 1e-4"
    "--lr 3e-6 --score-norm-probes 32 --score-projection-steps 5 --score-norm-damping 1e-4"

    # --- D: the noise floor (3 runs) ----------------------------------------------------------
    # Three reruns of A's second config. Without these the whole sweep is uninterpretable: no
    # ranking means anything until the seed-to-seed spread is known to be smaller than it.
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 1 --score-norm-damping 1e-2 --seed 43"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 1 --score-norm-damping 1e-2 --seed 44"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 1 --score-norm-damping 1e-2 --seed 45"
)

# Most recent state slurm holds for each run name. A config whose job is alive or finished is left
# alone, so rerunning this script submits exactly the ones that died -- there is no list of failures
# to maintain by hand, and nothing is duplicated if it is run twice. TIMEOUT counts as finished: at
# 100 epochs the wall clock is what ends a healthy run, not the epoch count. sacct lists ascending
# by job id, so a later attempt overwrites the verdict of an earlier one.
declare -A STATE
while IFS='|' read -r name st; do
    case "$st" in
        PENDING | RUNNING | COMPLETING | SUSPENDED | REQUEUED | COMPLETED | TIMEOUT) STATE[$name]=$st ;;
        *) STATE[$name]= ;;
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
