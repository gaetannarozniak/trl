#!/bin/bash
# Second OPSFT sweep, centred on lr3e-6 / hutch8 / proj8 / damp0.1 -- the best config of the first
# one -- and adding the SFT-gradient EMA as an axis. Same contract as sweep_lean.sh: one 8-GPU job
# per config, names derived by sft_lean.py, and a config whose job is alive or finished is skipped,
# so rerunning submits exactly what died.
#
#   DRY=1 ./sweep_lean2.sh          # show what would go
#   TIME=48:00:00 ./sweep_lean2.sh  # 16 arms at 96h is 12k GPU-hours; 48h still clears the transient
#   STAGGER=900 ./sweep_lean2.sh    # widen the start spread while the NFS is degraded
#
# Every name carries the 1_sweep prefix, so the four seed43/seed44 centre jobs already running under
# unprefixed names do not match and would be launched again: scancel them before the first run.
set -euo pipefail

REPO=/storage/home/gaetan/projects/trl
BASE="--trainer opsft --use-vllm --vllm-gpu-mem 0.2 --beta ${BETA:-1e-8} --name-prefix ${PREFIX:-1_sweep}"
CENTRE="--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 1e-1"

CONFIGS=(
    # --- A: damping, probed upward (4) ---------------------------------------------------------
    # 1e-1 won the first sweep while sitting at the top of a grid that only went down from there,
    # so the interesting direction is heavier damping. 3 is close to no projection at all: the
    # filter is nearly flat, which makes it the anchor for how much the solve is buying.
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 3e-2"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 3e-1"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 1"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 3"

    # --- B: conjugate-gradient steps at the winning damping (3) --------------------------------
    # Damping caps how far the solve gets, and 1e-1 caps it early, so most of the eight steps are
    # probably paid for nothing. With the centre this gives the curve 2/4/8/16: if 2 matches 8,
    # each step drops six backward+jvp pairs and the same wall clock buys far more of them.
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 2  --score-norm-damping 1e-1"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 4  --score-norm-damping 1e-1"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 16 --score-norm-damping 1e-1"

    # --- C: EMA on the SFT target gradient (3) -------------------------------------------------
    # With the running seed43/44 pair at 0.9 this gives 0.5/0.9/0.95/0.99. The buffer is bf16, so
    # 0.99 is the last coefficient whose update is larger than one ulp; it is in to find the edge.
    "$CENTRE --sft-gradient-ema 0.5"
    "$CENTRE --sft-gradient-ema 0.95"
    "$CENTRE --sft-gradient-ema 0.99"

    # --- D: EMA against damping (2) ------------------------------------------------------------
    # Both absorb noise in the right-hand side, so they are partly substitutable: an averaged
    # target may tolerate the lighter damping that was too noisy without it.
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 3e-2 --sft-gradient-ema 0.9"
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 3e-1 --sft-gradient-ema 0.9"

    # --- E: learning rate at this centre (2) ---------------------------------------------------
    # The first sweep swept lr at proj1. proj8 with damping 1e-1 is a differently scaled update,
    # and the EMA shrinks it further through the overshoot clamp, so the curve is worth redoing.
    "--lr 1e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 1e-1"
    "--lr 1e-5 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 1e-1"

    # --- G: best CG iterate, paired with the damp3e-2 cell of block A (1) ----------------------
    "--lr 3e-6 --score-norm-probes 8 --score-projection-steps 8 --score-norm-damping 3e-2 --score-projection-best-iterate"

    # --- F: third seed on both arms of the centre (2) ------------------------------------------
    # Seeds 43 and 44 are already running on each arm. A third puts a spread on the comparison
    # every single-seed cell above has to be read against.
    "$CENTRE --seed 45"
    "$CENTRE --seed 45 --sft-gradient-ema 0.9"

    # --- already running: listed so a rerun revives them if they die ---------------------------
    "$CENTRE --seed 43"
    "$CENTRE --seed 43 --sft-gradient-ema 0.9"
    "$CENTRE --seed 44"
    "$CENTRE --seed 44 --sft-gradient-ema 0.9"
)

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
