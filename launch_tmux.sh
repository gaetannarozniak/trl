#!/bin/bash
# Background launcher for TRL runs: reserves a node via srun (in the background), runs the training
# command inside a tmux window on that node, and tees all output to a log file.
# Attach any time with:   ssh <node>   then   tmux attach -t <session>
#
# Usage: ./launch_tmux.sh [--qos <qos>] [--name <job>] "<full training command>"
#
#   ./launch_tmux.sh "torchrun --nproc_per_node=8 trl/experimental/fd/scripts/feedback_model.py \
#       --config trl/experimental/fd/configs/feedback_model.yaml \
#       --run_name 'feedback_model qwen - incorrect only - NO_ADVICE' \
#       --feedback_only_on_incorrect True"
#
# Quote the WHOLE command as one argument, and use single quotes inside it (e.g. around --run_name)
# for values that contain spaces. The job/session/log name is taken from --run_name unless --name is given.

REPO_DIR="$HOME/projects/trl"
LOG_DIR="/checkpoint/scientific-reasoning/gaetan/logs"
ACCOUNT="scientific-reasoning"
QOS="h200_scientific-reasoning_high"
NAME_ARG=""

# Leading options; everything else is the (single, quoted) training command.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --qos)  QOS="$2"; shift 2 ;;
        --name) NAME_ARG="$2"; shift 2 ;;
        *)      break ;;
    esac
done

CMD=${1:?Usage: $0 [--qos <qos>] [--name <job>] \"<training command>\"}

# Job name: explicit --name, else the --run_name value pulled from the command, else a default.
if [ -n "$NAME_ARG" ]; then
    JOB_NAME="$NAME_ARG"
else
    JOB_NAME=$(printf '%s' "$CMD" | grep -oP -- "--run_name[=[:space:]]+\K(?:'[^']*'|\"[^\"]*\"|\S+)" | head -1)
    JOB_NAME="${JOB_NAME%\"}"; JOB_NAME="${JOB_NAME#\"}"   # strip surrounding double quotes
    JOB_NAME="${JOB_NAME%\'}"; JOB_NAME="${JOB_NAME#\'}"   # strip surrounding single quotes
    [ -z "$JOB_NAME" ] && JOB_NAME="trl_exp"
fi

# tmux session / log file names: alphanumerics only (spaces, dots, dashes, slashes -> underscore).
SAFE_NAME=$(printf '%s' "$JOB_NAME" | tr -c 'A-Za-z0-9' '_' | tr -s '_' | sed 's/^_//; s/_$//')

mkdir -p "$LOG_DIR"
export JOB_NAME SAFE_NAME CMD REPO_DIR LOG_DIR

srun \
    --job-name="$JOB_NAME" \
    --qos="$QOS" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --gpus=8 \
    --mem=1024G \
    --cpus-per-task=128 \
    --time=96:00:00 \
    bash -c '
        SESSION="$SAFE_NAME"
        LOG="$LOG_DIR/${SAFE_NAME}_${SLURM_JOB_ID}.out"
        export TERM=xterm-256color  # headless tmux needs TERM + explicit dimensions
        # One window: activate the venv, run the command, tee to the log, then kill the session so the poll loop exits.
        tmux new-session -d -s "$SESSION" -n "train" -x 220 -y 50 \
            "cd \"$REPO_DIR\" && source .venv/bin/activate && $CMD 2>&1 | tee \"$LOG\"; tmux kill-session -t \"$SESSION\""
        # Keep srun alive until the tmux session ends (i.e. training finishes) so the node is released.
        while tmux has-session -t "$SESSION" 2>/dev/null; do sleep 10; done
    ' &

SRUN_PID=$!
echo "Launched '$JOB_NAME' (srun pid $SRUN_PID, qos=$QOS)"
echo "  session: $SAFE_NAME"
echo "  log:     $LOG_DIR/${SAFE_NAME}_<jobid>.out"
echo "  attach:  ssh <node>   then   tmux attach -t $SAFE_NAME"
