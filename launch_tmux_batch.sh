#!/bin/bash
# Batch launcher (analog of verl-veomni's launch_tmux_dir.sh): fires off several TRL runs at once,
# one per non-empty, non-comment line of a commands file. Each line is a full training command,
# exactly as you'd pass (quoted internally with single quotes) to launch_tmux.sh.
#
# Usage: ./launch_tmux_batch.sh <commands.txt> [--qos <qos>]
#
#   commands.txt:
#     torchrun --nproc_per_node=8 trl/experimental/fd/scripts/feedback_model.py --config ... --run_name 'exp A' ...
#     torchrun --nproc_per_node=8 trl/experimental/fd/scripts/feedback_model.py --config ... --run_name 'exp B' ...
#     # blank lines and #-comments are skipped
#
# Options after the file are forwarded to every run — use --qos here; do NOT pass --name (each run
# should keep its own name from its --run_name, otherwise the tmux sessions collide).

FILE=${1:?Usage: $0 <commands.txt> [--qos <qos>]}
shift
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "${line//[[:space:]]/}" || "$line" == \#* ]] && continue  # skip blank / comment lines
    echo "==> launching: $line"
    bash "$SCRIPT_DIR/launch_tmux.sh" "$@" "$line"
done < "$FILE"
