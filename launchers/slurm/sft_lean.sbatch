#!/bin/bash
# ---------------------------------------------------------------------------
# Single node, 4x GB300, DDP, for trl/scripts/sft_lean.py
#
#   sbatch launchers/slurm/sft_lean.sbatch
#   sbatch --export=ALL,TRAINER=opsft,EXPE=lean_opsft_lr1e6,LR=1e-6 \
#          launchers/slurm/sft_lean.sbatch
#
# Overridable via --export=ALL,VAR=value :
#   TRAINER  sft | opsft            (default: sft)
#   EXPE     run name + ckpt dir    (default: lean_<trainer>_<jobid>)
#   LR       learning rate          (default: unset -> each config's own default)
#
# One-time setup before the first submit:
#   conda activate trl
#   python -c "from datasets import load_dataset; load_dataset('AI-MO/NuminaMath-LEAN')"
# The Olmo-3-7B weights are already in ~/.cache/huggingface; the dataset is not.
# ---------------------------------------------------------------------------
#SBATCH --job-name=sft_lean
#SBATCH --partition=g3
#SBATCH --account=scientific-reasoning
#SBATCH --qos=g3_scientific-reasoning_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # ONE task: accelerate forks the 4 ranks itself.
#SBATCH --gpus-per-node=4            # whole node; g3 nodes are gpu:gb300:4
#SBATCH --cpus-per-task=144          # whole node: 2 Grace CPUs x 72 cores, ThreadsPerCore=1
#SBATCH --mem=0                      # all node memory
#SBATCH --time=00-24:00:00
#SBATCH --output=/home/ilabiad/projects/trl/slurm_logs/%x-%j/%x-%j.out
#SBATCH --error=/home/ilabiad/projects/trl/slurm_logs/%x-%j/%x-%j.err

set -euo pipefail

REPO=/home/ilabiad/projects/trl

# source /home/ilabiad/miniconda3/etc/profile.d/conda.sh
# conda activate trl
# cd "${REPO}"

: "${CONDA_EXE:?run sbatch from a shell where conda is initialised}"
: "${CONDA_PREFIX:?conda activate your env before sbatch}"

CONDA_ENV="${CONDA_PREFIX}"                       # save it: conda.sh/base activation clobbers it
source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO}"


# site-packages holds a non-editable trl 1.12.0 that has no OPSFTTrainer, and
# sys.path[0] is the *script's* dir (trl/scripts), not the cwd -- so without this
# `import trl` resolves to site-packages and sft_lean.py dies on its line 42.
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

# Lean toolchain: LeanEval shells out to `lake` and runs the prebuilt REPL from
# ~/lean/workspace-v4.15 (Mathlib v4.15.0, matching what NuminaMath-LEAN was verified against).
export PATH="${HOME}/.elan/bin:${PATH}"

# 4 ranks x 8 threads = 32, comfortably inside the 128 allocated CPUs.
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

export HF_HOME=/home/ilabiad/.cache/huggingface
# Opt-in: only safe once every dataset/model is in the cache (see header).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# FAIR's wandb instance, not public wandb.ai (which is firewalled off).
# Credentials come from ~/.netrc, which is on the shared home filesystem.
export WANDB_BASE_URL=https://fairwandb.org
export WANDB_DIR=/checkpoint/fort/ilabiad/wandb
mkdir -p "${WANDB_DIR}"

TRAINER="${TRAINER:-sft}"
EXPE="${EXPE:-lean_${TRAINER}_${SLURM_JOB_ID}}"

# Default 29500 collides if two jobs land on the same node; derive from the job id.
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

echo "job ${SLURM_JOB_ID} on $(hostname) | trainer=${TRAINER} expe=${EXPE} port=${MASTER_PORT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

EXTRA_ARGS=()
if [[ -n "${LR:-}" ]]; then
  EXTRA_ARGS+=(--learning-rate "${LR}")
fi

srun --cpu-bind=none accelerate launch \
  --num_machines 1 \
  --num_processes 4 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "${MASTER_PORT}" \
  trl/scripts/sft_lean.py \
    --trainer "${TRAINER}" \
    --expe-name "${EXPE}" \
    "${EXTRA_ARGS[@]}"
