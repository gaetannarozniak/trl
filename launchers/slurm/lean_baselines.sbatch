#!/bin/bash
# ---------------------------------------------------------------------------
# Single node, 4x GB300, ZeRO-2, for launchers/baselines/train.py
#
#   sbatch launchers/slurm/lean_baselines.sbatch
#   sbatch --export=ALL,CONFIG=sft_kl launchers/slurm/lean_baselines.sbatch
#   sbatch --export=ALL,CONFIG=sft_kl,EXPE=kl_lr5e6,EXTRA="--beta 0.2 --learning_rate 5e-6" \
#          launchers/slurm/lean_baselines.sbatch
#
# Overridable via --export=ALL,VAR=value :
#   CONFIG   sft | sft_kl               (default: sft)
#   EXPE     run name + ckpt dir        (default: lean_<config>_<jobid>)
#   EXTRA    any extra CLI overrides    (default: unset)
#   SAVE     1 to keep the final model  (default: off)
#   SMOKE    1 to run the ~3 min end-to-end smoke test on 1 GPU
#
# Every field of LeanSFTConfig is settable in the YAML and overridable via EXTRA, so this
# launcher never needs editing to change a hyperparameter.
# ---------------------------------------------------------------------------
#SBATCH --job-name=lean_baselines
#SBATCH --partition=g3
#SBATCH --account=scientific-reasoning
#SBATCH --qos=g3_scientific-reasoning_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # ONE task: accelerate forks the ranks itself.
#SBATCH --gpus-per-node=4            # whole node; g3 nodes are gpu:gb300:4
#SBATCH --cpus-per-task=144          # whole node: 2 Grace CPUs x 72 cores, ThreadsPerCore=1
#SBATCH --mem=0                      # all node memory
#SBATCH --time=00-24:00:00
#SBATCH --output=/home/ilabiad/projects/trl/slurm_logs/%x-%j/%x-%j.out
#SBATCH --error=/home/ilabiad/projects/trl/slurm_logs/%x-%j/%x-%j.err

set -euo pipefail

REPO=/home/ilabiad/projects/trl

: "${CONDA_EXE:?run sbatch from a shell where conda is initialised}"
: "${CONDA_PREFIX:?conda activate your env before sbatch}"

CONDA_ENV="${CONDA_PREFIX}"                       # save it: conda.sh/base activation clobbers it
source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO}"

# trl is installed editable, so the fork is authoritative wherever it is imported from. This is a
# safety net for environments where that is not true: `sys.path[0]` under `accelerate launch` is
# the *script's* directory, not the cwd, so without it `import trl` can hit site-packages.
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

# Lean toolchain: the verifier shells out to `lake` and runs the prebuilt REPL from
# ~/lean/workspace-v4.15 (Mathlib v4.15.0, matching what NuminaMath-LEAN was verified against).
export PATH="${HOME}/.elan/bin:${PATH}"

# 4 ranks x 8 threads = 32 of 144 CPUs. The Lean pool takes 30 per rank (120); the rest is
# headroom for the dataloader.
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# FlashInfer cannot fetch prebuilt kernels from an offline node and JIT-compiling them needs a
# coherent CUDA toolkit that the pip-installed nvidia-* wheels do not provide (nvcc 13.3 against
# 13.0 headers). This disables its sampler; the attention backend is set to FLASH_ATTN in the YAML.
export VLLM_USE_FLASHINFER_SAMPLER=0

export HF_HOME=/home/ilabiad/.cache/huggingface
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# FAIR's wandb instance, not public wandb.ai (which is firewalled off).
# Credentials come from ~/.netrc, which is on the shared home filesystem.
export WANDB_BASE_URL=https://fairwandb.org
export WANDB_DIR=/checkpoint/fort/ilabiad/wandb
mkdir -p "${WANDB_DIR}"

CONFIG="${CONFIG:-sft}"
EXPE="${EXPE:-lean_${CONFIG}_${SLURM_JOB_ID}}"
CONFIG_FILE="${REPO}/launchers/baselines/configs/${CONFIG}.yaml"
[[ -f "${CONFIG_FILE}" ]] || { echo "no such config: ${CONFIG_FILE}" >&2; exit 1; }

# Default 29500 collides if two jobs land on the same node; derive from the job id.
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

NUM_PROCESSES=4
ACCELERATE_ARGS=(--config_file "${REPO}/examples/accelerate_configs/deepspeed_zero2.yaml")
EXTRA_ARGS=(--run_name "${EXPE}" --output_dir "/checkpoint/fort/ilabiad/opsft2/${EXPE}")

SRUN_ARGS=(--cpu-bind=none)

if [[ "${SMOKE:-0}" == "1" ]]; then
  # Exercise the whole pipeline fast on ONE GPU. The GPU count matters as much as the process
  # count: with a single process but several visible GPUs, HF Trainer sets n_gpu = device_count
  # and wraps the policy in nn.DataParallel, scattering replicas across devices while the
  # reference model stays on cuda:0 -- so the KL then compares tensors on different devices.
  # Exporting CUDA_VISIBLE_DEVICES here does not work: srun recomputes it from the step's GRES
  # allocation, so the allocation itself has to be narrowed.
  # (Neither this nor device_map sharding affects the real run: 4 processes means
  # distributed_type=MULTI_GPU, n_gpu=1, and no DataParallel.)
  SRUN_ARGS+=(--gres=gpu:1)
  NUM_PROCESSES=1
  ACCELERATE_ARGS=()
  EXTRA_ARGS+=(--smoke true)
fi

[[ "${SAVE:-0}" == "1" ]] && EXTRA_ARGS+=(--save_final_model true)

# shellcheck disable=SC2206
[[ -n "${EXTRA:-}" ]] && EXTRA_ARGS+=(${EXTRA})

echo "job ${SLURM_JOB_ID} on $(hostname) | config=${CONFIG} expe=${EXPE} port=${MASTER_PORT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srun "${SRUN_ARGS[@]}" accelerate launch \
  "${ACCELERATE_ARGS[@]}" \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "${MASTER_PORT}" \
  launchers/baselines/train.py \
    --config "${CONFIG_FILE}" \
    "${EXTRA_ARGS[@]}"
