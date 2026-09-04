# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for the Lean SFT baselines.

`LeanSFTConfig` extends [`SFTConfig`] with everything the Lean baselines need: the reference
model / KL knobs, the dataset registry key, the vLLM evaluation engine settings, the Lean
verification pool settings, and the independent evaluation cadences.

Every field is settable from YAML (via `TrlParser`) and overridable from the command line, so
the sbatch never has to hardcode anything.
"""

from dataclasses import dataclass, field

from trl import SFTConfig


@dataclass
class LeanSFTConfig(SFTConfig):
    """Configuration for [`LeanSFTTrainer`] and the Lean evaluation callbacks.

    Args:
        > Reference model and KL

        ref_model_path (`str`, *optional*):
            Path of the frozen reference model used for the KL term. Defaults to the policy's own
            initial checkpoint, which makes the KL measure drift from initialisation. The reference
            is **always** loaded: `beta` only decides whether the KL enters the loss, never whether
            it is computed, so `train/kl` and `eval/kl` are available for every baseline.
        beta (`float`, *optional*, defaults to `0.0`):
            Coefficient of the KL term in the loss. `0.0` reproduces plain SFT while still logging
            the KL as a metric.
        chunk_size (`int`, *optional*, defaults to `8192`):
            Vocabulary chunk size for the chunked log-prob kernel. Log-probs are computed with an
            online logsumexp so the `[N, vocab]` logits are never materialised; lower this if the
            per-chunk buffers are too large.

        > Dataset

        dataset_key (`str`, *optional*, defaults to `"numina_lean"`):
            Key into the dataset registry in `datasets.py`.
        eval_fraction (`float`, *optional*, defaults to `0.05`):
            Fraction of the dataset held out for evaluation.
        dataset_seed (`int`, *optional*, defaults to `0`):
            Seed for the train/test split. Keep fixed across baselines so they are comparable.

        > vLLM evaluation engine

        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.3`):
            Fraction of GPU memory reserved for the colocated vLLM engine.
        vllm_max_model_length (`int`, *optional*, defaults to `9216`):
            Engine context length. Must cover the longest prompt plus `lean_max_new_tokens`; it is
            fixed at construction, so per-call generation lengths can only go below it.
        vllm_max_num_seqs (`int`, *optional*, defaults to `24`):
            Maximum concurrent sequences. Olmo-3 uses MHA (no GQA) with only 8 of 32 layers doing
            full attention, which still costs roughly 2.8 GB of KV cache per 9216-token sequence,
            so this is a real memory knob rather than a formality.
        vllm_max_num_batched_tokens (`int`, *optional*, defaults to `8192`):
            Chunked-prefill chunk size.
        vllm_attention_backend (`str`, *optional*, defaults to `"FLASH_ATTN"`):
            Attention backend. FlashInfer (the vLLM default) JIT-compiles its kernels, which needs
            a coherent CUDA toolkit and network access to fetch prebuilt kernels; neither is
            available on the compute nodes. `"FLASH_ATTN"` uses kernels prebuilt into the wheel and
            measured ~17% faster here. Pair it with `VLLM_USE_FLASHINFER_SAMPLER=0` (set in the
            YAML `env:` block), which disables the other FlashInfer entry point.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Offload the engine between generations. Off by default: the node has enough memory, and
            waking costs host-device transfers.

        > Lean verification

        lean_workspace (`str`, *optional*):
            Path to the prebuilt Mathlib workspace. Must match the version the dataset was verified
            against; v4.15 compiles 100/100 human gold proofs, v4.25.2 only 64/100.
        lean_repl_path (`str`, *optional*):
            Path to the Lean REPL. Defaults to `<lean_workspace>/.lake/packages/REPL`.
        lake_path (`str`, *optional*, defaults to `"/home/ilabiad/.elan/bin/lake"`):
            Path to the `lake` binary.
        lean_num_workers (`int`, *optional*, defaults to `30`):
            Lean workers per rank. The node has 144 CPUs for 4 ranks, so 30 leaves 6 per rank for
            the training process and dataloader.
        lean_timeout (`int`, *optional*, defaults to `60`):
            Per-proof verification timeout in seconds, hard-enforced by killing the REPL. Correct
            proofs verify in ~1.6s median / 35.8s max once workers are warm, so 60s costs nothing
            and lowering it mainly risks discarding genuine successes.
        lean_eval_max_seconds (`int`, *optional*, defaults to `600`):
            Wall-clock budget for the verification phase, per rank. `0` disables it.

            This is a circuit breaker, not a tuning knob. Every rank blocks in an `all_reduce`
            after verifying, so one slow shard stalls the others -- and NCCL kills any collective
            exceeding 1800s, which has already destroyed one run. Capping below that turns a lost
            job into a degraded measurement.

            Prefer bounding `lean_eval_n` over relying on this. The proofs left unrun are the slow
            ones rather than a random sample, so a budget that actually bites makes the compile
            rate depend on machine load and stop being comparable across steps. When it fires,
            `eval/lean_coverage` drops below 1 and a warning is printed.
        lean_verify_truncated (`bool`, *optional*, defaults to `False`):
            Whether to send truncated generations to Lean. They are cut off mid-proof and cannot
            compile, so verifying them only occupies workers; they are counted as failures in the
            `truncated` bucket regardless, keeping `compile_rate = ok / total` honest.
        lean_eval_n (`int`, *optional*):
            Number of held-out theorems to verify. `None` (the default) uses the whole eval split.
        lean_max_new_tokens (`int`, *optional*, defaults to `8192`):
            Generation budget for Lean evaluation. Gold completions are median 656 / p90 2808 /
            p99 8705 tokens, so 8192 covers ~99%; the previous 1024 truncated 36.6% of them and
            capped the achievable compile rate at ~0.63.

        > Checkpointing

        save_final_model (`bool`, *optional*, defaults to `False`):
            Whether to write the final model at the end of training. Off by default because these
            are 7B checkpoints and most baseline runs are read through their metrics, not their
            weights. When enabled the absolute destination is printed.

        > Evaluation cadence

        lean_eval_steps (`int`, *optional*, defaults to `140`):
            Run the Lean evaluation every N steps. Kept well above `eval_steps` because
            verification is CPU-bound and synchronous.
        vibecheck_steps (`int`, *optional*, defaults to `140`):
            Print sample proofs every N steps. The previous implementation ran on every `on_log`,
            which with `logging_steps=1` consumed ~52% of total runtime.
        vibecheck_n (`int`, *optional*, defaults to `2`):
            Number of held-out theorems to print.
        vibecheck_max_new_tokens (`int`, *optional*, defaults to `1024`):
            Generation budget for the qualitative check.
    """

    # Reference model and KL
    ref_model_path: str | None = field(
        default=None,
        metadata={"help": "Frozen reference model for the KL term. Defaults to the policy's initial checkpoint."},
    )
    beta: float = field(
        default=0.0,
        metadata={"help": "KL coefficient in the loss. 0.0 is plain SFT; the KL is logged either way."},
    )
    chunk_size: int = field(
        default=8192,
        metadata={"help": "Vocabulary chunk size for the chunked log-prob kernel."},
    )

    # Dataset
    dataset_key: str = field(
        default="numina_lean",
        metadata={"help": "Key into the dataset registry in datasets.py."},
    )
    eval_fraction: float = field(
        default=0.05,
        metadata={"help": "Fraction of the dataset held out for evaluation."},
    )
    dataset_seed: int = field(
        default=0,
        metadata={"help": "Seed for the train/test split. Keep fixed across baselines."},
    )

    # vLLM evaluation engine
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={"help": "Fraction of GPU memory reserved for the colocated vLLM engine."},
    )
    vllm_max_model_length: int = field(
        default=9216,
        metadata={"help": "Engine context length; must cover longest prompt + lean_max_new_tokens."},
    )
    vllm_max_num_seqs: int = field(
        default=24,
        metadata={"help": "Maximum concurrent sequences in the vLLM engine."},
    )
    vllm_max_num_batched_tokens: int = field(
        default=8192,
        metadata={"help": "Chunked-prefill chunk size for the vLLM engine."},
    )
    vllm_attention_backend: str | None = field(
        default="FLASH_ATTN",
        metadata={"help": "vLLM attention backend. FLASH_ATTN avoids FlashInfer's JIT compilation."},
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={"help": "Offload the vLLM engine between generations."},
    )

    # Lean verification
    lean_workspace: str | None = field(
        default=None,
        metadata={"help": "Path to the prebuilt Mathlib workspace. Required when Lean eval is enabled."},
    )
    lean_repl_path: str | None = field(
        default=None,
        metadata={"help": "Path to the Lean REPL. Defaults to <lean_workspace>/.lake/packages/REPL."},
    )
    lake_path: str = field(
        default="/home/ilabiad/.elan/bin/lake",
        metadata={"help": "Path to the lake binary."},
    )
    lean_num_workers: int = field(
        default=30,
        metadata={"help": "Lean verification workers per rank."},
    )
    lean_timeout: int = field(
        default=60,
        metadata={"help": "Per-proof verification timeout in seconds."},
    )
    lean_eval_max_seconds: int = field(
        default=600,
        metadata={
            "help": "Wall-clock budget for the verification phase, per rank; 0 disables. A safety "
            "net against the NCCL collective watchdog, not a tuning knob -- bound `lean_eval_n` "
            "instead. Firing it makes the compile rate load-dependent; watch `eval/lean_coverage`."
        },
    )
    lean_verify_truncated: bool = field(
        default=False,
        metadata={
            "help": "Send generations that hit the token budget to Lean anyway. They cannot "
            "compile (they are cut off mid-proof) and still count as failures either way, so the "
            "default skips them and spends the workers on proofs that could actually succeed."
        },
    )
    lean_eval_n: int | None = field(
        default=None,
        metadata={"help": "Theorems to verify per Lean eval. None uses the whole eval split."},
    )
    lean_max_new_tokens: int = field(
        default=8192,
        metadata={"help": "Generation budget for Lean evaluation."},
    )

    # Checkpointing
    save_final_model: bool = field(
        default=False,
        metadata={"help": "Write the final model at the end of training."},
    )

    # Evaluation cadence
    lean_eval_steps: int = field(
        default=140,
        metadata={"help": "Run the Lean evaluation every N steps."},
    )
    vibecheck_steps: int = field(
        default=140,
        metadata={"help": "Print sample proofs every N steps."},
    )
    vibecheck_n: int = field(
        default=2,
        metadata={"help": "Number of held-out theorems to print."},
    )
    vibecheck_max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Generation budget for the qualitative check."},
    )

    # Smoke test
    smoke: bool = field(
        default=False,
        metadata={"help": "Tiny end-to-end run: a few steps, a handful of theorems, short generations."},
    )

    def __post_init__(self):
        # `LeanSFTTrainer` computes the loss itself from chunked log-probs, so TRL must not install
        # its own chunked cross-entropy patch on `model.forward` -- the two patches would collide.
        self.loss_type = "nll"
        super().__post_init__()

        if self.smoke:
            self.max_steps = 20
            self.eval_steps = 10
            self.lean_eval_steps = 10
            self.vibecheck_steps = 10
            self.lean_eval_n = 8
            self.lean_num_workers = 4
            self.lean_max_new_tokens = 256
            self.vibecheck_max_new_tokens = 128
            self.vllm_max_model_length = 1024
            self.save_strategy = "no"

        if self.beta < 0:
            raise ValueError(f"`beta` must be non-negative, got {self.beta}.")
        if self.lean_repl_path is None and self.lean_workspace is not None:
            self.lean_repl_path = f"{self.lean_workspace}/.lake/packages/REPL"
