# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field

from ...trainer.grpo_config import GRPOConfig


@dataclass
class OPSFTConfig(GRPOConfig):
    r"""
    Configuration class for the [`OPSFTTrainer`].

    This subclasses [`GRPOConfig`], so every GRPO training and generation option (including all vLLM options) applies
    unchanged. It adds only the parameters specific to OPSFT.

    Parameters:
        compute_eval_kl (`bool`, *optional*, defaults to `False`):
            Whether to keep a frozen copy of the base model $\theta_0$ in memory and, at every evaluation step, log
            the k3 KL divergence (`eval_kl`) between the trained policy and the base model over the model's
            self-generated completions. Independent of `beta`: the base model is held and the KL is logged even when
            `beta == 0.0` (no GRPO reference model). Adds one extra model to memory and one extra forward per eval
            step; it is a pure diagnostic and has no effect on training.
        whitening_probes (`int`, *optional*, defaults to `0`):
            If `> 0`, diagonally whiten the update: rescale each token coefficient by an estimate of its score norm,
            $c_t \to c_t / (\|\nabla \log f_t\|^2 + \lambda)$ (the steepest in-subspace step when the per-token score
            vectors are orthogonal). The norms are estimated with this many Hutchinson random-tangent JVPs (each an
            extra forward-mode pass). `0` disables it (plain OPSFT $c$).
        whitening_damping (`float`, *optional*, defaults to `0.01`):
            Relative damping $\lambda$ for the whitening, as a fraction of the mean estimated norm:
            $\lambda = \texttt{whitening\_damping}\cdot\operatorname{mean}_t\|\nabla\log f_t\|^2$. Keeps tiny/noisy
            estimates from blowing up the coefficient while barely biasing typical tokens; only used when
            `whitening_probes > 0`. Set `0.0` for the pure (numerically unstable) version.
        sft_baseline (`bool`, *optional*, defaults to `False`):
            If `True`, train with the plain SFT gradient $u = \nabla\log\pi(y\mid x)$ (minimize the reference NLL)
            instead of the OPSFT step $J^\top c$ — a controlled baseline sharing the same rollout/eval pipeline. The
            self-generated rollout still runs upstream but is unused by the loss.
    """

    compute_eval_kl: bool = field(
        default=False,
        metadata={
            "help": "Whether to keep a frozen copy of the base model in memory and, at every evaluation step, log "
            "the k3 KL divergence (`eval_kl`) between the trained policy and the base model over the model's "
            "self-generated completions. Independent of `beta`: logged even when `beta == 0.0`. Adds one extra model "
            "to memory and one extra forward per eval step; it is a pure diagnostic and has no effect on training."
        },
    )
    whitening_probes: int = field(
        default=0,
        metadata={
            "help": "If > 0, diagonally whiten the update: rescale each token coefficient by an estimate of its score "
            "norm, c_t -> c_t / (||grad log f_t||^2 + lambda) (steepest in-subspace step when the per-token score "
            "vectors are orthogonal). The norms are estimated with this many Hutchinson random-tangent JVPs (each an "
            "extra forward-mode pass). 0 disables it (plain OPSFT c)."
        },
    )
    whitening_damping: float = field(
        default=0.01,
        metadata={
            "help": "Relative damping lambda for the whitening, as a fraction of the mean estimated norm "
            "(lambda = whitening_damping * mean_t ||grad log f_t||^2). Tames tiny/noisy estimates without biasing "
            "typical tokens; only used when whitening_probes > 0. Set 0.0 for the pure (unstable) version."
        },
    )
    sft_baseline: bool = field(
        default=False,
        metadata={
            "help": "If True, train with the plain SFT gradient u = grad log pi(y|x) (minimize the reference NLL) "
            "instead of the OPSFT step J^T c. A controlled baseline sharing the same rollout/eval pipeline; the "
            "generated rollout still runs upstream but is unused by the loss."
        },
    )
