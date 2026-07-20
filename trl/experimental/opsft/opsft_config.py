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
