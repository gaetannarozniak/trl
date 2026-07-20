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

import torch

from ...trainer.grpo_trainer import GRPOTrainer as _GRPOTrainer


def _no_reward(completions, **kwargs):
    # OPSFT trains on self-generations with no reward. This constant placeholder satisfies GRPO's
    # "at least one reward source" guard; a constant reward makes every group advantage collapse to
    # zero, and `_compute_loss` ignores advantages entirely anyway.
    return [0.0] * len(completions)


class OPSFTTrainer(_GRPOTrainer):
    r"""
    Trainer for **OPSFT** (On-Policy SFT): the model samples completions from its own policy (optionally with vLLM)
    and is trained on those self-generated tokens with the Fisher / Gauss-Newton–vector product

    $$
    S = \sum_t \left(\nabla_\theta \log f_t \cdot u\right) \nabla_\theta \log f_t = J^\top J u = F u,
    $$

    where $f_t = \pi(y_t \mid x, y_{<t})$ is the policy's probability of self-generated completion token $t$, $J$ is
    the Jacobian of the stacked per-token log-probs (row $t$ is $\nabla_\theta \log f_t$), $F = \sum_t \nabla\log f_t
    \nabla\log f_t^\top$ is the empirical Fisher (outer-product) matrix, and $u$ is a constant vector in parameter
    space (**hardcoded to all-ones** for now). There is no reward model and no advantage weighting: the objective is
    purely $S = F u$ over every generated token.

    All on-policy generation machinery (transformers `generate`, vLLM colocate/server modes, weight syncing, rollout
    buffering) is inherited unchanged from [`GRPOTrainer`], so every generation and vLLM option in [`GRPOConfig`]
    applies. Only the loss is replaced: instead of GRPO's advantage-weighted policy gradient, `_compute_loss` builds
    a surrogate scalar

    $$
    L = \sum_t \operatorname{sg}(c_t) \log f_t, \qquad c_t = \nabla_\theta \log f_t \cdot u = (J u)_t,
    $$

    whose gradient is exactly $\nabla_\theta L = J^\top c = F u = S$ ($\operatorname{sg}$ is stop-gradient). The
    directional derivatives $c = J u$ are obtained with a single training forward pass and a double reverse-mode pass
    (a JVP built from two VJPs), so the cost is ~1 forward + 3 backward-equivalents and the memory is independent of
    both the number of tokens and $\dim\theta$ — $J$ is never materialized.

    <Tip warning={true}>

    The double backward (`create_graph=True`) in the *training* forward constrains the setup (generation is a
    separate no-grad path and is unaffected, so vLLM composes fine):

    - Load the model with `attn_implementation="eager"` (flash/SDPA attention kernels do not support double backward).
    - Set `gradient_checkpointing=False` in the [`GRPOConfig`].
    - Prefer `bf16=True` over `fp16=True` (the fp16 gradient scaler interferes with the inner gradients).
    - Do not use DeepSpeed ZeRO (gradients with respect to partitioned parameters are unavailable).

    </Tip>

    Example:

    ```python
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig
    from trl.experimental.opsft import OPSFTTrainer

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", attn_implementation="eager")
    dataset = load_dataset("trl-lib/tldr", split="train")  # prompt-only dataset
    trainer = OPSFTTrainer(
        model=model,
        args=GRPOConfig(
            output_dir="Qwen2.5-0.5B-OPSFT",
            gradient_checkpointing=False,
            bf16=True,
            use_vllm=True,  # optional; colocate by default
        ),
        train_dataset=dataset,
    )
    trainer.train()
    ```
    """

    _tag_names = ["trl", "opsft"]
    _name = "OPSFT"

    def __init__(self, model, args=None, reward_funcs=None, **kwargs):
        # No reward by default: inject a constant placeholder so GRPO's rollout stack runs (mirrors
        # the pattern in trl/experimental/minillm). Users may still pass reward_funcs, but they are
        # unused by the OPSFT loss.
        if reward_funcs is None:
            reward_funcs = _no_reward
        super().__init__(model=model, reward_funcs=reward_funcs, args=args, **kwargs)

    def _compute_loss(self, model, inputs):
        # 1. Per-token log f_t = log pi(y_t | x, y_<t) of the self-generated completion. Reuse GRPO's helper (same
        # call as GRPOTrainer._compute_loss), but we need neither entropy nor the MoE aux loss here.
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=False,
            compute_aux_loss=False,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
            num_tiles=inputs.get("num_tiles"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            image_position_ids=inputs.get("image_position_ids"),
        )
        logps = per_token_logps[mask.bool()]  # (T,), keep only the (non-padded) generated tokens

        # 2. u: a constant vector in parameter space, hardcoded to all-ones for now.
        params = [p for p in model.parameters() if p.requires_grad]
        u = [torch.ones_like(p) for p in params]

        # 3. c = J u: the directional derivative of each log f_t along u, i.e. c_t = grad log f_t . u. We get all T of
        # them with one JVP, built from two reverse-mode passes ("double backward") to avoid forward-mode AD:
        #   w = grad(v . logps) = J^T v          (linear in the dummy vector v)
        #   c = grad_v (w . u) = grad_v (v^T J u) = J u
        v = torch.zeros_like(logps, requires_grad=True)
        w = torch.autograd.grad((v * logps).sum(), params, create_graph=True, allow_unused=True)
        inner = sum((w_p * u_p).sum() for w_p, u_p in zip(w, u) if w_p is not None)
        c = torch.autograd.grad(inner, v, retain_graph=True)[0].detach()  # (T,) = J u

        # 4. Surrogate loss whose gradient is S = J^T c = F u. `c` is detached, so backprop only flows through logps;
        # `retain_graph=True` above keeps the forward graph alive for this backward. Normalized by the number of
        # generated tokens (mean) and by the gradient-accumulation count, matching GRPO (HF auto-scaling is off here).
        loss = - (c * logps).sum() / mask.sum().clamp(min=1.0)
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        loss = loss / normalizer

        # The magnitude of c (= sum of grad-log-prob components, since u = ones) is the key quantity to watch: it sets
        # the effective step size and can be large, so log it.
        self._metrics[mode]["c_abs_mean"].append(self.accelerator.gather(c.abs().mean()).nanmean().item())
        # Weight-movement probes. The global parameter *mean* is near-invariant (updates cancel out when averaged over
        # all params), so it is a blind drift signal; the L2 norm responds to any coordinate change. Both are kept.
        # No gather: params are replicated across ranks (DDP), so every rank holds identical values.
        with torch.no_grad():
            param_norm = torch.stack([p.norm() for p in model.parameters()]).norm()
            params_mean = sum(p.sum() for p in model.parameters()) / sum(p.numel() for p in model.parameters())
        self._metrics[mode]["param_norm"].append(param_norm.item())
        self._metrics[mode]["parameters_mean"].append(params_mean.item())
        return loss
