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

from ...data_utils import apply_chat_template, is_conversational
from ...trainer.grpo_trainer import GRPOTrainer as _GRPOTrainer
from ...trainer.utils import pad


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

    where $f_t = \pi(y_t \mid x, \tilde{y}_{<t})$ is the policy's probability of self-generated completion token $t$,
    $J$ is the Jacobian of the stacked per-token log-probs (row $t$ is $\nabla_\theta \log f_t$), $F = \sum_t
    \nabla\log f_t \nabla\log f_t^\top$ is the empirical Fisher (outer-product) matrix, and $u = \nabla_\theta \log
    \pi(y \mid x)$ is the SFT gradient of the **dataset reference** completion $y$ (so the dataset must provide
    prompt–completion pairs $(x, y)$). There is no reward model and no advantage weighting: the objective is the
    on-policy-Fisher-preconditioned SFT step $S = F\,\nabla_\theta \log \pi(y \mid x)$ over every generated token.

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
    dataset = load_dataset("trl-lib/tldr", split="train")  # prompt-completion dataset (x, y)
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
        # Snapshot of the initial parameters θ₀ (captured lazily on the first `_compute_loss` call) to log drift.
        self._theta0 = None

    def _generate_and_score_completions(self, inputs):
        # Run GRPO's usual on-policy rollout (self-generated ỹ defines the Fisher J), then additionally tokenize the
        # dataset reference (x, y) so `_compute_loss` can build u = ∇ log π(y|x). The raw `inputs` list is already
        # repeated `num_generations` times by the sampler, so row i here aligns with row i of `output`.
        output = super()._generate_and_score_completions(inputs)
        ref_prompt_ids, ref_completion_ids = [], []
        for example in inputs:
            if is_conversational(example):
                applied = apply_chat_template(
                    {"prompt": example["prompt"], "completion": example["completion"]},
                    self.processing_class,
                    **self.chat_template_kwargs,
                )
                prompt_text, completion_text = applied["prompt"], applied["completion"]
            else:
                prompt_text, completion_text = example["prompt"], example["completion"]
            p_ids = self.processing_class(text=prompt_text, add_special_tokens=False)["input_ids"]
            c_ids = self.processing_class(text=completion_text, add_special_tokens=False)["input_ids"]
            ref_prompt_ids.append(torch.tensor(p_ids, dtype=torch.long))
            ref_completion_ids.append(torch.tensor(c_ids, dtype=torch.long))

        # Prompt left-padded, completion right-padded (mirrors GRPO's prompt/completion layout), so `_compute_loss` can
        # slice the last `logits_to_keep` positions and mask padding with `ref_completion_mask`.
        device = self.accelerator.device
        pad_id = self._tokenizer.pad_token_id
        output["ref_prompt_ids"] = pad(ref_prompt_ids, pad_id, "left", self.pad_to_multiple_of).to(device)
        output["ref_prompt_mask"] = pad([torch.ones_like(x) for x in ref_prompt_ids], 0, "left").to(device)
        output["ref_completion_ids"] = pad(ref_completion_ids, pad_id, "right", self.pad_to_multiple_of).to(device)
        output["ref_completion_mask"] = pad([torch.ones_like(x) for x in ref_completion_ids], 0, "right").to(device)
        return output

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
        mode = "train" if self.model.training else "eval"

        # 2. u = ∇_θ log π(y | x): the SFT gradient of the *dataset reference* (x, y), a constant vector in parameter
        # space. A separate forward on the reference tokens gives log π(y | x); one backward (no `create_graph`) gives
        # u, which we detach so it acts as a constant in the JVP below. This forward goes through the *unwrapped* model:
        # DDP calls `prepare_for_backward` at the end of every wrapped forward, so a second wrapped forward would corrupt
        # the reducer state that the final `loss.backward()` (on the step-1 graph) relies on. `autograd.grad` also skips
        # DDP's all-reduce, so u is the local (per-rank) SFT gradient — good enough for a preconditioning direction.
        params = [p for p in model.parameters() if p.requires_grad]
        unwrapped_model = self.accelerator.unwrap_model(model)
        ref_completion_mask = inputs["ref_completion_mask"]
        ref_input_ids = torch.cat([inputs["ref_prompt_ids"], inputs["ref_completion_ids"]], dim=1)
        ref_attention_mask = torch.cat([inputs["ref_prompt_mask"], ref_completion_mask], dim=1)
        ref_logits_to_keep = inputs["ref_completion_ids"].size(1)
        ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
            unwrapped_model,
            ref_input_ids,
            ref_attention_mask,
            ref_logits_to_keep,
            compute_entropy=False,
            compute_aux_loss=False,
        )
        ref_logp = (ref_per_token_logps * ref_completion_mask).sum()  # scalar = Σ_batch log π(y | x)
        u = torch.autograd.grad(ref_logp, params, allow_unused=True)
        u = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, u)]

        # Eval-only probe (not used for training): the model's per-token log π(y | x) on the reference.
        ref_logp_mean = ref_logp.detach() / ref_completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["ref_logp"].append(self.accelerator.gather(ref_logp_mean).nanmean().item())

        # 3. c = J u: the directional derivative of each log f_t along u, i.e. c_t = grad log f_t . u. We get all T of
        # them with one JVP, built from two reverse-mode passes ("double backward") to avoid forward-mode AD:
        #   w = grad(v . logps) = J^T v          (linear in the dummy vector v)
        #   c = grad_v (w . u) = grad_v (v^T J u) = J u
        v = torch.zeros_like(logps, requires_grad=True)  # (T,), dummy vector
        w = torch.autograd.grad((v * logps).sum(), params, create_graph=True, allow_unused=True)  # list of param-shaped tensors = J^T v
        inner = sum((w_p * u_p).sum() for w_p, u_p in zip(w, u) if w_p is not None)  # scalar () = w . u = v^T J u
        c = torch.autograd.grad(inner, v, retain_graph=True)[0].detach()  # (T,) = J u

        # 4. Surrogate loss whose gradient is S = J^T c = F u. `c` is detached, so backprop only flows through logps;
        # `retain_graph=True` above keeps the forward graph alive for this backward. Normalized by the number of
        # generated tokens (mean) and by the gradient-accumulation count, matching GRPO (HF auto-scaling is off here).
        loss = - (c * logps).sum() / mask.sum().clamp(min=1.0)
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        loss = loss / normalizer

        # The magnitude of c (= J u, the reference-gradient response of each self-generated token) is the key quantity
        # to watch: it sets the effective step size and can be large, so log it.
        self._metrics[mode]["c_abs_mean"].append(self.accelerator.gather(c.abs().mean()).nanmean().item())
        # Weight-movement probes. `param_drift` = ‖θ_t − θ₀‖ is the reliable signal; ‖θ‖ is near-invariant. No gather:
        # params are replicated across ranks (DDP), so every rank holds identical values.
        with torch.no_grad():
            params_list = list(model.parameters())
            if self._theta0 is None:
                self._theta0 = [p.detach().clone() for p in params_list]
            param_drift = torch.stack([(p - p0).norm() for p, p0 in zip(params_list, self._theta0)]).norm()
            param_norm = torch.stack([p.norm() for p in params_list]).norm()
        self._metrics[mode]["param_drift"].append(param_drift.item())
        self._metrics[mode]["param_norm"].append(param_norm.item())
        return loss
