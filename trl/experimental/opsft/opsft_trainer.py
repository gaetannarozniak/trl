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
from ...models import prepare_deepspeed, prepare_fsdp
from ...trainer.grpo_trainer import GRPOTrainer as _GRPOTrainer
from ...trainer.utils import create_model_from_path, get_config_model_id, pad, selective_log_softmax


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
    directional derivatives $c = J u$ are obtained with forward-mode autodiff ([`torch.func.jvp`]), so $J$ is never
    materialized and the memory is independent of both the number of tokens and $\dim\theta$.

    <Tip warning={true}>

    The forward-mode JVP ([`torch.func.jvp`]) that computes $c$ constrains the setup (generation is a separate
    no-grad path and is unaffected, so vLLM composes fine):

    - Load the model with `attn_implementation="eager"` (flash/SDPA attention kernels do not support forward-mode AD).
    - Set `gradient_checkpointing=False` in the [`GRPOConfig`].
    - Prefer `bf16=True` over `fp16=True` (the fp16 gradient scaler interferes with the JVP).
    - Do not use DeepSpeed ZeRO (gradients with respect to partitioned parameters are unavailable).

    </Tip>

    Example:

    ```python
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM
    from trl.experimental.opsft import OPSFTConfig, OPSFTTrainer

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", attn_implementation="eager")
    dataset = load_dataset("trl-lib/tldr", split="train")  # prompt-completion dataset (x, y)
    trainer = OPSFTTrainer(
        model=model,
        args=OPSFTConfig(
            output_dir="Qwen2.5-0.5B-OPSFT",
            gradient_checkpointing=False,
            bf16=True,
            use_vllm=True,  # optional; colocate by default
            compute_eval_kl=True,  # optional; log eval_kl vs. the frozen base model
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
        # Diagonal whitening: 0 = plain OPSFT c; >0 = rescale c_t by an m-probe estimate of ‖∇log f_t‖² (see step 4).
        self.whitening_probes = args.whitening_probes
        self.whitening_damping = args.whitening_damping  # relative damping λ (fraction of mean estimated norm)
        # Eval-only KL diagnostic: hold a frozen base model θ₀ and log KL(policy ‖ base) at eval (see `_log_eval_kl`).
        self.compute_eval_kl = args.compute_eval_kl
        if self.compute_eval_kl and args.beta != 0.0:
            # `beta != 0.0` already makes GRPO load its own frozen reference model (`self.ref_model`); holding a second
            # frozen base model here would keep two copies of θ₀ in memory. Pick one KL source.
            raise ValueError(
                "`compute_eval_kl=True` cannot be combined with `beta != 0.0`: both hold a frozen copy of the base "
                "model. Set `beta=0.0`, or leave `compute_eval_kl=False` and use GRPO's own `kl` metric."
            )
        self.base_model = self._create_base_model(args) if self.compute_eval_kl else None

    def _create_base_model(self, args):
        # Build the frozen base model θ₀ used only for the eval-time KL diagnostic. Independent of GRPO's
        # `beta`-gated `self.ref_model` (which is `None` at `beta == 0.0`): we hold our own copy and never sync it
        # forward, so it stays pinned to the initial weights. Built and prepared exactly like GRPO's reference model
        # (fresh reload of the base checkpoint, put in eval mode via `evaluation_mode=True`).
        model_init_kwargs = dict(args.model_init_kwargs or {})  # copy to avoid mutating model_init_kwargs
        # Distributed training requires device_map=None ("auto" fails)
        if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
            model_init_kwargs["device_map"] = None
        model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
        base_model_id = get_config_model_id(self.accelerator.unwrap_model(self.model).config)
        base_model = create_model_from_path(base_model_id, **model_init_kwargs)
        if self.is_deepspeed_enabled:
            return prepare_deepspeed(base_model, self.accelerator)
        elif self.is_fsdp_enabled:
            return prepare_fsdp(base_model, self.accelerator)
        else:
            return self.accelerator.prepare_model(base_model, evaluation_mode=True)

    def _log_eval_kl(self, input_ids, attention_mask, logits_to_keep, logps, mask, mode):
        # KL between the policy and the frozen base θ₀ over the self-generated completions, via GRPO's k3 estimator
        # `exp(Δ) - Δ - 1` (Δ = log f_base - log f). Never enters the loss; runs under the eval loop's `no_grad`.
        base_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
            self.base_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=False,
            compute_aux_loss=False,
        )
        base_logps = base_per_token_logps[mask.bool()]  # (T,), aligned with `logps`
        per_token_kl = torch.exp(base_logps - logps) - (base_logps - logps) - 1  # k3 estimator of KL(policy ‖ base)
        self._metrics[mode]["kl"].append(self.accelerator.gather(per_token_kl.mean()).nanmean().item())

    def _compute_c_torch_jvp(self, model, param_names, params, u, input_ids, attention_mask, logits_to_keep, mask):
        # c = J u via forward-mode autodiff. J is the Jacobian of the map (θ ↦ per-token logps), so c = J u is exactly
        # the jvp of that map at the current params with tangent u. This never materializes J^T v over all params nor a
        # second-order graph — it propagates a single tangent through one forward. Requires forward-mode-capable ops
        # (eager attention). The functional forward mirrors the text-only path of `_get_per_token_logps_and_entropies`
        # so the returned logps (hence c) align with `logps` from step 1.
        def logps_of_params(*param_values):
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1  # +1: the last logit (next-token pred) is dropped
            params_dict = dict(zip(param_names, param_values))
            logits = torch.func.functional_call(model, params_dict, args=(), kwargs=model_inputs).logits
            logits = logits[:, :-1, :][:, -logits_to_keep:, :] / self.temperature  # (B, logits_to_keep, V)
            per_token_logps = selective_log_softmax(logits, input_ids[:, -logits_to_keep:])  # (B, logits_to_keep)
            return per_token_logps[mask.bool()]  # (T,)

        _, c = torch.func.jvp(logps_of_params, tuple(params), tuple(u))
        return c.detach()  # (T,) = J u

    def _diag_whiten(self, c, model, param_names, params, input_ids, attention_mask, logits_to_keep, mask):
        # Diagonally whiten c: return c_t / (‖∇log f_t‖² + λ), the steepest in-subspace step when the per-token score
        # vectors are orthogonal (so JJ^T = diag(‖∇log f_t‖²)). J is never materialized: diag(JJ^T)_t = ‖∇log f_t‖² is
        # estimated by Hutchinson — E_v[(J v)_t²] = ‖∇log f_t‖² for random ±1 v — averaging `whitening_probes`
        # random-tangent JVPs (each one extra forward-mode pass). λ = `whitening_damping` × mean(diag) is a small
        # relative damping that keeps tiny/noisy estimates from blowing up the coefficient (see the config).
        diag = torch.zeros_like(c)
        for _ in range(self.whitening_probes):
            v = [(torch.randint(0, 2, p.shape, device=p.device) * 2 - 1).to(p.dtype) for p in params]  # Rademacher ±1
            jv = self._compute_c_torch_jvp(model, param_names, params, v, input_ids, attention_mask, logits_to_keep, mask)
            diag += jv * jv
            del v, jv  # free the param-sized tangent + its forward before the next probe
        diag /= self.whitening_probes
        diag = diag + self.whitening_damping * diag.mean()
        return c / diag

    def _log_peak_memory(self, mode):
        # Peak CUDA memory allocated during this `_compute_loss` call (the tracker is reset at its start), so the two
        # `c` implementations can be compared. Excludes the later surrogate `loss.backward()`, which is identical for
        # both. No gather: report each rank's own peak.
        if torch.cuda.is_available():
            self._metrics[mode]["peak_mem_gb"].append(torch.cuda.max_memory_allocated() / 1e9)

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
        # Reset the CUDA peak tracker so `_log_peak_memory` reports the peak of *this* call (double-backward vs jvp).
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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
        )
        logps = per_token_logps[mask.bool()]  # (T,), keep only the (non-padded) generated tokens
        mode = "train" if self.model.training else "eval"

        # Eval-only KL(policy ‖ base θ₀) diagnostic over the self-generated eval completions (see `_log_eval_kl`).
        if self.compute_eval_kl and mode == "eval":
            self._log_eval_kl(input_ids, attention_mask, logits_to_keep, logps, mask, mode)

        # 2. Reference forward: log π(y | x) on the *dataset reference* (x, y). This gives both the reference
        # cross-entropy (logged in both modes) and, in training, the SFT direction u. The forward goes through the
        # *unwrapped* model: DDP calls `prepare_for_backward` at the end of every wrapped forward, so a second wrapped
        # forward would corrupt the reducer state that the final `loss.backward()` (on the step-1 graph) relies on.
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

        # Reference cross-entropy per token (−mean log π(y | x)): logged in both modes and, at eval, returned as the
        # eval loss (directly comparable to a plain-SFT baseline's eval loss).
        ref_logp_mean = ref_logp.detach() / ref_completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["ref_logp"].append(self.accelerator.gather(ref_logp_mean).nanmean().item())

        # Weight-movement probes (both modes). `param_drift` = ‖θ_t − θ₀‖ is the reliable signal; ‖θ‖ is near-invariant.
        # No gather: params are replicated across ranks (DDP), so every rank holds identical values.
        with torch.no_grad():
            params_list = list(model.parameters())
            if self._theta0 is None:
                self._theta0 = [p.detach().clone() for p in params_list]
            param_drift = torch.stack([(p - p0).norm() for p, p0 in zip(params_list, self._theta0)]).norm()
            param_norm = torch.stack([p.norm() for p in params_list]).norm()
        self._metrics[mode]["param_drift"].append(param_drift.item())
        self._metrics[mode]["param_norm"].append(param_norm.item())

        # Eval takes no optimizer step and the eval loop runs under `torch.no_grad`, so the autograd-based direction u
        # and the JVP c below cannot be built (and enabling grad here would break DDP's reducer). Return the reference
        # cross-entropy as the eval loss; the diagnostics above plus reward-based accuracy are all we track at eval.
        if mode == "eval":
            self._log_peak_memory(mode)
            return -ref_logp_mean

        # 3. u = ∇_θ log π(y | x): the SFT gradient of the reference, a constant vector in parameter space. We take the
        # gradient of the *mean* (÷ token count) log π(y | x) rather than the sum, so u is the per-token SFT gradient —
        # matching both the generated-token normalization of the loss (step 5) and a plain-SFT baseline (mean cross
        # entropy), which makes the step scale-invariant to answer length. One backward (no `create_graph`) gives u,
        # which we detach so it acts as a constant in the JVP below. `autograd.grad` skips DDP's all-reduce, so u is the
        # local (per-rank) SFT gradient — good enough for a preconditioning direction.
        named_params = [(n, p) for n, p in unwrapped_model.named_parameters() if p.requires_grad]
        param_names = [n for n, _ in named_params]
        params = [p for _, p in named_params]
        ref_token_count = ref_completion_mask.sum().clamp(min=1.0)
        u = torch.autograd.grad(ref_logp / ref_token_count, params, allow_unused=True)
        u = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, u)]
        u_norm = torch.sqrt(sum((up * up).sum() for up in u))  # ‖u‖ (for the grad-alignment metric)

        # 4. c = J u (see `_compute_c_torch_jvp`). `c_raw` is the raw directional derivatives; when `whitening_probes`
        # > 0 the coefficient `c` is the diagonally-whitened `c_raw` (see `_diag_whiten`).
        c_raw = self._compute_c_torch_jvp(
            unwrapped_model, param_names, params, u, input_ids, attention_mask, logits_to_keep, mask
        )
        del u  # u is unused past the JVP; free it before the whitening probes / alignment backward
        if self.whitening_probes:
            c = self._diag_whiten(
                c_raw, unwrapped_model, param_names, params, input_ids, attention_mask, logits_to_keep, mask
            )
        else:
            c = c_raw

        # Angle between the SFT gradient u = ∇log π(y|x) and the update direction S = J^T c: cos = <u,S>/(‖u‖‖S‖).
        # <u,S> = (J u)·c = c_raw·c is free; ‖S‖ = ‖J^T c‖ needs one extra (local) backward of the surrogate.
        grad_s = torch.autograd.grad((c * logps).sum(), params, retain_graph=True, allow_unused=True)
        s_norm = torch.sqrt(sum((g * g).sum() for g in grad_s if g is not None))
        cos = (c_raw * c).sum() / (u_norm * s_norm).clamp(min=1e-12)
        self._metrics[mode]["grad_alignment_cos"].append(self.accelerator.gather(cos).nanmean().item())
        del grad_s

        # 5. Surrogate loss whose gradient is S = J^T c = F u. `c` is detached, so backprop only flows through logps;
        # the step-1 forward graph is kept alive for this backward. Normalized by the number of generated tokens (mean)
        # and by the gradient-accumulation count, matching GRPO (HF auto-scaling is off here).
        loss = - (c * logps).sum() / mask.sum().clamp(min=1.0)
        loss = loss / self.current_gradient_accumulation_steps

        # The magnitude of c (= J u, the reference-gradient response of each self-generated token) is the key quantity
        # to watch: it sets the effective step size and can be large, so log it.
        self._metrics[mode]["c_abs_mean"].append(self.accelerator.gather(c.abs().mean()).nanmean().item())
        self._log_peak_memory(mode)
        return loss
