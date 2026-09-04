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

"""SFT trainer with an always-on KL measurement against a frozen reference model."""

import time

import torch
from accelerate.utils import is_deepspeed_available

from trl import SFTTrainer
from trl.models.utils import prepare_deepspeed, prepare_fsdp
from trl.trainer.utils import (
    create_model_from_path,
    disable_dropout_in_model,
    get_config_model_id,
    patch_chunked_lm_head,
)


def log_exact(state, metrics: dict, fallback=None) -> None:
    """Log to wandb under the exact key names given.

    `Trainer.log` routes through transformers' `rewrite_logs`
    (`integrations/integration_utils.py:550`), which prefixes every key with `train/` unless it
    already starts with `eval_` or `test_`. That turns `time/eval_verification` into
    `train/time/eval_verification` and files evaluation metrics under `train/`. Writing to the
    wandb run directly keeps the namespace intact.

    No `step=` is passed, deliberately. transformers' own `WandbCallback` logs without it
    (`integration_utils.py:900`) and records `train/global_step` as an ordinary field, so wandb's
    internal counter advances once per log call and does *not* track `state.global_step`. Passing
    `step=state.global_step` therefore writes into a different step space, and any value below
    wandb's current counter is dropped with no error -- which silently lost `time/step` entirely.
    Omitting it merges these metrics into the surrounding row, next to `train/global_step`.
    """
    try:
        import wandb

        run = wandb.run
    except ImportError:
        run = None
    if run is not None:
        run.log(metrics)
    elif fallback is not None:
        fallback(metrics)


class LeanSFTTrainer(SFTTrainer):
    """[`SFTTrainer`] that always measures the KL to a frozen reference model.

    The reference model is loaded unconditionally so `train/kl` and `eval/kl` exist for every
    baseline; `args.beta` only decides whether the KL is added to the loss. That makes plain SFT
    (`beta=0`) and SFT+KL (`beta>0`) the same code path with the same observability, differing by
    one number in a YAML file.

    ## Why the loss is computed here rather than by [`SFTTrainer`]

    The KL needs per-token log-probs from *both* the policy and the reference. Getting those the
    obvious way means materialising `[batch, seq, vocab]` logits, which at 8192 tokens and a
    100k vocabulary is ~3.3 GB per sequence in fp32. `SFTTrainer`'s default `chunked_nll` path
    avoids that cost but only ever returns scalar sums (`outputs.logits` is `None`), so there is
    nothing to build a KL from.

    Instead both models are patched with [`patch_chunked_lm_head`], which computes per-token
    log-probs and entropy with an online logsumexp over vocabulary chunks and never materialises
    the full logits. The negative log-likelihood is then just a masked sum over the same
    log-probs, so one policy forward yields both the SFT loss and the policy side of the KL.

    The KL uses the k3 estimator, `exp(r) - r - 1` where `r = logp_ref - logp`, which is
    non-negative and unbiased for `KL(policy || ref)`.

    Cost: one extra 7B forward pass per step for the reference. That is the price of having the
    metric at every step.
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model=model, args=args, **kwargs)

        self.beta = args.beta
        self._logged_devices = False
        self._step_times: list[float] = []
        model_id = get_config_model_id(self.model.config)

        # Per-token log-probs without materialising `[N, vocab]` logits. Temperature is 1.0: this
        # is teacher-forced likelihood, not sampling.
        patch_chunked_lm_head(self.model, chunk_size=args.chunk_size, temperature=1.0)

        ref_path = args.ref_model_path or model_id
        model_init_kwargs = dict(args.model_init_kwargs or {})
        # `device_map="auto"` fails under distributed training.
        if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
            model_init_kwargs["device_map"] = None
        model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)

        # Reload from the checkpoint rather than deep-copying the policy: a deepcopy of a sharded
        # or already-prepared model is fragile, and this is what GRPOTrainer does.
        self.ref_model = create_model_from_path(ref_path, **model_init_kwargs)
        disable_dropout_in_model(self.ref_model)
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        patch_chunked_lm_head(self.ref_model, chunk_size=args.chunk_size, temperature=1.0)

        if self.is_deepspeed_enabled and is_deepspeed_available():
            self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
        elif self.is_fsdp_enabled:
            self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
        else:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    @staticmethod
    def _chunked_log_probs(model, inputs: dict, completion_mask: torch.Tensor) -> torch.Tensor:
        """Per-token log-probs of the gold tokens, shape `[batch, seq - 1]`.

        `labels` is passed as `input_ids` because the patched forward gathers the log-prob of the
        target token id, and `inputs["labels"]` carries `-100` at prompt and padding positions,
        which is not a valid vocabulary index. Which positions count is expressed separately by
        `completion_mask`; the patched forward zeroes everything outside it.

        Everything is pinned to the model's own parameter device. On a multi-GPU node the policy
        and the reference can land on different devices, and the chunked kernel indexes hidden
        states with the completion mask (`hidden_flat[valid_mask]`), which requires both on the
        same device. These `.to()` calls are no-ops whenever the devices already agree.
        """
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        position_ids = inputs.get("position_ids")
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            position_ids=position_ids.to(device) if position_ids is not None else None,
            labels=input_ids,
            completion_mask=completion_mask.to(device),
            use_cache=False,
        )
        return outputs["log_probs"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)

        labels = inputs["labels"]
        # The collator folds the prompt mask into `labels` (`-100` on prompt and padding), so this
        # is the only completion signal available in `inputs`.
        completion_mask = (labels != -100).to(inputs["input_ids"].dtype)
        # The patched forward shifts by one (position i predicts token i+1), so the mask that
        # aligns with its output drops the first column.
        mask = completion_mask[:, 1:].float()
        n_valid = mask.sum()

        if not self._logged_devices:
            self._logged_devices = True
            print(
                f"[devices] policy={next(self.model.parameters()).device} "
                f"ref={next(self.ref_model.parameters()).device} "
                f"inputs={inputs['input_ids'].device} accelerator={self.accelerator.device}",
                flush=True,
            )

        log_probs = self._chunked_log_probs(model, inputs, completion_mask)

        with torch.no_grad():
            ref_log_probs = self._chunked_log_probs(self.ref_model, inputs, completion_mask)
        # The reference can sit on a different device than the policy on a multi-GPU node.
        ref_log_probs = ref_log_probs.to(log_probs.device)
        mask = mask.to(log_probs.device)
        n_valid = n_valid.to(log_probs.device)

        # k3 estimator: non-negative, unbiased for KL(policy || ref), and needs only the log-prob
        # of the realised token rather than the full distribution.
        log_ratio = ref_log_probs - log_probs
        per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0

        nll_sum = -(log_probs * mask).sum()
        kl_sum = (per_token_kl * mask).sum()

        # Match `SFTTrainer`'s normalisation exactly: divide by the global token count when the
        # Trainer supplies it (correct under gradient accumulation), else by the local count.
        if num_items_in_batch is None:
            denominator = n_valid.clamp(min=1.0)
        else:
            denominator = (
                num_items_in_batch.to(nll_sum.device)
                if isinstance(num_items_in_batch, torch.Tensor)
                else num_items_in_batch
            )

        loss = nll_sum / denominator
        if self.beta != 0.0:
            loss = loss + self.beta * (kl_sum / denominator)

        # Whole micro-batch masked (completion-only loss plus truncation). Keep the loss attached
        # to the graph so `.backward()` succeeds and DDP does not hang waiting on a missing param.
        if n_valid == 0:
            loss = log_probs.sum() * 0.0

        self._log_kl_metrics(mode, per_token_kl, log_probs, mask, n_valid)

        if return_outputs:
            return loss, {"log_probs": log_probs, "ref_log_probs": ref_log_probs}
        return loss

    def training_step(self, *args, **kwargs):
        """Time each optimisation step.

        Recorded here rather than derived from `train_runtime` so the number excludes evaluation
        and callback time -- it answers "how expensive is a step", which is what you watch when
        changing batch size, sequence length, or the reference-model forward pass.

        Kept out of `self._metrics` so it can be logged under a bare `time/` namespace; anything
        routed through `Trainer.log` comes back as `train/time/step`.
        """
        start = time.perf_counter()
        output = super().training_step(*args, **kwargs)
        self._step_times.append(time.perf_counter() - start)
        return output

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        super().log(logs, start_time)
        if self._step_times:
            log_exact(self.state, {"time/step": sum(self._step_times) / len(self._step_times)})
            self._step_times.clear()

    def _log_kl_metrics(self, mode, per_token_kl, log_probs, mask, n_valid) -> None:
        """Token-weighted means reduced across ranks.

        A plain `.mean()` of per-rank means would be wrong whenever ranks hold different numbers
        of completion tokens, which is the normal case with variable-length proofs.
        """
        with torch.no_grad():
            sums = torch.stack(
                [
                    (per_token_kl * mask).sum(),
                    (-log_probs * mask).sum(),
                    n_valid,
                ]
            )
            kl_sum, nll_sum, count = self.accelerator.reduce(sums, reduction="sum").tolist()
            if count > 0:
                self._metrics[mode]["kl"].append(kl_sum / count)
                self._metrics[mode]["nll_per_token"].append(nll_sum / count)
