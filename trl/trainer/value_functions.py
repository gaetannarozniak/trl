import collections
import copy
import time
from dataclasses import dataclass
from html import escape

import torch
import torch.nn as nn
from transformers import PreTrainedModel, is_wandb_available

from .utils import pad
from .value_function_prompts import (
    ANALYSIS_ATTEMPT,
    ANALYSIS_MARKER,
    ANALYSIS_PREFILL,
    ANALYSIS_PROMPT,
    ANALYSIS_ATTEMPT_TOKENS,
)


if is_wandb_available():
    import wandb


@dataclass(frozen=True)
class ValueFunctionConfig:
    zero_init_value_head: bool = True
    value_lambda: float | None = None
    value_loss_skip_unanimous: bool = False  # drop groups whose rollouts all agree from the value loss
    adv_lambda: float | None = None
    critic_lr: float = 1e-5
    critic_layer: int | None = None  # read the backbone after this block instead of the last one
    privileged_template: str | None = None  # extra user turn shown only to the critic, e.g. "Solution:\n{solution}"
    probe_token: str | None = None  # interleave this token after every completion token and read the value there
    yes_no_tokens_as_value: bool = False  # read the value as p(yes) / (p(yes) + p(no)) at the probe slots
    target: str = (
        "residual"  # "residual" for R - Rloo, "plain" for R, "grpo" for linear interpolation between Rloo and R
    )
    example_proof: bool = False  # show the critic the dataset's own worked solution, through `{example_proof}`
    analysis_model: str | None = None  # frozen model that reviews a group's rollouts into `{analysis}`
    analysis_max_tokens: int = 4096
    analysis_gpu_memory_utilization: float = 0.05  # ~14 GiB, against ~4 actually needed
    analysis_max_model_length: int = 32768

    def __post_init__(self):
        if self.yes_no_tokens_as_value and self.probe_token is None:
            raise ValueError("yes_no_tokens_as_value requires probe_token")
        if self.example_proof and (self.privileged_template is None or "{example_proof}" not in self.privileged_template):
            raise ValueError("example_proof requires `{example_proof}` in privileged_template")
        if self.analysis_model is None:
            return
        if not self.value_loss_skip_unanimous:
            raise ValueError("analysis_model requires value_loss_skip_unanimous=True")
        if self.privileged_template is None or "{analysis}" not in self.privileged_template:
            raise ValueError("analysis_model requires `{analysis}` in privileged_template")


# maybe still need to deal with the vllm iter_name_params for the critic, it seems redundant in the way we find the critic params (already two ways)
# remove scale_rewards for the grpo trainer
class ValueFunction:
    def __init__(self, args: ValueFunctionConfig):
        self.args = args
        self._last_plot_step = -1
        self._processing_class = None
        self._logged_privileged = False
        self._probe_id = None
        self._pending_curves = {}
        # Seconds spent in the critic, summed over the micro-batches of one optimizer step and flushed by
        # log_metrics, the same way _analysis_stats is. `step_time` is per optimizer step and covers
        # generation, every forward and backward and the optimizer, so subtracting these leaves the policy.
        self._times = collections.defaultdict(float)
        self._num_generations = None
        self._yes_ids = None
        self._no_ids = None
        self._policy_head = None
        self._analyser = None
        self._analysis_example = None
        self._analysis_stats = {}
        self._logged_analysis = False

    def extra_inputs(self, rows, processing_class, completions=None, rewards=None, args=None) -> dict:
        """Privileged prompts, tokenised and merged into the batch so `get_values` can read them back."""
        self._processing_class = processing_class
        if args is not None:  # absent only when called directly from a test
            self._num_generations = args.num_generations
        if self.args.probe_token is not None and self._probe_id is None:
            ids = processing_class.encode(self.args.probe_token, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"probe_token must be a single token, {self.args.probe_token!r} is {len(ids)}")
            self._probe_id = ids[0]
        if self.args.yes_no_tokens_as_value and self._yes_ids is None:
            # Every vocab token starting with "yes" or "no", ignoring case and leading space
            texts = processing_class.batch_decode([[i] for i in range(processing_class.vocab_size)])
            special = set(processing_class.all_special_ids)
            yes, no = [], []
            for i, text in enumerate(texts):
                if i in special:
                    continue
                text = text.casefold().lstrip()
                if text.startswith("yes"):
                    yes.append(i)
                elif text.startswith("no"):
                    no.append(i)
            both = set(yes) & set(no)
            self._yes_ids = [i for i in yes if i not in both]
            self._no_ids = [i for i in no if i not in both]
            if not self._yes_ids or not self._no_ids:
                raise ValueError("yes_no_tokens_as_value found no 'yes' or no 'no' tokens in the vocabulary")
        if self.args.privileged_template is None:
            return {}
        analyses = self._group_analyses(rows, completions, rewards, args)
        conversation = [
            row["prompt"] + [{"role": "user", "content": self.args.privileged_template.format(**row, **extra)}]
            for row, extra in zip(rows, analyses)
        ]
        ids = processing_class.apply_chat_template(
            conversation=conversation, add_generation_prompt=True, tokenize=True, return_dict=True
        )["input_ids"]
        ids = [torch.tensor(i) for i in ids]
        return {
            "critic_prompt_ids": pad(ids, padding_value=processing_class.pad_token_id, padding_side="left"),
            "critic_prompt_mask": pad([torch.ones_like(i) for i in ids], padding_value=0, padding_side="left"),
        }

    def _n_tokens(self, text: str) -> int:
        return len(self._processing_class(text, add_special_tokens=False)["input_ids"])

    @staticmethod
    def _extract_block(out: str) -> str:
        """The two headed lists from the analyser's reply, dropping its reasoning and anything trailing."""
        if ANALYSIS_MARKER not in out:
            return ""
        kept = []
        for line in out[out.index(ANALYSIS_MARKER) :].splitlines():
            line = line.rstrip()
            if line in (ANALYSIS_MARKER, "CORRECT:") or line.startswith("- "):
                kept.append(line)
            elif kept and line:  # prose, a fence, anything else: the block is over
                break
        return "\n".join(kept)

    def _group_analyses(self, rows, completions, rewards, args) -> list[dict]:
        """One `{analysis}` per row: a frozen model reviews the whole group's rollouts against the reference.

        Unanimous groups are skipped, since `value_loss_skip_unanimous` gives their rows no loss weight anyway.
        """
        if self.args.analysis_model is None:
            return [{} for _ in rows]
        from vllm import LLM, SamplingParams

        if self._analyser is None:
            self._analyser = LLM(
                model=self.args.analysis_model,
                gpu_memory_utilization=self.args.analysis_gpu_memory_utilization,
                max_model_len=self.args.analysis_max_model_length,
                # one sequence per group; the default of 256 makes the profiling pass allocate enormously
                max_num_seqs=max(1, args.per_device_train_batch_size // args.num_generations),
                # both as in VLLMGeneration: use this rank's device rather than cuda:0, and keep the v1 memory
                # profiler from being misled by the default 32768
                distributed_executor_backend="external_launcher",
                max_num_batched_tokens=4096,
                enforce_eager=True,
            )
        sampling = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k or -1,
            min_p=args.min_p or 0.0,
            repetition_penalty=args.repetition_penalty,
            max_tokens=self.args.analysis_max_tokens,
        )

        groups = {}
        for row, prompt in enumerate(str(r["prompt"]) for r in rows):
            groups.setdefault(prompt, []).append(row)
        groups = [g for g in groups.values() if 0 < sum(rewards[i] > 0 for i in g) < len(g)]

        prompts, truncated, total = [], 0, 0
        for g in groups:
            question, solution = rows[g[0]]["prompt"][-1]["content"], rows[g[0]]["solution"]
            scaffold = ANALYSIS_PROMPT.format(question=question, solution=solution, n=len(g), attempts="")
            # what the attempts may occupy once the output and the measured scaffolding have their share
            room = self.args.analysis_max_model_length - self.args.analysis_max_tokens - self._n_tokens(scaffold)
            keep = max(1, room // len(g) - ANALYSIS_ATTEMPT_TOKENS)  # end of each attempt: that is where it lands

            texts = [completions[i][0]["content"] if isinstance(completions[i], list) else completions[i] for i in g]
            ids = self._processing_class(texts, add_special_tokens=False)["input_ids"]
            truncated += sum(len(x) > keep for x in ids)
            total += len(texts)
            texts = [self._processing_class.decode(x[-keep:]) if len(x) > keep else t for x, t in zip(ids, texts)]
            attempts = "\n\n".join(
                ANALYSIS_ATTEMPT.format(i=n + 1, verdict="CORRECT" if rewards[i] > 0 else "WRONG", completion=text)
                for n, (i, text) in enumerate(zip(g, texts))
            )
            text = ANALYSIS_PROMPT.format(question=question, solution=solution, n=len(g), attempts=attempts)
            # prefill the reasoning header, or a small model just copies the block template and skips it
            prompts.append(
                self._processing_class.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
                )
                + ANALYSIS_PREFILL
            )

        analyses = [{"analysis": ""} for _ in rows]
        if not prompts:
            self._analysis_stats = {"analysis/seconds": 0.0, "analysis/groups": 0}
            return analyses
        started = time.perf_counter()
        outputs = [o.outputs[0].text for o in self._analyser.generate(prompts, sampling)]
        elapsed = time.perf_counter() - started
        failures, raw_len, kept_len = 0, [], []
        for g, prompt, out in zip(groups, prompts, outputs):
            raw_len.append(len(out))
            kept = self._extract_block(out)
            failures += kept == ""
            kept_len.append(len(kept))
            for i in g:
                analyses[i] = {"analysis": kept}
            if self._analysis_example is None:
                self._analysis_example = (prompt, out, kept)
        self._analysis_stats = {
            "analysis/seconds": elapsed,
            "analysis/groups": len(outputs),
            "analysis/format_failure_rate": failures / len(outputs),
            "analysis/raw_chars": sum(raw_len) / len(raw_len),
            "analysis/extracted_chars": sum(kept_len) / len(kept_len),
            "analysis/clipped_rate": sum(len(o) >= 4 * self.args.analysis_max_tokens for o in outputs) / len(outputs),
            "analysis/truncated_rate": truncated / max(total, 1),
        }
        return analyses

    def get_critic(self, policy_net: PreTrainedModel) -> nn.ModuleDict:
        head_in = policy_net.config.get_text_config().hidden_size
        backbone = copy.deepcopy(policy_net.base_model)
        if self.args.critic_layer is not None:
            if not 1 <= self.args.critic_layer <= len(backbone.layers):
                raise ValueError(f"critic_layer must be in [1, {len(backbone.layers)}], got {self.args.critic_layer}")
            # Dropping the blocks above it, so `last_hidden_state` is that layer's activation
            backbone.layers = backbone.layers[: self.args.critic_layer]
            backbone.config.num_hidden_layers = self.args.critic_layer
        critic = nn.ModuleDict(
            {
                "backbone": backbone,
                "head": nn.Sequential(
                    nn.Linear(head_in, 1, dtype=policy_net.dtype),
                    nn.Tanh(),
                ),
            }
        )
        if self.args.zero_init_value_head:
            nn.init.zeros_(critic["head"][-2].weight)
            nn.init.zeros_(critic["head"][-2].bias)
        critic["head"].requires_grad_(not self.args.yes_no_tokens_as_value)
        # Shared with the policy, not a copy: no extra memory or state_dict entry, and it tracks the policy's
        # device. Deliberately not frozen: freezing would freeze the policy's own lm_head.
        self._policy_head = policy_net.get_output_embeddings()
        if self.args.target == "grpo":
            critic.requires_grad_(False)
        return critic

    def _critic_params(self, model: PreTrainedModel, accelerator):
        critic_ids = {id(p) for p in accelerator.unwrap_model(model).critic.parameters()}
        critic, policy = [], []
        for p in model.parameters():
            (critic if id(p) in critic_ids else policy).append(p)
        return critic, policy

    def add_critic_optimizer(
        self,
        model: PreTrainedModel,
        decay_parameter_names: set[str],
        weight_decays,
        optimizer: torch.optim.Optimizer,
    ):
        critic = {n: p for n, p in model.named_parameters() if n.startswith("critic.") and p.requires_grad}
        critic_ids = {id(p) for p in critic.values()}
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in critic_ids]
        for weight_decay, names in (
            (weight_decays, [n for n in critic if n in decay_parameter_names]),
            (0.0, [n for n in critic if n not in decay_parameter_names]),
        ):
            if names:
                optimizer.add_param_group(
                    {
                        "params": [critic[n] for n in names],
                        "lr": self.args.critic_lr,
                        "weight_decay": weight_decay,
                    }
                )
        return optimizer

    def _clip_grad_norm(self, model: PreTrainedModel, accelerator, grad_norm: float):
        start = time.perf_counter()
        critic, policy = self._critic_params(model, accelerator)
        accelerator.clip_grad_norm_(critic, grad_norm)
        norm = accelerator.clip_grad_norm_(policy, grad_norm)
        # Once per optimizer step, but over every parameter of both models, which is not free at this size.
        self._times["time/clip_grad_norm"] += time.perf_counter() - start
        return norm

    @torch.no_grad()
    def _gae_advantages(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def get_advantages(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.detach()
        if self.args.adv_lambda is not None:
            return self._gae_advantages(values, mask)
        return (values[:, 1:] - values[:, :-1]) * mask

    def _probe_hidden(self, critic, input_ids, attention_mask, logits_to_keep) -> torch.Tensor:
        """Interleave a probe token after every completion token and read the backbone at those slots.

        Nothing attends to a probe but itself, and the real tokens keep their own positions, so their forward
        pass is the one they would have had alone. A probe takes the position of the token it precedes, so it
        sees the same distances a real next token would.
        """
        B, T, device = input_ids.size(0), logits_to_keep, input_ids.device
        P = input_ids.size(1) - T  # uniform: prompts are left-padded
        L = P + 2 * T + 1

        seq = input_ids.new_empty(B, L)
        seq[:, :P], seq[:, P::2], seq[:, P + 1 :: 2] = input_ids[:, :P], self._probe_id, input_ids[:, P:]
        slots = torch.arange(L, device=device)
        position_ids = torch.where(slots < P, slots, P + (slots - P) // 2).expand(B, -1)

        is_probe = torch.zeros(L, dtype=torch.bool, device=device)
        is_probe[P::2] = True
        valid = torch.zeros(B, L, dtype=torch.bool, device=device)
        valid[:, :P], valid[:, P + 1 :: 2], valid[:, P::2] = attention_mask[:, :P].bool(), attention_mask[:, P:].bool(), True
        causal = (slots.unsqueeze(0) <= slots.unsqueeze(1)) & ~is_probe
        # the diagonal is always kept, so a padding query never gets an all-masked row and softmaxes to NaN
        visible = (causal & valid.unsqueeze(1)) | torch.eye(L, dtype=torch.bool, device=device)

        # additive, not boolean: a 4D boolean mask is cast to 1.0/0.0 and added, which masks nothing
        dtype = critic["backbone"].get_input_embeddings().weight.dtype
        mask = torch.zeros(B, 1, L, L, dtype=dtype, device=device)
        mask.masked_fill_(~visible.unsqueeze(1), torch.finfo(dtype).min)
        hidden = critic["backbone"](
            input_ids=seq, attention_mask=mask, position_ids=position_ids
        ).last_hidden_state
        return hidden[:, P::2, :]

    def _yes_no_values(self, hidden: torch.Tensor) -> torch.Tensor:
        """Value as the normalised yes-probability at each probe slot: `2 * p(yes) / (p(yes) + p(no)) - 1`."""
        logits = self._policy_head(hidden).float()
        yes = torch.logsumexp(logits[..., self._yes_ids], dim=-1)
        no = torch.logsumexp(logits[..., self._no_ids], dim=-1)
        return 2 * torch.sigmoid(yes - no) - 1

    def get_values(
        self,
        critic: nn.ModuleDict,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        start = time.perf_counter()
        try:
            return self._get_values(critic, input_ids, attention_mask, logits_to_keep, inputs)
        finally:
            self._times["time/critic_forward"] += time.perf_counter() - start

    def _get_values(
        self,
        critic: nn.ModuleDict,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.args.target == "grpo":
            lengths = inputs["completion_mask"].sum(dim=1, keepdim=True).clamp(min=1)
            positions = torch.arange(logits_to_keep + 1, device=lengths.device).expand(lengths.size(0), -1)
            baseline = inputs["rloo_baseline"].unsqueeze(-1)
            return baseline + (inputs["rewards"].unsqueeze(-1) - baseline) * positions / lengths
        if "critic_prompt_ids" in inputs:
            device = inputs["completion_ids"].device
            input_ids = torch.cat([inputs["critic_prompt_ids"].to(device), inputs["completion_ids"]], dim=1)
            attention_mask = torch.cat([inputs["critic_prompt_mask"].to(device), inputs["completion_mask"]], dim=1)
        if self.args.probe_token is not None:
            hidden = self._probe_hidden(critic, input_ids, attention_mask, logits_to_keep)
        else:
            hidden = critic["backbone"](input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            hidden = hidden[:, -(logits_to_keep + 1) :, :]
        if self.args.yes_no_tokens_as_value:
            values = self._yes_no_values(hidden)
        else:
            values = critic["head"](hidden).squeeze(-1)
        if self.args.target == "residual":
            values = values + inputs["rloo_baseline"].unsqueeze(-1)
        return values  # (B, logits_to_keep + 1)

    @torch.no_grad()
    def _lambda_returns(self, values: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Regression target as an exponentially weighted mix of the n-step returns, rather than the reward alone.

        gamma is 1 and the only reward is terminal, so the last token bootstraps on the outcome itself:
        `value_lambda=1` telescopes back to the plain reward, `0` regresses on the next value alone.
        """
        values = values.detach()
        positions = torch.arange(mask.size(1), device=mask.device).expand_as(mask)
        is_last = positions == mask.sum(dim=1, keepdim=True) - 1
        deltas = (torch.where(is_last, rewards, values[:, 1:]) - values[:, :-1]) * mask
        returns = torch.zeros_like(deltas)
        running = torch.zeros_like(deltas[:, 0])
        for t in range(mask.size(1) - 1, -1, -1):
            running = deltas[:, t] + self.args.value_lambda * running * mask[:, t]
            returns[:, t] = running
        return returns + values[:, :-1]

    def get_value_loss(
        self, values: torch.Tensor, inputs: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        target = inputs["rewards"].unsqueeze(-1)
        if self.args.value_lambda is not None:
            target = self._lambda_returns(values, target, mask)
        if self.args.value_loss_skip_unanimous:  # after the scan, so the recursion keeps its own row lengths
            centered = inputs["rewards"] - inputs["rloo_baseline"]
            mask = mask * (centered.abs() > 1e-8).to(mask.dtype).unsqueeze(-1)

        return ((values[:, :-1] - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def log_privileged_example(self, accelerator, inputs):
        """Log one critic prompt and its generation, once, to eyeball that the privileged text is wired right."""
        if self._logged_privileged or "critic_prompt_ids" not in inputs:
            return
        if not (accelerator.is_main_process and is_wandb_available() and wandb.run is not None):
            return
        self._logged_privileged = True

        decode = self._processing_class.decode
        prompt = decode(inputs["critic_prompt_ids"][0].tolist(), skip_special_tokens=False)
        completion_ids = inputs["completion_ids"][0][inputs["completion_mask"][0].bool()]
        completion = decode(completion_ids.tolist(), skip_special_tokens=False)
        block = (
            '<div style="white-space:pre-wrap;font:12px ui-monospace,monospace;background:{};'
            'padding:8px;border-radius:4px;margin:4px 0 12px">{}</div>'
        )
        html = (
            '<!DOCTYPE html><meta charset="utf-8">'
            '<div style="background:#fff;padding:8px;font:13px ui-sans-serif,system-ui,sans-serif;color:#111827">'
            f"<b>critic prompt</b>{block.format('#eff6ff', escape(prompt))}"
            f"<b>completion</b>{block.format('#f9fafb', escape(completion))}</div>"
        )
        wandb.log({"privileged/example": wandb.Html(html, inject=False)})

    @torch.no_grad()
    def log_analysis_example(self, accelerator):
        """Log one analyser prompt with its full answer and the extracted block, once."""
        if self._logged_analysis or self._analysis_example is None:
            return
        if not (accelerator.is_main_process and is_wandb_available() and wandb.run is not None):
            return
        self._logged_analysis = True
        block = (
            '<b>{}</b><div style="white-space:pre-wrap;font:12px ui-monospace,monospace;background:{};'
            'padding:8px;border-radius:4px;margin:4px 0 12px">{}</div>'
        )
        sections = zip(("analyser prompt", "raw answer", "extracted"), ("#eff6ff", "#f9fafb", "#ecfdf5"))
        html = (
            '<!DOCTYPE html><meta charset="utf-8">'
            '<div style="background:#fff;padding:8px;font:13px ui-sans-serif,system-ui,sans-serif;color:#111827">'
            + "".join(block.format(t, c, escape(x)) for (t, c), x in zip(sections, self._analysis_example))
            + "</div>"
        )
        wandb.log({"analysis/example": wandb.Html(html, inject=False)})

    @torch.no_grad()
    def log_advantage_plot(self, accelerator, inputs, advantages, step, every=10):
        """Log an HTML chart of the cumulative advantage along the completions of one prompt.

        A group's `num_generations` rollouts are shuffled across the micro-batches of one optimizer step, so
        each call contributes the few rows it holds and the chart is emitted once a group is whole. The group
        is picked among those holding both a correct and an incorrect rollout, since the point is to compare
        how credit is spread in either case. Blue is a correct rollout, vermillion an incorrect one
        (Okabe-Ito, which stays separable under colour-vision deficiency). Emitted as inline SVG rather than
        through a plotting library, so it needs no dependency beyond wandb, and logged without an explicit
        `step=`: wandb's own counter runs ahead of `global_step` and anything below it is silently discarded.
        """
        if step % every or step == self._last_plot_step:
            return
        if not (accelerator.is_main_process and is_wandb_available() and wandb.run is not None):
            return

        mask = inputs["completion_mask"]
        correct = inputs["rewards"] > 0
        prompt_ids = inputs.get("critic_prompt_ids", inputs["prompt_ids"])
        for row, prompt in enumerate(inputs["prompt_ids"].tolist()):
            length = int(mask[row].sum())
            if length == 0:
                continue
            deltas = advantages[row, :length].float().cpu()
            cumulative = deltas.cumsum(0)
            self._pending_curves.setdefault(tuple(prompt), []).append(
                {
                    "prompt_ids": prompt_ids[row].tolist(),
                    "completion_ids": inputs["completion_ids"][row, :length].tolist(),
                    "reward": inputs["rewards"][row].item(),
                    "ok": bool(correct[row]),
                    "delta": deltas.tolist(),
                    "cum": cumulative.tolist(),
                }
            )
        whole = [rollouts for rollouts in self._pending_curves.values() if len(rollouts) >= self._num_generations]
        if not whole:
            return
        mixed = [rollouts for rollouts in whole if 0 < sum(r["ok"] for r in rollouts) < len(rollouts)]
        group = (mixed or whole)[0]
        self._last_plot_step = step
        self._pending_curves.clear()

        curves = []
        for r in group:
            length = len(r["delta"])
            stride = max(1, length // 400)  # keeps the payload small on long completions
            curves.append((list(range(0, length, stride)), r["cum"][::stride], r["ok"]))

        width, height, left, right, top, bottom = 860, 460, 64, 18, 32, 44
        xmax = max(xs[-1] for xs, _, _ in curves) or 1
        flat = [y for _, ys, _ in curves for y in ys] + [0.0]
        lo, hi = min(flat), max(flat)
        pad = (hi - lo) * 0.06 or 0.1
        lo, hi = lo - pad, hi + pad

        def fx(x):
            return left + (width - left - right) * x / xmax

        def fy(y):
            return top + (height - top - bottom) * (hi - y) / (hi - lo)

        svg = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg">']
        for i in range(5):
            y = lo + (hi - lo) * i / 4
            svg.append(f'<line x1="{left}" y1="{fy(y):.1f}" x2="{width - right}" y2="{fy(y):.1f}" stroke="#e5e7eb"/>')
            svg.append(f'<text x="{left - 8}" y="{fy(y) + 4:.1f}" text-anchor="end" fill="#6b7280">{y:.3g}</text>')
            x = xmax * i / 4
            svg.append(
                f'<text x="{fx(x):.1f}" y="{height - bottom + 18}" text-anchor="middle" '
                f'fill="#6b7280">{x:.0f}</text>'
            )
        svg.append(
            f'<line x1="{left}" y1="{fy(0):.1f}" x2="{width - right}" y2="{fy(0):.1f}" stroke="#9ca3af" '
            f'stroke-dasharray="2 3"/>'
        )
        for gen, (xs, ys, ok) in enumerate(curves):
            color, label = ("#0072B2", "correct") if ok else ("#D55E00", "incorrect")
            path = " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(xs, ys))
            svg.append(
                f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" opacity="0.55">'
                f"<title>gen {gen} - {label}</title></polyline>"
            )
            # the index at the curve's end is what ties it to the completion listed below
            svg.append(
                f'<text x="{fx(xs[-1]) + 4:.1f}" y="{fy(ys[-1]) + 3:.1f}" font-size="10" fill="{color}">{gen}</text>'
            )
        mid = (top + height - bottom) / 2
        svg.append(
            f'<text x="{left}" y="18" font-size="13" fill="#111827">step {step} - {len(curves)} generations - '
            f'mean reward {sum(r["reward"] for r in group) / len(group):.2f}</text>'
        )
        svg.append(
            f'<text x="{(left + width - right) / 2:.0f}" y="{height - 8}" text-anchor="middle" fill="#6b7280">'
            f"completion token</text>"
        )
        svg.append(
            f'<text x="16" y="{mid:.0f}" transform="rotate(-90 16 {mid:.0f})" text-anchor="middle" fill="#6b7280">'
            f"cumulative advantage</text>"
        )
        svg.append("</svg>")

        legend = "".join(
            f'<span style="margin-right:20px"><svg width="26" height="10" style="vertical-align:middle">'
            f'<line x1="1" y1="5" x2="25" y2="5" stroke="{color}" stroke-width="2"/></svg> {text}</span>'
            for color, text in (("#0072B2", "correct rollout"), ("#D55E00", "incorrect rollout"))
        )
        html = (
            '<!DOCTYPE html><meta charset="utf-8">'
            '<div style="background:#fff;padding:8px;color:#111827;font:11px ui-sans-serif,system-ui,sans-serif">'
            f'{"".join(svg)}<div style="padding-top:6px;color:#374151">{legend}</div>'
            f"{self._completions_html(group)}</div>"
        )
        wandb.log({"advantages/cumsum": wandb.Html(html, inject=False)})

    def _completions_html(self, group) -> str:
        """The same generations as the curves, token by token, shaded blue for positive and red for negative.

        One scale for the whole group, so intensity is comparable between generations. Tokens carry both the
        cumulative and the per-token value, and a button swaps which one drives the colour.
        """
        scale_cum = max((abs(v) for r in group for v in r["cum"]), default=0.0) or 1.0
        scale_delta = max((abs(v) for r in group for v in r["delta"]), default=0.0) or 1.0

        def shade(value, scale):  # white -> #2166ac positive, white -> #b2182b negative (ColorBrewer RdBu)
            t = max(-1.0, min(1.0, value / scale))
            r, g, b = (33, 102, 172) if t > 0 else (178, 24, 43)
            t = abs(t)
            return f"#{round(255 - (255 - r) * t):02x}{round(255 - (255 - g) * t):02x}{round(255 - (255 - b) * t):02x}"

        prompt = escape(self._processing_class.decode(group[0]["prompt_ids"], skip_special_tokens=True))
        blocks = [
            '<details open style="margin-top:10px"><summary style="cursor:pointer;color:#374151">prompt</summary>'
            f'<div style="white-space:pre-wrap;font:12px ui-monospace,monospace;background:#f9fafb;padding:8px;'
            f'border-radius:4px;margin:4px 0">{prompt}</div></details>'
        ]
        for gen, r in enumerate(group):
            delta, cum, length = r["delta"], r["cum"], len(r["delta"])
            pieces = self._processing_class.batch_decode([[i] for i in r["completion_ids"]])
            spans = [
                f'<span data-c="{shade(cum[i], scale_cum)}" data-d="{shade(delta[i], scale_delta)}" '
                f'style="background:{shade(cum[i], scale_cum)}" '
                f'title="token {i} | cumsum {cum[i]:+.4g} | delta {delta[i]:+.4g}">{escape(piece)}</span>'
                for i, piece in enumerate(pieces)
            ]
            colour, verdict = ("#0072B2", "&#10003; correct") if r["ok"] else ("#D55E00", "&#10007; incorrect")
            # collapsed by default: 16 full completions is a lot to scroll past to reach the one you want
            blocks.append(
                f'<details style="margin-top:6px"><summary style="cursor:pointer">'
                f'<span style="font-weight:600;color:{colour}">gen {gen}</span> '
                f'<span style="color:{colour}">{verdict}</span> '
                f'<span style="color:#6b7280">R={r["reward"]:.2f} &middot; '
                f"sum={cum[-1]:+.3g} &middot; {length} tokens</span></summary>"
                f'<div style="white-space:pre-wrap;font:12px ui-monospace,monospace;border:1px solid #e5e7eb;'
                f'border-radius:4px;padding:6px;margin-top:3px">{"".join(spans)}</div></details>'
            )
        return (
            '<div style="margin-top:12px;border-top:1px solid #e5e7eb;padding-top:8px">'
            '<button onclick="var c=this.dataset.m!==\'d\';this.dataset.m=c?\'d\':\'c\';'
            "this.textContent='colour: '+(c?'per-token':'cumsum');"
            'document.querySelectorAll(\'[data-c]\').forEach(function(s){s.style.background=c?s.dataset.d:s.dataset.c;});" '
            'style="cursor:pointer;font:11px ui-sans-serif,system-ui,sans-serif;padding:3px 8px">'
            "colour: cumsum</button>" + "".join(blocks) + "</div>"
        )

    def log_metrics(
        self,
        accelerator,
        metrics,
        value_loss,
        inputs,
        advantages,
        values,
        step,
    ):
        metrics["value_loss"].append(accelerator.gather(value_loss).nanmean().item())

        # Share of the reward's variance around the group baseline that the critic explains: 0 if it does no
        # better than predicting Rloo, 1 if it predicts R exactly, negative if it is worse than the baseline.
        # Both sums are gathered before dividing, since a ratio cannot be averaged across ranks afterwards.
        # Unanimous groups are dropped from both sums: their leave-one-out advantage is zero, so they add
        # nothing to the signal while still charging their error to the residual, which only drags the ratio down.
        rewards = inputs["rewards"].unsqueeze(-1)
        centered = rewards - inputs["rloo_baseline"].unsqueeze(-1)
        mask = inputs["completion_mask"] * (centered.abs() > 1e-8)
        residual = ((values[:, :-1].detach() - rewards) ** 2 * mask).sum()
        signal = (centered**2 * mask).sum()
        residual, signal = accelerator.gather(residual).sum(), accelerator.gather(signal).sum()
        if signal > 0:  # every group unanimous: the baseline already is the reward, so there is nothing to explain
            metrics["value_ev"].append((1.0 - residual / signal).item())

        for key, value in self._analysis_stats.items():
            metrics[key].append(value)
        for key, value in self._times.items():
            metrics[key].append(value)
        self._times.clear()
        self.log_privileged_example(accelerator, inputs)
        self.log_analysis_example(accelerator)
        self.log_advantage_plot(accelerator, inputs, advantages, step)
