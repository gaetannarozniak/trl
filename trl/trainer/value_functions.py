import copy
from dataclasses import dataclass
from html import escape

import torch
import torch.nn as nn
from transformers import PreTrainedModel, is_wandb_available

from .utils import pad


if is_wandb_available():
    import wandb


@dataclass(frozen=True)
class ValueFunctionConfig:
    zero_init_value_head: bool = True
    value_lambda: float | None = None
    adv_lambda: float | None = None
    critic_lr: float = 1e-5
    critic_layer: int | None = None  # read the backbone after this block instead of the last one
    privileged_template: str | None = None  # extra user turn shown only to the critic, e.g. "Solution:\n{solution}"
    probe_token: str | None = None  # interleave this token after every completion token and read the value there
    target: str = (
        "residual"  # "residual" for R - Rloo, "plain" for R, "grpo" for linear interpolation between Rloo and R
    )


# maybe still need to deal with the vllm iter_name_params for the critic, it seems redundant in the way we find the critic params (already two ways)
# remove scale_rewards for the grpo trainer
class ValueFunction:
    def __init__(self, args: ValueFunctionConfig):
        self.args = args
        self._last_plot_step = -1
        self._processing_class = None
        self._logged_privileged = False
        self._probe_id = None

    def extra_inputs(self, rows, processing_class) -> dict[str, torch.Tensor]:
        """Privileged prompts, tokenised and merged into the batch so `get_values` can read them back."""
        self._processing_class = processing_class
        if self.args.probe_token is not None and self._probe_id is None:
            ids = processing_class.encode(self.args.probe_token, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"probe_token must be a single token, {self.args.probe_token!r} is {len(ids)}")
            self._probe_id = ids[0]
        if self.args.privileged_template is None:
            return {}
        conversation = [
            row["prompt"] + [{"role": "user", "content": self.args.privileged_template.format(**row)}]
            for row in rows
        ]
        ids = processing_class.apply_chat_template(
            conversation=conversation, add_generation_prompt=True, tokenize=True, return_dict=True
        )["input_ids"]
        ids = [torch.tensor(i) for i in ids]
        return {
            "critic_prompt_ids": pad(ids, padding_value=processing_class.pad_token_id, padding_side="left"),
            "critic_prompt_mask": pad([torch.ones_like(i) for i in ids], padding_value=0, padding_side="left"),
        }

    def get_critic(self, policy_net: PreTrainedModel) -> nn.ModuleDict:
        head_in = policy_net.config.hidden_size
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
        critic, policy = self._critic_params(model, accelerator)
        accelerator.clip_grad_norm_(critic, grad_norm)
        return accelerator.clip_grad_norm_(policy, grad_norm)

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

    def get_values(
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
        values = critic["head"](hidden).squeeze(-1)
        if self.args.target == "residual":
            values = values + inputs["rloo_baseline"].unsqueeze(-1)
        return values  # (B, logits_to_keep + 1)

    def get_value_loss(
        self, values: torch.Tensor, inputs: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        target = inputs["rewards"].unsqueeze(-1)
        if self.args.value_lambda is not None:
            raise NotImplementedError

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
    def log_advantage_plot(self, accelerator, inputs, advantages, step, every=25):
        """Log an HTML chart of the cumulative advantage along the completions of one prompt.

        The prompt is picked among those holding both a correct and an incorrect rollout, since the point is to
        compare how credit is spread in either case. Blue is a correct rollout, vermillion an incorrect one
        (Okabe-Ito, which stays separable under colour-vision deficiency). Emitted as inline SVG rather than
        through a plotting library, so it needs no dependency beyond wandb, and logged without an explicit
        `step=`: wandb's own counter runs ahead of `global_step` and anything below it is silently discarded.
        """
        if step % every or step == self._last_plot_step:
            return
        if not (accelerator.is_main_process and is_wandb_available() and wandb.run is not None):
            return
        self._last_plot_step = step

        mask = inputs["completion_mask"]
        correct = inputs["rewards"] > 0
        # `_prepare_inputs` shuffles the generation batch, so recover the groups from the prompts themselves
        groups = {}
        for row, prompt in enumerate(inputs["prompt_ids"].tolist()):
            groups.setdefault(tuple(prompt), []).append(row)
        mixed = [rows for rows in groups.values() if 0 < int(correct[rows].sum()) < len(rows)]
        rows = (mixed or list(groups.values()))[0]

        curves = []
        for row in rows:
            length = int(mask[row].sum())
            if length == 0:
                continue
            stride = max(1, length // 400)  # keeps the payload small on long completions
            cumulative = advantages[row, :length].float().cpu().cumsum(0)
            curves.append((list(range(0, length, stride)), cumulative[::stride].tolist(), bool(correct[row])))
        if not curves:
            return

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
        for xs, ys, ok in curves:
            color, label = ("#0072B2", "correct") if ok else ("#D55E00", "incorrect")
            path = " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(xs, ys))
            svg.append(
                f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" opacity="0.55">'
                f"<title>{label}</title></polyline>"
            )
        mid = (top + height - bottom) / 2
        svg.append(
            f'<text x="{left}" y="18" font-size="13" fill="#111827">step {step} - {len(curves)} generations - '
            f'mean reward {inputs["rewards"][rows].mean().item():.2f}</text>'
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
            f'{"".join(svg)}<div style="padding-top:6px;color:#374151">{legend}</div></div>'
        )
        wandb.log({"advantages/cumsum": wandb.Html(html, inject=False)})

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

        self.log_privileged_example(accelerator, inputs)
        self.log_advantage_plot(accelerator, inputs, advantages, step)
