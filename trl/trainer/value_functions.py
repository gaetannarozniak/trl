import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel


@dataclass(frozen=True)
class ValueFunctionConfig:
    zero_init_value_head: bool = True
    value_lambda: float | None = None
    adv_lambda: float | None = None
    critic_lr: float = 1e-5
    target: str = "residual"  # "residual" for R - Rloo, "plain" for R


# maybe still need to deal with the vllm iter_name_params for the critic, it seems redundant in the way we find the critic params (already two ways)
# remove scale_rewards for the grpo trainer
class ValueFunction:
    def __init__(self, args: ValueFunctionConfig):
        self.args = args

    def get_critic(self, policy_net: PreTrainedModel) -> nn.ModuleDict:
        head_in = policy_net.config.hidden_size
        critic = nn.ModuleDict(
            {
                "backbone": copy.deepcopy(policy_net.base_model),
                "head": nn.Sequential(
                    nn.Linear(head_in, 1, dtype=policy_net.dtype),
                    nn.Tanh(),
                ),
            }
        )
        if self.args.zero_init_value_head:
            nn.init.zeros_(critic["head"][-2].weight)
            nn.init.zeros_(critic["head"][-2].bias)
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

    def get_values(
        self,
        critic: nn.ModuleDict,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
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

    def log_metrics(self, metrics, inputs, advantages, values, value_loss):
        raise NotImplementedError
