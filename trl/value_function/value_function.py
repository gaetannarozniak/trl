import json
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch
import wandb
from accelerate import PartialState
from transformers import AutoModelForTokenClassification, TrainerCallback


@dataclass
class ValueConfig:
    value_model_name_or_path: str = field(
        metadata={"help": "Checkpoint to initialize the value model from, usually a GRPO checkpoint."}
    )
    policy_model_name_or_path: str = field(
        metadata={"help": "Checkpoint of the frozen policy, usually the same GRPO checkpoint as the value model."}
    )
    value_learning_rate: float = field(default=3e-5, metadata={"help": "Learning rate of the value model."})
    value_max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm of the value model."})
    value_save_steps: int = field(default=200, metadata={"help": "Save the value model every this many steps."})
    value_num_train_problems: int | None = field(
        default=None,
        metadata={
            "help": "If set, train only on the problems of the first N MC trajectories that pass the filters, which are "
            "also the ones used for the angle and the plot."
        },
    )
    value_mc_path: str | None = field(
        default=None,
        metadata={
            "help": "`.jsonl` from value_mc_estimate.py. If set, the angle between the MC and the value function "
            "advantages is logged at each evaluation."
        },
    )


def load_mc_estimates(path, args, config, eval_prompts):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    if "mc_args" not in rows[0]:
        raise ValueError(f"{path} was produced by an older value_mc_estimate.py without `mc_args`, rerun it.")
    mc_args = rows[0]["mc_args"]
    policy = config.policy_model_name_or_path
    expected = {
        "model_id": os.path.realpath(policy) if os.path.isdir(policy) else policy,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p or 0.0,
        "repetition_penalty": args.repetition_penalty,
    }
    errors = [
        f"  {key}: the MC run used {mc_args[key]!r}, this run uses {value!r}"
        for key, value in expected.items()
        if mc_args[key] != value
    ]
    eval_prompts = set(eval_prompts)
    num_outside = sum(row["prompt"] not in eval_prompts for row in rows)
    if num_outside:
        errors.append(
            f"  {num_outside} of {len(rows)} MC prompts are not in the eval split, the value model may train on them"
        )
    # Skip almost always/never solved prompts, and flat MC values for which the angle is undefined
    rows = [row for row in rows if 0.1 <= row["values"][0] <= 0.9 and len(set(row["values"])) > 1]
    num_problems = config.value_num_train_problems
    if num_problems is not None and num_problems > len(rows):
        errors.append(
            f"  value_num_train_problems is {num_problems}, but only {len(rows)} MC trajectories pass the filters"
        )
    if args.eval_strategy == "no":
        errors.append("  eval_strategy is 'no', so the MC metric would never be computed")
    if errors:
        raise ValueError(f"The MC estimates in {path} are incompatible with this run:\n" + "\n".join(errors))
    return rows[:num_problems], mc_args["eval_interval"]


class ValueFunction(TrainerCallback):
    """
    Value model trained with the MSE between the predicted value of each completion prefix and the final reward of the
    completion (no discount). Its loss is added to the GRPO loss, and it is stepped with its own optimizer.
    """

    def __init__(self, config: ValueConfig, args, eval_prompts):
        self.mc_rows = None
        self.mc_step = -1
        if config.value_num_train_problems is not None and config.value_mc_path is None:
            raise ValueError("`--value_num_train_problems` needs `--value_mc_path`, the problems are taken from it.")
        if config.value_mc_path is not None:
            self.mc_rows, self.mc_eval_interval = load_mc_estimates(config.value_mc_path, args, config, eval_prompts)
        self.model, loading_info = AutoModelForTokenClassification.from_pretrained(
            config.value_model_name_or_path, num_labels=1, dtype=torch.float32, output_loading_info=True
        )
        # A new head (loaded from a policy checkpoint) starts at V = 0, so its first errors don't wreck the backbone
        if "score.weight" in loading_info["missing_keys"]:
            torch.nn.init.zeros_(self.model.score.weight)
            torch.nn.init.zeros_(self.model.score.bias)
        self.model.to(PartialState().device)
        self.model.dropout.p = 0.0
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.value_learning_rate)
        self.max_grad_norm = config.value_max_grad_norm
        self.save_steps = config.value_save_steps

    def compute_loss(self, trainer, inputs):
        prompt_length = inputs["prompt_ids"].size(1)
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        with trainer.accelerator.autocast():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        # values[:, t] is the value of the prompt followed by the first t completion tokens
        values = logits[:, prompt_length - 1 : -1, 0].float()
        mask = inputs["completion_mask"]
        loss = ((values - inputs["rewards"].unsqueeze(1)) ** 2 * mask).sum() / mask.sum()

        mode = "train" if trainer.model.training else "eval"
        trainer._metrics[mode]["value_loss"].append(trainer.accelerator.gather(loss.detach()).mean().item())
        if mode == "eval" and self.mc_rows is not None and trainer.state.global_step != self.mc_step:
            self.mc_step = trainer.state.global_step
            angle, num_trajectories = self.mc_advantage_angle(trainer)
            trainer._metrics[mode]["value/mc_advantage_angle"].append(angle)
            trainer._metrics[mode]["value/mc_num_trajectories"].append(num_trajectories)
            if trainer.accelerator.is_main_process and wandb.run is not None:
                self.log_mc_curves(trainer)
        return loss / (trainer.current_gradient_accumulation_steps if mode == "train" else 1)

    @torch.no_grad()
    def mc_values(self, trainer, row):
        # Value of the prompt followed by the first t completion tokens, for every t
        prompt_ids = row["prompt_token_ids"]
        input_ids = torch.tensor([prompt_ids + row["completion_token_ids"]], device=trainer.accelerator.device)
        with trainer.accelerator.autocast():
            logits = self.model(input_ids=input_ids).logits
        return logits[0, len(prompt_ids) - 1 : -1, 0].float()

    def mc_advantage_angle(self, trainer):
        # Mean over MC trajectories of the angle (degrees) between the MC and the value function advantages
        accelerator = trainer.accelerator
        stats = torch.zeros(2, device=accelerator.device)  # sum of angles, number of trajectories
        for row in self.mc_rows[accelerator.process_index :: accelerator.num_processes]:
            value_advantages = self.mc_values(trainer, row)[:: self.mc_eval_interval].diff()
            mc_advantages = torch.tensor(row["values"], device=accelerator.device).diff()
            cos = torch.nn.functional.cosine_similarity(value_advantages, mc_advantages, dim=0)
            stats += torch.stack([torch.rad2deg(torch.arccos(cos.clamp(-1, 1))), torch.ones_like(cos)])
        angle_sum, count = accelerator.reduce(stats, reduction="sum")
        return (angle_sum / count).item(), count.item()

    def log_mc_curves(self, trainer):
        # One image per evaluation under the same key, so the wandb panel gets a step slider
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
        for ax, row in zip(axes.flat, self.mc_rows[:6], strict=False):
            ax.plot(self.mc_values(trainer, row).cpu(), label="value function")
            mc_tokens = range(0, len(row["completion_token_ids"]), self.mc_eval_interval)
            ax.plot(mc_tokens, row["values"], "o-", label="Monte Carlo")
            ax.set_xlabel("completion token")
        axes[0, 0].set_ylabel("value")
        axes[0, 0].legend()
        fig.suptitle(f"step {trainer.state.global_step}")
        wandb.log({"value/mc_curves": wandb.Image(fig), "train/global_step": trainer.state.global_step})
        plt.close(fig)

    def on_optimizer_step(self, args, state, control, **kwargs):
        num_processes = PartialState().num_processes
        if num_processes > 1:
            for param in self.model.parameters():
                torch.distributed.all_reduce(param.grad)
                param.grad.div_(num_processes)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0 and state.is_world_process_zero:
            self.model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{state.global_step}", "value_model"))
