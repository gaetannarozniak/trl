import json
import os
import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch
import wandb
from accelerate import PartialState
from transformers import AutoModelForTokenClassification, TrainerCallback


HELDOUT_PROBLEMS = 128  # first problems of the shuffled dataset, never trained on
POOL_PROBLEMS = 1024  # eligible problems after the held-out ones, among which the training problems are drawn


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
            "help": "If set, train only on N problems drawn with `--seed` among the first 1024 problems of the MC "
            "estimates after the 128 held-out ones, and also log the angle on their MC trajectories."
        },
    )
    value_mc_path: str | None = field(
        default=None,
        metadata={
            "help": "`.jsonl` from value_mc_estimate.py. If set, the angle between the MC and the value function "
            "advantages is logged at each evaluation, on the held-out problems and on the training problems."
        },
    )


def load_mc_estimates(path, args, config):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    if "index" not in rows[0]:
        raise ValueError(f"{path} was produced by an older value_mc_estimate.py without problem indices, rerun it.")
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
    # Flat MC values have no advantages, so no angle
    rows = [row for row in rows if len(set(row["values"])) > 1]
    pool = sorted({row["index"] for row in rows if row["index"] >= HELDOUT_PROBLEMS})[:POOL_PROBLEMS]
    num_problems = config.value_num_train_problems
    if num_problems is not None and num_problems > len(pool):
        errors.append(f"  value_num_train_problems is {num_problems}, but the MC estimates have {len(pool)} problems")
    if args.eval_strategy == "no":
        errors.append("  eval_strategy is 'no', so the MC metric would never be computed")
    if errors:
        raise ValueError(f"The MC estimates in {path} are incompatible with this run:\n" + "\n".join(errors))
    # A shuffle, so that for a given seed the problems of a smaller N are among those of a larger N
    random.Random(args.seed).shuffle(pool)
    train_problems = set(pool[:num_problems]) if num_problems is not None else set()
    splits = {
        "train": [row for row in rows if row["index"] in train_problems],
        "heldout": [row for row in rows if row["index"] < HELDOUT_PROBLEMS],
    }
    splits = {name: [add_mc_advantages(row) for row in rows] for name, rows in splits.items() if rows}
    return splits, mc_args["eval_interval"]


def add_mc_advantages(row):
    row["mc_advantages"] = torch.tensor(row["values"]).diff()
    # REINFORCE's advantage R - V0 is the same for every token, with the sign of 2R - 1 since 0 < V0 < 1
    reinforce_advantages = torch.full_like(row["mc_advantages"], 2 * row["reward"] - 1)
    row["reinforce_cos"] = torch.nn.functional.cosine_similarity(
        reinforce_advantages, row["mc_advantages"], dim=0
    ).item()
    return row


class ValueFunction(TrainerCallback):
    """
    Value model trained with the MSE between the predicted value of each completion prefix and the final reward of the
    completion (no discount). Its loss is added to the GRPO loss, and it is stepped with its own optimizer.
    """

    def __init__(self, config: ValueConfig, args):
        self.splits = {}  # MC trajectories of the training and held-out problems
        self.mc_step = -1
        if config.value_num_train_problems is not None and config.value_mc_path is None:
            raise ValueError("`--value_num_train_problems` needs `--value_mc_path`, the problems are taken from it.")
        if config.value_mc_path is not None:
            self.splits, self.mc_eval_interval = load_mc_estimates(config.value_mc_path, args, config)
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
        if mode == "eval" and self.splits and trainer.state.global_step != self.mc_step:
            self.mc_step = trainer.state.global_step
            names = ("angle", "reinforce_angle", "num_problems")
            for split, rows in self.splits.items():
                for name, value in zip(names, self.mc_angles(trainer, rows), strict=True):
                    trainer._metrics[mode][f"value/{split}_{name}"].append(value)
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

    def mc_angles(self, trainer, rows):
        # Mean over problems of the mean over their trajectories of the angles (degrees) between the MC advantages and
        # those of the value function, and of REINFORCE as a reference
        accelerator = trainer.accelerator
        problems = {index: position for position, index in enumerate(sorted({row["index"] for row in rows}))}
        # Per problem, sums of the 2 angles and number of trajectories
        stats = torch.zeros(len(problems), 3, device=accelerator.device)
        for row in rows[accelerator.process_index :: accelerator.num_processes]:
            value_advantages = self.mc_values(trainer, row)[:: self.mc_eval_interval].diff()
            mc_advantages = row["mc_advantages"].to(accelerator.device)
            cos = torch.nn.functional.cosine_similarity(value_advantages, mc_advantages, dim=0)
            cosines = torch.stack([cos, cos.new_tensor(row["reinforce_cos"])])
            stats[problems[row["index"]]] += torch.cat(
                [torch.rad2deg(torch.arccos(cosines.clamp(-1, 1))), cos.new_ones(1)]
            )
        stats = accelerator.reduce(stats, reduction="sum")
        return [*(stats[:, :2] / stats[:, 2:]).mean(0).tolist(), len(problems)]

    def log_mc_curves(self, trainer):
        # One image per evaluation under the same key, so the wandb panel gets a step slider
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
        rows = self.splits.get("train", self.splits.get("heldout"))
        for ax, row in zip(axes.flat, rows[:6], strict=False):
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
