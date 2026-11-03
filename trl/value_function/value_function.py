import os
from dataclasses import dataclass, field

import torch
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
    value_learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate of the value model."})
    value_max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm of the value model."})


class ValueFunction(TrainerCallback):
    """
    Value model trained with the MSE between the predicted value of each completion prefix and the final reward of the
    completion (no discount). Its loss is added to the GRPO loss, and it is stepped with its own optimizer.
    """

    def __init__(self, config: ValueConfig):
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.value_model_name_or_path, num_labels=1, dtype=torch.float32
        ).to(PartialState().device)
        self.model.dropout.p = 0.0
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.value_learning_rate)
        self.max_grad_norm = config.value_max_grad_norm

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
        return loss / (trainer.current_gradient_accumulation_steps if mode == "train" else 1)

    def on_optimizer_step(self, args, state, control, **kwargs):
        num_processes = PartialState().num_processes
        if num_processes > 1:
            for param in self.model.parameters():
                torch.distributed.all_reduce(param.grad)
                param.grad.div_(num_processes)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{state.global_step}", "value_model"))
