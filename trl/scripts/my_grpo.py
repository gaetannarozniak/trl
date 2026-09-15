import os

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.rewards import accuracy_reward


os.environ["WANDB_PROJECT"] = "value_function2"

# Defaults for this experiment. Any of them, and any other GRPOConfig field, can be overridden from the
# command line, e.g. `--learning-rate 1e-5 --num-train-epochs 2`.
DEFAULTS = {
    "report_to": ["wandb"],
    "logging_first_step": True,
    "logging_steps": 1,
    "per_device_train_batch_size": 32,
    "lr_scheduler_type": "constant",
    "save_strategy": "no",
    "use_vllm": True,
    "bf16": True,
    "only_train_value_function": True,
}


def main():
    parser = TrlParser(dataclass_types=[GRPOConfig])
    parser.set_defaults(**DEFAULTS)
    (config,) = parser.parse_args_and_config()

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        args=config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
