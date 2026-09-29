import os

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.rewards import accuracy_reward
from trl.trainer.value_functions import ValueFunctionConfig


os.environ["WANDB_PROJECT"] = "value_function3"

# Defaults for this experiment. Any of them, and any other GRPOConfig or ValueFunctionConfig field, can be
# overridden from the command line, e.g. `--learning_rate 1e-5 --critic_lr 1e-4 --target plain`.
# `scale_rewards` is "none" rather than False: argparse would hand the dataclass the string "False", which
# GRPOConfig does not map to "none" and which fails validation at the first generation step.
DEFAULTS = {
    "report_to": ["wandb"],
    "logging_first_step": True,
    "logging_steps": 1,
    "per_device_train_batch_size": 32,
    "lr_scheduler_type": "constant",
    "save_strategy": "no",
    "use_vllm": True,
    "bf16": True,
    "max_completion_length": 2048,
    "num_generations": 16,
    "scale_rewards": "none",
}


def main():
    parser = TrlParser(dataclass_types=[GRPOConfig, ValueFunctionConfig])
    parser.add_argument("--expe_name", type=str, required=True, help="names the wandb run and its output_dir")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="policy to train; the critic backbone is a copy of it, e.g. `--model_name allenai/Olmo-3-7B-Think`",
    )
    parser.set_defaults(**DEFAULTS)
    config, value_function_config, script_args = parser.parse_args_and_config()

    config.run_name = script_args.expe_name
    # `|` separates the fields of a run name, which reads badly as a directory
    config.output_dir = os.path.join("trainer_output", script_args.expe_name.replace("|", "_"))

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    trainer = GRPOTrainer(
        model=script_args.model_name,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        args=config,
        value_function_config=value_function_config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
