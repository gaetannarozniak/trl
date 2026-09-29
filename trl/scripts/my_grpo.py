import os

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.models.olmo3.modeling_olmo3 import Olmo3Model

from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.rewards import accuracy_reward
from trl.trainer.value_functions import ValueFunctionConfig


os.environ["WANDB_PROJECT"] = "value_function3"

# vLLM has no olmo3, so its Transformers backend decorates transformers' Olmo3Model in this very
# process; the policy and its critic copy predate the engine and miss the `do_not_compile` that the
# decorated __call__ reads. vLLM's own instances set it in __init__ and shadow this default.
Olmo3Model.do_not_compile = True


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
    parser.add_argument("--dataset", type=str, default="trl-lib/DeepMath-103K", help="names the dataset to use")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="policy to train; the critic backbone is a copy of it, e.g. `--model_name allenai/Olmo-3-7B-Think`",
    )
    parser.add_argument(
        "--overfit_sample",
        type=int,
        default=None,
        help="train on this single dataset row only, e.g. `--overfit_sample 75468`, to test overfitting",
    )
    parser.set_defaults(**DEFAULTS)
    config, value_function_config, script_args = parser.parse_args_and_config()

    config.run_name = script_args.expe_name
    # `|` separates the fields of a run name, which reads badly as a directory
    config.output_dir = os.path.join("trainer_output", script_args.expe_name.replace("|", "_"))

    # Problem 75468 of the train set is quite hard (1/16 correct answers for Qwen2.5-0.5B) and can be used for the only one problem overfitting configuration
    dataset = load_dataset(script_args.dataset, split="train")
    if "prompt" not in dataset.column_names:
        dataset = dataset.map(
            lambda row: {"prompt": [{"role": "user", "content": row["problem"]}], "solution": f"${row['answer']}$"},
            remove_columns=dataset.column_names,
        )
    if script_args.overfit_sample is not None:
        dataset = dataset.select([script_args.overfit_sample] * 100000)

    trainer = GRPOTrainer(
        model=script_args.model_name,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        args=config,
        # A multimodal checkpoint would otherwise get an AutoProcessor, whose batched
        # apply_chat_template cannot stack prompts of different lengths.
        processing_class=AutoTokenizer.from_pretrained(
            script_args.model_name, truncation_side="left", padding_side="left"
        ),
        value_function_config=value_function_config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
