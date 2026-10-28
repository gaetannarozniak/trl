import os

from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
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
    "num_train_epochs": 20,
}


class DropEmbeddingGradHook(TrainerCallback):
    """Undo the hook `gradient_checkpointing_enable` puts on the input embeddings.

    It forces the embedding output to require grad so that gradients reach LoRA layers under a frozen base
    model. With the whole policy frozen that only rebuilds a graph nothing consumes, which costs the saved
    activations and a backward pass. It is installed inside `train()`, hence a callback rather than a line
    next to the freeze below.
    """

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        model.disable_input_require_grads()


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
    parser.add_argument(
        "--eos_token",
        type=str,
        default=None,
        help="override the tokenizer's eos, e.g. `--eos_token '<|im_end|>'` for a base model prompted in ChatML",
    )
    parser.set_defaults(**DEFAULTS)
    config, value_function_config, script_args = parser.parse_args_and_config()

    config.run_name = script_args.expe_name
    # `|` separates the fields of a run name, which reads badly as a directory
    config.output_dir = os.path.join("trainer_output", script_args.expe_name.replace("|", "_"))

    # Problem 75468 of the train set is quite hard (1/16 correct answers for Qwen2.5-0.5B) and can be used for the only one problem overfitting configuration
    dataset = load_dataset(script_args.dataset, split="train")
    if "prompt" not in dataset.column_names:

        def to_trl_format(row):
            mapped = {"prompt": [{"role": "user", "content": row["problem"]}], "solution": f"${row['answer']}$"}
            if value_function_config.example_proof:
                # extra_inputs formats the privileged template with the row, so any column is a template field
                mapped["example_proof"] = row["solutions"][0]
            return mapped

        dataset = dataset.map(to_trl_format, remove_columns=dataset.column_names)
    if script_args.overfit_sample is not None:
        dataset = dataset.select([script_args.overfit_sample] * 100000)

    # A multimodal checkpoint would otherwise get an AutoProcessor, whose batched
    # apply_chat_template cannot stack prompts of different lengths.
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, truncation_side="left", padding_side="left")
    if script_args.eos_token is not None:
        tokenizer.eos_token = script_args.eos_token

    trainer = GRPOTrainer(
        model=script_args.model_name,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        args=config,
        processing_class=tokenizer,
        value_function_config=value_function_config,
    )

    # At lr 0 the policy never moves, so it needs no gradients, no Adam states, no saved activations and no
    # backward: freezing it leaves only the critic in the graph.
    if config.learning_rate == 0:
        for name, param in trainer.model.named_parameters():
            if not name.startswith("critic."):
                param.requires_grad_(False)
        trainer.add_callback(DropEmbeddingGradHook())

    trainer.train()


if __name__ == "__main__":
    main()
