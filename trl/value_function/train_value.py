"""
CKPT=/checkpoint/fort/gaetan/projects/value_function4/first_run_qwen4b_ckpts/checkpoint-300
sbatch --job-name=value-ckpt300 --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=120 \
--time=48:00:00 --partition=g3 --account=scientific-reasoning --qos=g3_scientific-reasoning_high \
--output=logs/%x-%j.out \
--wrap="accelerate launch --num_processes 4 trl/value_function/train_value.py --run_name value_ckpt300 \
    --policy_model_name_or_path $CKPT --value_model_name_or_path $CKPT"
"""

import os

import datasets

from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.value_function.reward_function import reward_function
from trl.value_function.value_function import ValueConfig, ValueFunction


def accuracy_reward(completions: list[str], label: list[str], **kwargs) -> list[float]:
    return [reward_function(completion, gt) for completion, gt in zip(completions, label, strict=True)]


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "value_function4"

    # Same defaults as train_grpo.py, but with a frozen policy (learning_rate=0)
    parser = TrlParser((GRPOConfig, ValueConfig))
    parser.set_defaults_with_config(
        output_dir="/checkpoint/fort/gaetan/projects/value_function4",
        report_to="wandb",
        use_vllm=True,
        learning_rate=0.0,
        gradient_checkpointing=False,
        lr_scheduler_type="constant",
        num_train_epochs=100,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        max_completion_length=2048,
        logging_steps=1,
        log_completions=False,
        num_completions_to_print=1,
        eval_strategy="steps",
        eval_steps=10,
        eval_on_start=True,
        per_device_eval_batch_size=16,
        num_generations_eval=1,
        save_strategy="no",
    )
    training_args, value_args = parser.parse_args_and_config()
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)

    # Same shuffle as value_mc_estimate.py, so its first `--max_prompts` (<= 256) prompts are the eval split
    dataset = datasets.load_dataset("zhuzilin/dapo-math-17k", split="train").shuffle(seed=0)
    dataset = dataset.map(lambda example: {"prompt": example["prompt"][0]["content"]})
    eval_dataset = dataset.select(range(256))
    train_dataset = dataset.select(range(256, len(dataset)))

    value_function = ValueFunction(value_args, training_args, eval_dataset["prompt"])
    if value_args.value_num_train_problems is not None:
        # Train on the problems of the MC trajectories of the angle, repeated since GRPO only samples full batches
        rows = value_function.mc_rows
        train_dataset = datasets.Dataset.from_dict(
            {"prompt": [row["prompt"] for row in rows], "label": [row["ground_truth"] for row in rows]}
        ).repeat(len(train_dataset) // len(rows))
    trainer = GRPOTrainer(
        model=value_args.policy_model_name_or_path,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[value_function],
        value_function=value_function,
    )
    # Freeze the policy so it needs no gradients nor optimizer states; DDP requires one trainable parameter
    for name, param in trainer.model.named_parameters():
        param.requires_grad = name == "model.norm.weight"
    trainer.train()
