import os

import datasets

from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.value_function.reward_function import reward_function


def accuracy_reward(completions: list[str], label: list[str], **kwargs) -> list[float]:
    return [reward_function(completion, gt) for completion, gt in zip(completions, label, strict=True)]


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "value_function4"

    # Defaults of this experiment: any GRPOConfig field can be overridden from the CLI, e.g. `--learning_rate 2e-6`
    parser = TrlParser(GRPOConfig)
    parser.set_defaults_with_config(
        output_dir="/checkpoint/fort/gaetan/projects/value_function4",
        report_to="wandb",
        use_vllm=True,
        lr_scheduler_type="constant",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        max_completion_length=2048,
        use_adaptive_entropy=True,
        entropy_target=0.3,
        logging_steps=1,
        log_completions=False,
        num_completions_to_print=1,
        eval_strategy="steps",
        eval_steps=25,
        eval_on_start=False,
        per_device_eval_batch_size=32,
        num_generations_eval=1,
        save_steps=50,
        save_only_model=True,
    )
    (training_args,) = parser.parse_args_and_config()
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)

    # Same shuffle as value_mc_estimate.py, so its first `--max_prompts` (<= 256) prompts are the eval split
    dataset = datasets.load_dataset("zhuzilin/dapo-math-17k", split="train").shuffle(seed=0)
    # Plain-text prompts: GRPOTrainer only applies the chat template to conversational prompts
    dataset = dataset.map(lambda example: {"prompt": example["prompt"][0]["content"]})
    eval_dataset = dataset.select(range(256))
    train_dataset = dataset.select(range(256, len(dataset)))

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-4B-Base",
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
