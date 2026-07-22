import os
from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoModelForCausalLM

from trl import TrlParser
from trl.experimental.opsft import OPSFTConfig, OPSFTTrainer
from trl.rewards import accuracy_reward  # math_verify-based; needs `pip install math_verify latex2sympy2_extended`


os.environ["WANDB_PROJECT"] = "opsft"  # log this run under the "opsft" wandb project


@dataclass
class ScriptArguments:
    # Script-level knobs; everything else is OPSFTConfig, overridable on the CLI or via `--config <yaml>`.
    model_name: str = "Qwen/Qwen3-4B"
    eval_size: int = 64  # fixed test subset for eval accuracy (comparable across runs)


script_args, training_args = TrlParser((ScriptArguments, OPSFTConfig)).parse_args_and_config()

# GSM8K: `question` is the problem, `answer` is the worked solution ending in "#### <final number>".
dataset = load_dataset("openai/gsm8k", "main")  # has "train" and "test" splits

SYSTEM_PROMPT = (
    "Solve the math problem step by step. Show your reasoning, then give the final answer inside \\boxed{}, "
    "e.g. \\boxed{42}."
)


def make_example(example):
    # `prompt`/`completion` are the (x, y) pair OPSFT needs: y (the reference answer) defines the SFT direction u, and
    # is also exactly what a plain-SFT baseline would train on. We rewrite GSM8K's "#### <n>" ending into a \boxed{<n>}
    # final answer so on-policy SFT teaches the boxed format that `accuracy_reward` (math_verify) parses. `solution`
    # (the gold number) is forwarded by the trainer to `accuracy_reward`.
    answer = example["answer"]
    solution = answer.split("####")[-1].strip()
    reasoning = answer.split("####")[0].strip()
    completion = f"{reasoning}\nThe final answer is \\boxed{{{solution}}}"
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"] + "\nGive the final answer inside \\boxed{}."},
        ],
        "completion": [{"role": "assistant", "content": completion}],
        "solution": solution,
    }


dataset = dataset.map(make_example, remove_columns=["question", "answer"])

model = AutoModelForCausalLM.from_pretrained(script_args.model_name, attn_implementation="eager")
trainer = OPSFTTrainer(
    model=model,
    # `accuracy_reward` is *not* used by the OPSFT loss; passing it only makes the trainer generate, score, and log
    # `rewards/accuracy_reward/mean` (train) and `eval_rewards/accuracy_reward/mean` (test) every step — i.e. the
    # current accuracy on both splits.
    reward_funcs=accuracy_reward,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"].select(range(script_args.eval_size)),
)
trainer.train()
