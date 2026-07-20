import os
import re

from datasets import load_dataset
from transformers import AutoModelForCausalLM

from trl.experimental.opsft import OPSFTConfig, OPSFTTrainer


os.environ["WANDB_PROJECT"] = "opsft"  # log this run under the "opsft" wandb project

# GSM8K: `question` is the problem, `answer` is the worked solution ending in "#### <final number>".
dataset = load_dataset("openai/gsm8k", "main")  # has "train" and "test" splits

# The reference answers `y` end with "#### <number>", so on-policy SFT teaches the model this exact format. The prompt
# asks for it too, so generation matches the target from the start.
SYSTEM_PROMPT = (
    "Solve the math problem step by step. Show your reasoning, then give the final answer on its own line in the "
    "format '#### <number>'."
)


def _extract_hash_answer(text):
    # Number after the last "####" (GSM8K format), commas/$ stripped, as a float. `None` if absent/unparseable.
    if "####" not in text:
        return None
    match = re.search(r"-?[\d,]+(?:\.\d+)?", text.split("####")[-1])
    if match is None:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def final_answer_accuracy(completions, solution, **kwargs):
    # Reward 1.0 if the number after the model's final "####" matches the gold answer, else 0.0. Mirrors GSM8K's own
    # answer format (the `solution` column, forwarded by the trainer), which is what the model learns to produce.
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution, strict=True):
        pred = _extract_hash_answer(content)
        gold = _extract_hash_answer("#### " + sol)
        rewards.append(1.0 if pred is not None and gold is not None and pred == gold else 0.0)
    return rewards


def make_example(example):
    # `prompt`/`completion` are the (x, y) pair OPSFT needs: y (the reference answer) defines the SFT direction u, and
    # is also exactly what a plain-SFT baseline would train on. `solution` (the gold "#### <n>" number) is forwarded by
    # the trainer to `final_answer_accuracy`.
    answer = example["answer"]
    solution = answer.split("####")[-1].strip()
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "completion": [{"role": "assistant", "content": answer}],
        "solution": solution,
    }


dataset = dataset.map(make_example, remove_columns=["question", "answer"])

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", attn_implementation="eager")
trainer = OPSFTTrainer(
    model=model,
    # `final_answer_accuracy` is *not* used by the OPSFT loss; passing it only makes the trainer generate, score, and
    # log `rewards/final_answer_accuracy/mean` (train) and `eval_rewards/final_answer_accuracy/mean` (test) every
    # step — i.e. the current accuracy on both splits.
    reward_funcs=final_answer_accuracy,
    args=OPSFTConfig(
        output_dir="Qwen2.5-0.5B-OPSFT-gsm8k",
        gradient_checkpointing=False,
        bf16=True,
        max_completion_length=512,  # GSM8K chains of thought are long
        num_generations=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        eval_steps=50,
        eval_on_start=True,
        logging_steps=1,
        compute_eval_kl=True,  # log eval_kl vs. the frozen base θ₀ (catastrophic-forgetting probe)
        # use_vllm=True,  # optional; colocate by default
        report_to=["wandb"],
        log_completions=True,
        num_completions_to_print=1,
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"].select(range(128)),  # fixed 128-sample eval subset (faster, comparable across runs)
)
trainer.train()
