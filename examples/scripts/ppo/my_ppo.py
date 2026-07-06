"""Train only the value function with PPO on GSM8K, using GSM8K answer-correctness as the reward.

The policy is frozen (`freeze_policy=True`), so this performs pure policy evaluation: the value head learns to
predict the expected (discounted) probability that the frozen policy ends up with the correct GSM8K answer.

The reward is programmatic (not a neural reward model): completions are decoded and the extracted final answer is
compared to the gold solution. It is plugged in through `PPOTrainer(reward_fn=...)`, with `reward_model=None`.

    python examples/scripts/ppo/my_ppo.py
"""

import re
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl.experimental.ppo import PPOConfig, PPOTrainer


MODEL_ID = "Qwen/Qwen3.5-2B"

SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, and the assistant briefly solves it. "
    "The final answer (only a number) must be on its own line in the format\n`#### <answer>`."
)

os.environ["WANDB_PROJECT"] = "value_function"



# --- GSM8K helpers (mirrors trl/experimental/sdpo/sdpo.py) ---------------------------------------------------------
def _normalize_gsm8k_answer(answer_text: str) -> str:
    if "####" not in answer_text:
        return answer_text.strip()
    return answer_text.split("####", 1)[1].strip().replace(",", "")


def _extract_predicted_answer(completion_text: str) -> str | None:
    match = re.search(r"####\s*([^\n]+)", completion_text)
    if match:
        return match.group(1).strip().replace(",", "")
    matches = re.findall(r"(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)", completion_text)
    if not matches:
        return None
    return matches[-1].replace("$", "").replace(",", "").strip()


def _gsm8k_accuracy_reward(completions, solution) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, solution, strict=True):
        pred = _extract_predicted_answer(completion)
        rewards.append(1.0 if pred is not None and pred == gold else 0.0)
    return rewards


# --- Model & tokenizer --------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

policy = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
ref_policy = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
# Value model: same backbone, fresh scalar value head (`num_labels=1`) trained from scratch. Kept in fp32 — it is the
# only model being optimized, and training a value head in pure bf16 (no fp32 master / loss scaling) diverges to NaN.
value_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=1, dtype=torch.float32)


# --- Dataset ------------------------------------------------------------------------------------------------------
def make_example(example):
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]
    text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return {"input_ids": input_ids, "solution": _normalize_gsm8k_answer(example["answer"])}


raw = load_dataset("openai/gsm8k", "main")
train_dataset = raw["train"].map(make_example, remove_columns=raw["train"].column_names)
eval_dataset = raw["test"].select(range(256)).map(make_example, remove_columns=raw["test"].column_names)


def collate(features):
    """Left-pad `input_ids` and carry the gold `solution` strings through to the rollout."""
    batch = tokenizer.pad([{"input_ids": f["input_ids"]} for f in features], padding=True, return_tensors="pt")
    batch["solution"] = [f["solution"] for f in features]
    return batch


# --- Reward: GSM8K answer correctness over the decoded completions -------------------------------------------------
def reward_fn(prompts, completions, batch):
    return _gsm8k_accuracy_reward(completions, solution=batch["solution"])


# --- Training -----------------------------------------------------------------------------------------------------
training_args = PPOConfig(
    output_dir="models/ppo-value-gsm8k",
    freeze_policy=True,  # only the value function is updated
    kl_coef=0.0,  # frozen policy → KL anchor is irrelevant
    learning_rate=3e-6,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_mini_batches=1,
    num_ppo_epochs=4,
    total_episodes=20000,
    response_length=256,
    temperature=1.0,
    stop_token="eos",
    report_to="wandb",
    eval_value_function_freq=25,
)

trainer = PPOTrainer(
    args=training_args,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_policy,
    reward_model=None,
    value_model=value_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate,
    reward_fn=reward_fn,
)
trainer.train()
trainer.save_model(training_args.output_dir)
