from datasets import load_dataset

from trl import TrlParser
from trl.experimental.fd import FDConfig, FDTrainer
from trl.experimental.fd.scripts.utils import (
    _gsm8k_accuracy_reward,
    _normalize_gsm8k_answer,
    france_keyword_reward,
    french_ratio_reward,
)

print("Finished imports")

SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, and the assistant briefly solves it."
    "The final answer (only a number) must be on its own line in the format\n"
    "`#### <answer>`."
)

SHUFFLE_SEED = 42
EVAL_SIZE = 32


def _make_example(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "solution": _normalize_gsm8k_answer(example["answer"]),
    }


raw = load_dataset("openai/gsm8k", "main")
train_dataset = raw["train"].map(_make_example, remove_columns=raw["train"].column_names)
eval_dataset = (
    raw["test"]
    .shuffle(seed=SHUFFLE_SEED)
    .select(range(EVAL_SIZE))
    .map(_make_example, remove_columns=raw["test"].column_names)
)

privileged_context = "\nAnswer in French."
train_dataset = train_dataset.map(lambda x: {"privileged_context": privileged_context})
eval_dataset = eval_dataset.map(lambda x: {"privileged_context": privileged_context})

parser = TrlParser(FDConfig)
training_args, = parser.parse_args_and_config()

trainer = FDTrainer(
    model="Qwen/Qwen3.5-2B",
    reward_funcs=[french_ratio_reward, france_keyword_reward, _gsm8k_accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
