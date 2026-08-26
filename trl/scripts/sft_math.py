"""
HF_HUB_OFFLINE=1 accelerate launch \
  --config_file /home/gaetan/projects/trl/examples/accelerate_configs/deepspeed_zero2.yaml \
  scripts/sft_math.py
"""

import time
import os

t = time.time()
os.environ["WANDB_PROJECT"] = "opsft2"

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, TrainerCallback

print(f"Imports done after {time.time() - t}")

train_set = load_dataset("json", data_files="/checkpoint/scientific-reasoning/datasets/math/train.jsonl")
eval_set = load_dataset("json", data_files="/checkpoint/scientific-reasoning/datasets/math/test.jsonl")

def preprocess_function(example):
    return {
        "prompt": f"Problem: {example['problem']}\nSolution:",
        "completion": f" {example['rationale']}",
    }

train_set = train_set.map(preprocess_function, remove_columns=['problem', 'level', 'type', 'rationale'])["train"]
eval_set = eval_set.map(preprocess_function, remove_columns=['problem', 'level', 'type', 'rationale'])["train"]

def _generate(model, tok, prompts, max_new_tokens):
    texts = [f"Problem: {p}\nSolution:" for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


VIBE_QUESTIONS = [
    # answer: 4
    "What is $2+2$?",
    "What is the capital of France?",
    # MATH test set, Level 5, Prealgebra -- answer: 26\pi
    "The radius of a circle is $x+6$ and its diameter is $3x+5$. What is the circumference "
    "of the circle, in terms of $\\pi$? (Do not include the variable $x$ in your answer.)",
    "Can you show in Lean that if f comp g is surjective then f is surjective ?",
]

class VibeCheck(TrainerCallback):
    def __init__(self, tok, questions=VIBE_QUESTIONS, max_new_tokens=256):
        self.tok, self.questions, self.max_new_tokens = tok, questions, max_new_tokens

    def on_log(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return
        model.eval()
        answers = _generate(model, self.tok, self.questions, self.max_new_tokens)
        model.train()
        for q, a in zip(self.questions, answers):
            print(f"\n\033[1m[step {state.global_step}] Q:\033[0m {q}\n\033[1mA:\033[0m {a}")


config = SFTConfig(
    output_dir="/checkpoint/scientific-reasoning/gaetan/ckpt/opsft2/math_sft",
    logging_first_step=True,
    report_to="wandb",
    run_name="math_sft",
    project="opsft2",
    save_strategy="no",
    bf16=True,
)

trainer = SFTTrainer(
    model="allenai/Olmo-3-1025-7B",
    args=config,
    train_dataset=train_set,
    eval_dataset=eval_set,
    # processing_class=tok,
    callbacks=[VibeCheck(tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B"))],
)
        
print(f"Training begins after {time.time() - t} sec")
trainer.train()
