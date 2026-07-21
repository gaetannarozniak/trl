"""
Toy overfitting check for OPSFT: train on a *single* (prompt, completion) example, replicated N_COPIES times.

If OPSFT is actually learning to fit the reference y, its per-token reference log-likelihood — logged by the trainer
as `ref_logp` = mean log π(y | x) — should climb toward 0 (probability → 1). Watch `ref_logp` in the console dict /
wandb: it going to ~0 is the sanity check. `param_drift` should rise and `c_abs_mean` reflects the step scale.

Uses a small model so the double-backward-free jvp forward is cheap; swap the sample or model as you like.
"""

import os

from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM

from trl.experimental.opsft import OPSFTConfig, OPSFTTrainer


os.environ["WANDB_PROJECT"] = "opsft"

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SDFT_DATA = "/home/gaetan/projects/SDFT/data"
N_COPIES = 1000

# Take one science example and turn it into the (x, y) pair OPSFT needs, then replicate it N_COPIES times so every
# batch is the same example — the model should quickly memorize it.
src = load_from_disk(f"{SDFT_DATA}/science_data/train_data")[0]
example = {
    "prompt": src["messages"],
    "completion": [{"role": "assistant", "content": src["output_text"]}],
}
train_dataset = Dataset.from_list([example] * N_COPIES)

model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
trainer = OPSFTTrainer(
    model=model,
    args=OPSFTConfig(
        output_dir="checkpoints/Qwen2.5-0.5B-OPSFT-toy",
        gradient_checkpointing=False,
        bf16=True,
        max_completion_length=256,
        num_generations=2,
        per_device_train_batch_size=8,
        num_train_epochs=20,
        logging_steps=1,
        learning_rate=1e-6,  # OPSFT's F·u step scale is unusual; crank this up if `ref_logp` barely moves
        eval_strategy="no",
        report_to=["wandb"],
    ),
    train_dataset=train_dataset,
)
trainer.train()
