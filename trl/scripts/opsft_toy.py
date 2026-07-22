"""
Toy overfitting check for OPSFT: train on a *single* (prompt, completion) example, replicated N_COPIES times.

If OPSFT is actually learning to fit the reference y, its per-token reference log-likelihood — logged by the trainer
as `ref_logp` = mean log π(y | x) — should climb toward 0 (probability → 1). Watch `ref_logp` in the console dict /
wandb: it going to ~0 is the sanity check. `param_drift` should rise and `c_abs_mean` reflects the step scale.

Uses a small model so the double-backward-free jvp forward is cheap; swap the sample or model as you like.
"""

import os
from dataclasses import dataclass

from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM

from trl import TrlParser
from trl.experimental.opsft import OPSFTConfig, OPSFTTrainer


os.environ["WANDB_PROJECT"] = "opsft"


@dataclass
class ScriptArguments:
    # Script-level knobs; everything else is OPSFTConfig, overridable on the CLI or via `--config <yaml>`.
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    sdft_data: str = "/home/gaetan/projects/SDFT/data"
    n_copies: int = 1000  # times to replicate the single training example


script_args, training_args = TrlParser((ScriptArguments, OPSFTConfig)).parse_args_and_config()

MODEL = script_args.model_name
SDFT_DATA = script_args.sdft_data
N_COPIES = script_args.n_copies

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
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
