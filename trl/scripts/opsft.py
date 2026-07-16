import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import GRPOConfig
from trl.experimental.opsft import OPSFTTrainer


os.environ["WANDB_PROJECT"] = "opsft"  # log this run under the "opsft" wandb project

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", attn_implementation="eager")
dataset = load_dataset("trl-lib/tldr", split="train")  # prompt-only dataset
trainer = OPSFTTrainer(
    model=model,
    args=GRPOConfig(
        output_dir="Qwen2.5-0.5B-OPSFT",
        gradient_checkpointing=False,
        bf16=True,
        # use_vllm=True,  # optional; colocate by default
        report_to=["wandb"],
    ),
    train_dataset=dataset,
)
trainer.train()

