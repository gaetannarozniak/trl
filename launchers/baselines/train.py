# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for the Lean SFT baselines.

    accelerate launch launchers/baselines/train.py --config launchers/baselines/configs/sft.yaml
    accelerate launch launchers/baselines/train.py --config .../sft_kl.yaml --beta 0.2

Everything is driven by the YAML; any field of [`LeanSFTConfig`] can be overridden on the command
line, which is how the sbatch passes per-run values.
"""

import os
import sys
import time


# Allow `python launchers/baselines/train.py` as well as `-m`: under `accelerate launch`,
# `sys.path[0]` is this file's directory, not the repository root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LeanSFTConfig  # noqa: E402


def main(args: LeanSFTConfig, model_args) -> None:
    start = time.time()

    from accelerate import PartialState
    from callbacks import LeanEval, VibeCheck
    from datasets_registry import get_dataset
    from transformers import AutoTokenizer
    from verify import LeanVerifier

    from generation import EvalGenerator, restore_hf_call
    from trainer import LeanSFTTrainer

    state = PartialState()
    model_id = model_args.model_name_or_path
    args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype or "bfloat16",
        # `create_model_from_path` defaults to `device_map="auto"` whenever a GPU is visible
        # (trl/trainer/utils.py:1179), which on a 4-GPU node shards one model across all four:
        # early layers on cuda:0, final hidden states on cuda:3. Every part of this design wants
        # one complete model per rank instead -- DDP/ZeRO-2 replication, vLLM colocate weight
        # sync, and the chunked log-prob kernel, which indexes hidden states with a mask built on
        # the input's device.
        device_map=None,
    )

    # One rank builds the dataset cache, the rest read it: otherwise every rank redundantly reruns
    # filter+map and they race on the `datasets` fingerprint cache.
    with state.main_process_first():
        dataset = get_dataset(args.dataset_key, eval_fraction=args.eval_fraction, seed=args.dataset_seed)
    train_set, eval_set = dataset["train"], dataset["test"]
    if args.smoke:
        train_set = train_set.select(range(min(256, len(train_set))))
        eval_set = eval_set.select(range(min(16, len(eval_set))))
    if state.is_main_process:
        print(f"train: {len(train_set)}  eval: {len(eval_set)}  (setup {time.time() - start:.1f}s)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    trainer = LeanSFTTrainer(
        model=model_id,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        processing_class=tokenizer,
    )

    # Built after the trainer so the engine sees the prepared model, and shared by both callbacks:
    # a second colocated engine would double the GPU memory reservation.
    generator = EvalGenerator(
        model=trainer.model,
        accelerator=trainer.accelerator,
        processing_class=tokenizer,
        args=args,
    )

    # Building the engine patches the Hugging Face decoder class process-wide; undo it for our own
    # modules or the next forward pass dies on a missing `do_not_compile`. See `restore_hf_call`.
    patched = restore_hf_call(trainer.model, trainer.ref_model)
    if state.is_main_process and patched:
        print(f"[vllm] restored HF __call__ on {patched} modules", flush=True)

    callbacks = [VibeCheck(generator, eval_set.select(range(args.vibecheck_n))["prompt"], args)]
    if args.lean_workspace:
        lean_eval = LeanEval(
            generator=generator,
            verifier=LeanVerifier(args, rank=state.process_index),
            dataset=eval_set,
            args=args,
        )
        lean_eval.trainer = trainer  # so it can log to wandb
        callbacks.append(lean_eval)
    elif state.is_main_process:
        print("[lean] `lean_workspace` unset: skipping Lean verification", flush=True)

    for callback in callbacks:
        trainer.add_callback(callback)

    if state.is_main_process:
        print(f"training begins after {time.time() - start:.1f}s", flush=True)
    trainer.train()

    if args.save_final_model:
        destination = os.path.abspath(args.output_dir)
        if state.is_main_process:
            print(f"[save] saving the model in {destination}", flush=True)
        trainer.save_model()
        if state.is_main_process:
            print(f"[save] saved the model in {destination}", flush=True)


def make_parser():
    from trl import ModelConfig, TrlParser

    return TrlParser((LeanSFTConfig, ModelConfig))


if __name__ == "__main__":
    parser = make_parser()
    # Strict: an unrecognised key is an error, not a silent no-op. `warmup_ratio` was accepted and
    # ignored for several runs this way -- transformers 5.x only has `warmup_steps` -- so those
    # runs trained with no warmup while the YAML claimed otherwise. A misconfigured run that dies
    # in the first second is far cheaper than one that quietly trains the wrong thing for hours.
    training_args, model_args = parser.parse_args_and_config(fail_with_unknown_args=True)
    main(training_args, model_args)
