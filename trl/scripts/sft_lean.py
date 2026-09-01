"""
HF_HUB_OFFLINE=1 accelerate launch \
  --config_file /home/gaetan/projects/trl/examples/accelerate_configs/deepspeed_zero2.yaml \
  scripts/sft_lean.py --trainer sft      # or: --trainer opsft
                      [--lr 1e-6] [--expe-name my_run]
"""

import re
import time
import os
import argparse

t = time.time()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--trainer",
    choices=["sft", "opsft"],
    default="sft",
    help="sft: teacher-forced on the gold proofs. opsft: on-policy, generates its own completions.",
)
parser.add_argument(
    "--learning-rate",
    "--lr",
    type=float,
    default=None,
    help="Left to each config's own default when omitted (they differ between sft and opsft).",
)
parser.add_argument(
    "--expe-name",
    default=None,
    help="wandb run name and checkpoint directory. Defaults to lean_<trainer>.",
)
cli = parser.parse_args()
EXPE = cli.expe_name or f"lean_{cli.trainer}"

os.environ["WANDB_PROJECT"] = "opsft2"

import torch
import torch.distributed as dist
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, OPSFTTrainer, OPSFTConfig
from transformers import AutoTokenizer, TrainerCallback

print(f"Imports done after {time.time() - t}")

MODEL = "allenai/Olmo-3-1025-7B"

ds = load_dataset("AI-MO/NuminaMath-LEAN")["train"]
# Only human-written proofs that are actually complete: autoformalizer statements have no
# proof at all, "statement" is a bare sorry, and "with_sorry" has holes in it.
ds = ds.filter(lambda e: e["author"] == "human" and e["ground_truth_type"] == "complete")

DOCSTRING = re.compile(r"/-.*?-/", re.S)


def preprocess_function(example):
    # Split formal_ground_truth (not formal_statement) at the first ":= by": the header there
    # matches the proof body verbatim, so prompt + completion is a compilable Lean file.
    body = DOCSTRING.sub("", example["formal_ground_truth"])
    m = re.search(r"^\s*(theorem|lemma|example)\b", body, re.M)
    j = body.find(":= by", m.start()) if m else -1
    if j < 0:
        return {"prompt": None, "completion": None}
    # Keep everything above the theorem too: 36% of rows carry `open Real`, `open Finset`,
    # `noncomputable section`, ... and the proof does not elaborate without them.
    return {
        "prompt": body[: j + len(":= by")].strip(),
        "completion": body[j + len(":= by") :],
    }


ds = ds.map(preprocess_function, remove_columns=ds.column_names)
ds = ds.filter(lambda e: e["prompt"] is not None and e["completion"].strip())
split = ds.train_test_split(test_size=0.05, seed=0)
train_set, eval_set = split["train"], split["test"]
print(f"train: {len(train_set)}  eval: {len(eval_set)}")


def _generate(model, tok, prompts, max_new_tokens):
    enc = tok(list(prompts), return_tensors="pt", padding=True, padding_side="left").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id, temperature=1.0)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)


# Held-out theorems, so the vibe check is never something the model trained on.
VIBE_QUESTIONS = eval_set.select(range(2))["prompt"]


class VibeCheck(TrainerCallback):
    def __init__(self, tok, questions=VIBE_QUESTIONS, max_new_tokens=512):
        self.tok, self.questions, self.max_new_tokens = tok, questions, max_new_tokens

    def on_log(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return
        model.eval()
        answers = _generate(model, self.tok, self.questions, self.max_new_tokens)
        model.train()
        for q, a in zip(self.questions, answers):
            print(f"\n\033[1m[step {state.global_step}] THEOREM:\033[0m\n{q}\n\033[1mPROOF:\033[0m{a}")

# Mathlib v4.15.0, matching the version NuminaMath-LEAN was verified against. Calibration:
# 100/100 human gold proofs compile here, vs 64/100 on the v4.25.2 workspace.
WS = "/storage/home/gaetan/lean/workspace-v4.15"

IMPORT_LINE = re.compile(r"^\s*import\b.*$", re.M)


class LeanEval(TrainerCallback):
    """Generate proofs for held-out theorems and check them with Lean.

    Generation is sharded across ranks (one full model per GPU under ZeRO-2) and each rank
    verifies its own shard, so nothing sits idle waiting on rank 0 -- which also keeps every
    rank inside the NCCL collective timeout.
    """

    def __init__(self, tok, data, n=64, gen_batch_size=4, max_new_tokens=1024, num_workers=8, timeout=120):
        self.tok = tok
        self.data = data.select(range(min(n, len(data))))
        self.gen_batch_size, self.max_new_tokens = gen_batch_size, max_new_tokens
        self.num_workers, self.timeout = num_workers, timeout
        self.pool, self.env = None, None
        self.trainer = None

    def _get_pool(self, rank):
        # Built lazily on first eval: each worker loads Mathlib (~20 s), so we pay this once
        # and reuse the pool for every later eval.
        if self.pool is None:
            from lean_interact import Command, LeanREPLConfig, LocalProject
            from lean_interact.pool import LeanServerPool

            t0 = time.time()

            def make_cfg():
                return LeanREPLConfig(
                    project=LocalProject(directory=WS),
                    local_repl_path=f"{WS}/.lake/packages/REPL",  # capitalised at the v4.15 tag
                    lake_path="/home/gaetan/.elan/bin/lake",
                    build_repl=False,
                )

            # Constructing the config runs `lake exe cache get`, which unpacks into the shared
            # ~/.cache/mathlib. Eight ranks doing that at once corrupt each other's .ltar files
            # and the REPL then dies with BrokenPipeError, so rank 0 goes first and the rest
            # attach to the finished project.
            if rank == 0:
                cfg = make_cfg()
            if dist.is_initialized():
                dist.barrier()
            if rank != 0:
                cfg = make_cfg()

            self.pool = LeanServerPool(cfg, num_workers=self.num_workers)
            # One session-cached `import Mathlib`; workers replay it on first use, then cache it.
            self.env = self.pool.run(Command(cmd="import Mathlib")).env
            if rank == 0:
                print(f"[lean] pool ready: {self.num_workers} workers/rank, env={self.env} ({time.time()-t0:.0f}s)")
        return self.pool

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        from lean_interact import Command

        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        shard = self.data.shard(num_shards=world, index=rank) if world > 1 else self.data

        if rank == 0:
            print(f"\n[lean] step {state.global_step}: {len(self.data)} theorems over {world} rank(s)", flush=True)

        model.eval()
        t0 = time.time()
        completions = []
        for i in range(0, len(shard), self.gen_batch_size):
            completions += _generate(
                model, self.tok, shard[i : i + self.gen_batch_size]["prompt"], self.max_new_tokens
            )
        model.train()
        t_gen = time.time() - t0
        n_new = sum(len(self.tok(c)["input_ids"]) for c in completions)

        # Mathlib is already in the session env, so drop the `import` lines but keep the
        # preamble (`open Real`, `noncomputable section`, ...) the proof needs to elaborate.
        reqs = [
            Command(cmd=IMPORT_LINE.sub("", p + c).strip(), env=self.env)
            for p, c in zip(shard["prompt"], completions)
        ]
        pool = self._get_pool(rank)
        t1 = time.time()
        results = pool.run_batch(reqs, timeout_per_cmd=self.timeout)
        t_check = time.time() - t1

        # ok / error (Lean rejected it) / sorry (has holes) / dead (REPL error, crash, timeout)
        buckets, first_errors = {"ok": 0, "error": 0, "sorry": 0, "dead": 0}, []
        for r in results:
            if not hasattr(r, "env"):
                buckets["dead"] += 1
                continue
            errs = [m.data for m in (getattr(r, "messages", None) or []) if getattr(m, "severity", "") == "error"]
            if errs:
                buckets["error"] += 1
                first_errors.append(errs[0])
            elif getattr(r, "sorries", None):
                buckets["sorry"] += 1
            else:
                buckets["ok"] += 1

        counts = torch.tensor(
            [buckets["ok"], buckets["error"], buckets["sorry"], buckets["dead"], len(reqs)],
            dtype=torch.long,
            device=model.device,
        )
        if world > 1:
            dist.all_reduce(counts)
        ok, err, sry, dead, total = counts.tolist()
        rate = ok / max(total, 1)

        if rank == 0:
            print(
                f"[lean] generate {t_gen:6.1f}s  ({n_new/max(t_gen,1e-9):5.0f} tok/s/rank, "
                f"{len(shard)} proofs/rank)\n"
                f"[lean] verify   {t_check:6.1f}s  ({t_check/max(len(reqs),1):4.1f}s/proof, "
                f"{self.num_workers} workers/rank)\n"
                f"[lean] ok={ok}  error={err}  sorry={sry}  dead={dead}  "
                f"-> compile rate {rate:.3f}  ({ok}/{total})",
                flush=True,
            )
            for e in first_errors[:3]:
                print(f"[lean]   err: {str(e)[:160]}", flush=True)
            self.trainer.log(
                {
                    "eval/lean_compile_rate": rate,
                    "eval/lean_error": err / max(total, 1),
                    "eval/lean_sorry": sry / max(total, 1),
                    "eval/lean_dead": dead / max(total, 1),
                    "eval/lean_gen_seconds": t_gen,
                    "eval/lean_verify_seconds": t_check,
                }
            )


common = dict(
    output_dir=f"/checkpoint/scientific-reasoning/gaetan/ckpt/opsft2/{EXPE}",
    logging_first_step=True,
    report_to="wandb",
    run_name=EXPE,
    save_strategy="no",
    bf16=True,
    per_device_train_batch_size=8,
    eval_strategy="steps",
    eval_steps=25,
    logging_steps=1,
    # eval_on_start=True,
    # Lean proofs are long and unicode-heavy: the 1024 default truncates ~40% of them. In sft this
    # caps the training sequence, in opsft the one used for the SFT target gradient -- same meaning,
    # so set it once to keep the two runs comparable.
    max_length=4096,
)
if cli.learning_rate is not None:
    common["learning_rate"] = cli.learning_rate

if cli.trainer == "opsft":
    # max_completion_length bounds generation; max_length (above) bounds the teacher-forced pass.
    # One generation per prompt: the OPSFT advantages come from the SFT target gradient, not from a
    # group baseline, so a group of 1 is fine and halves the generation cost per prompt.
    config = OPSFTConfig(**common, max_completion_length=1024, num_generations=1)
else:
    config = SFTConfig(**common)

tok = AutoTokenizer.from_pretrained(MODEL)
lean_eval = LeanEval(tok, eval_set)

if cli.trainer == "opsft":
    # reward_funcs is required by the trainer, but the OPSFT loss builds its advantages from the
    # target gradient rather than from rewards, so a constant reward is a placeholder for now.
    # `completion` (the gold proof) still reaches here as a kwarg: remove_unused_columns is False.
    def zero_reward(completions, **kwargs):
        return [0.0] * len(completions)

    trainer = OPSFTTrainer(
        model=MODEL,
        reward_funcs=zero_reward,
        args=config,
        train_dataset=train_set,
        eval_dataset=eval_set,
        callbacks=[VibeCheck(tok=tok), lean_eval],
    )
else:
    trainer = SFTTrainer(
        model=MODEL,
        args=config,
        train_dataset=train_set,
        eval_dataset=eval_set,
        callbacks=[VibeCheck(tok=tok), lean_eval],
    )
lean_eval.trainer = trainer  # so it can log to wandb

print(f"Training begins after {time.time() - t} sec")
trainer.train()
