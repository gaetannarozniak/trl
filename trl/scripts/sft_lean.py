"""
HF_HUB_OFFLINE=1 accelerate launch \
  --config_file /home/gaetan/projects/trl/examples/accelerate_configs/deepspeed_zero2.yaml \
  scripts/sft_lean.py --trainer sft      # or: --trainer opsft
                      [--lr 1e-6] [--expe-name my_run] [--n-train 512] [--n-epochs 1]
                      [--beta 0] [--use-vllm] [--seed 7]
                      [--score-norm-probes 8] [--score-projection-steps 8] [--score-norm-damping 1e-6]
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
    "--n-train",
    type=int,
    default=None,
    help="Truncate the training set to this many examples. Uses the whole set when omitted.",
)
parser.add_argument(
    "--n-epochs",
    type=float,
    default=100,
    help="Number of training epochs.",
)
parser.add_argument(
    "--use-vllm",
    action="store_true",
    help="opsft only. Generate the completions with vLLM instead of transformers. Generation dominates the step, "
    "so this is the largest speedup available, but colocate mode reserves --vllm-gpu-mem of every GPU for it.",
)
parser.add_argument(
    "--vllm-gpu-mem",
    type=float,
    default=0.3,
    help="Fraction of each GPU vLLM reserves in colocate mode. Lower it if the training step then runs out of "
    "memory, raise it if vLLM cannot fit the KV cache.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seeds the training run: which prompts land in which batch, and the sampling of the completions. The "
    "train/eval split is NOT affected -- it stays on seed 0 so the held-out set is the same across every run.",
)
parser.add_argument(
    "--beta",
    type=float,
    default=0.05,
    help="opsft only. KL coefficient against the reference model. `0` drops the reference model entirely, which "
    "frees its memory and skips a forward pass per generation batch.",
)
parser.add_argument(
    "--score-norm-probes",
    type=int,
    default=0,
    help="opsft only. Divide the advantages by the per-token score norms, estimated with this many random "
    "probes, and log how much of the SFT gradient the update recovers. `0` keeps the raw inner products.",
)
parser.add_argument(
    "--score-projection-steps",
    type=int,
    default=0,
    help="opsft only. Conjugate-gradient steps solving for advantages that project the SFT gradient onto the span "
    "of the score vectors, rather than only rescaling them. Needs --score-norm-probes > 0. `3` is a good start.",
)
parser.add_argument(
    "--score-norm-damping",
    type=float,
    default=1e-2,
    help="opsft only. Damping on the projection solve, as a fraction of the mean score norm. This, not the step "
    "count, is what caps how close the solve gets: 1e-2 saturates immediately, 1e-6 keeps improving to ~8 steps.",
)
parser.add_argument(
    "--expe-name",
    default=None,
    help="wandb run name and checkpoint directory. Defaults to a name built from the settings that differ from "
    "the defaults, e.g. lean_opsft_lr1e-5_ntrain600_hutch8_proj3.",
)
parser.add_argument(
    "--print-run-name",
    action="store_true",
    help="Print the run name and exit, without importing torch. Used by submit_lean.sh to name the slurm job.",
)
cli = parser.parse_args()


def run_name():
    """lean_<trainer>, then one segment per setting that differs from its default."""
    fmt = lambda x: format(x, "g").replace("e-0", "e-").replace("e+0", "e")
    parts = [f"lean_{cli.trainer}"]
    if cli.learning_rate is not None:
        parts.append(f"lr{fmt(cli.learning_rate)}")
    if cli.n_train is not None:
        parts.append(f"ntrain{cli.n_train}")
    if cli.trainer == "opsft" and cli.use_vllm:
        parts.append("vllm")
    if cli.seed != parser.get_default("seed"):
        parts.append(f"seed{cli.seed}")
    if cli.trainer == "opsft" and cli.beta != parser.get_default("beta"):
        parts.append(f"beta{fmt(cli.beta)}")
    if cli.trainer == "opsft" and cli.score_norm_probes:
        parts.append(f"hutch{cli.score_norm_probes}")
        if cli.score_projection_steps:
            parts.append(f"proj{cli.score_projection_steps}")
            if cli.score_norm_damping != parser.get_default("score_norm_damping"):
                parts.append(f"damp{fmt(cli.score_norm_damping)}")
    return "_".join(parts)


EXPE = cli.expe_name or run_name()
if cli.print_run_name:
    print(EXPE)
    raise SystemExit

os.environ["WANDB_PROJECT"] = "opsft2"

import torch
import torch.distributed as dist
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, OPSFTTrainer, OPSFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

print(f"Imports done after {time.time() - t}")

MODEL = "allenai/Olmo-3-1025-7B"
tok = AutoTokenizer.from_pretrained(MODEL)

# Bounds the padded sequence the opsft JVP runs on. That pass is forced onto eager attention
# (forward-mode AD has no SDPA rule), so it materializes a B x heads x T x T float32 tensor: with
# T = max_prompt + max_completion, one 3.3k-token outlier in a batch takes it from 1.4 to 19 GiB.
# Filtered, not truncated -- a cut-off theorem statement is not a well-formed prompt, and the gold
# proof would no longer match it. Costs ~0.25% of the rows (prompts are p50 82, p90 149 tokens).
MAX_PROMPT_TOKENS = 512

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
n_before = len(ds)
ds = ds.filter(
    lambda batch: [len(ids) <= MAX_PROMPT_TOKENS for ids in tok(batch["prompt"])["input_ids"]],
    batched=True,
)
print(f"dropped {n_before - len(ds)}/{n_before} rows with prompts over {MAX_PROMPT_TOKENS} tokens")
split = ds.train_test_split(test_size=0.05, seed=0)
train_set, eval_set = split["train"], split["test"]
if cli.n_train is not None:
    train_set = train_set.select(range(min(cli.n_train, len(train_set))))
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
        self.last = None  # (prompts, completions) of the latest eval, for OnPolicyKL to reuse

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
        self.last = (list(shard["prompt"]), completions)
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


class OnPolicyKL(TrainerCallback):
    """mean_t [log pi_theta(y_t) - log pi_base(y_t)] over the policy's OWN generations.

    The same estimator and the same token-mean normalisation OPSFTTrainer logs as `kl`
    (opsft_trainer.py:1751), but run identically for both trainers, which is what makes SFT and
    OPSFT comparable on it. A KL over the *gold* tokens would be pointless here: it equals
    base_nll - eval_loss exactly, so it carries nothing the eval loss does not already.

    It reuses the completions LeanEval has just produced, so the only extra work is one forward of
    the frozen base model. Those are greedy (see `_generate`), so this measures drift of the greedy
    mode; OPSFTTrainer's own `kl` averages over temperature-1.0 samples. Trends are comparable
    between the two trainers, the absolute value is not comparable to `train/kl`.
    """

    def __init__(self, tok, lean_eval):
        self.tok, self.lean_eval = tok, lean_eval
        self.base, self.trainer = None, None

    def _sum_logps(self, model, prompts, completions):
        """(sum of log pi over the completion tokens, how many there were)."""
        total, count = 0.0, 0
        for prompt, completion in zip(prompts, completions):
            pi = self.tok(prompt)["input_ids"]
            ci = self.tok(completion, add_special_tokens=False)["input_ids"]
            if not ci:
                continue
            ids = torch.tensor([pi + ci], device=model.device)
            with torch.no_grad():
                logits = model(ids).logits[0, len(pi) - 1 : -1].float()
            target = torch.tensor(ci, device=model.device).unsqueeze(-1)
            total += logits.log_softmax(-1).gather(-1, target).sum().item()
            count += len(ci)
        return total, count

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or self.lean_eval.last is None:
            return
        if self.base is None:
            # Lazily loaded: a second copy of the model is 14 GB, so a run that never evaluates
            # never pays for it.
            self.base = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=torch.bfloat16, device_map={"": model.device}
            ).eval()
            self.base.requires_grad_(False)

        prompts, completions = self.lean_eval.last
        model.eval()
        theta, n = self._sum_logps(model, prompts, completions)
        base, _ = self._sum_logps(self.base, prompts, completions)
        model.train()

        # Every rank scores its own shard; sum both accumulators so the ratio is the token-mean over
        # the whole eval set rather than a per-rank average of per-rank means.
        stats = torch.tensor([theta - base, float(n)], device=model.device)
        if dist.is_initialized():
            dist.all_reduce(stats)
        if state.is_world_process_zero:
            self.trainer.log({"eval/on_policy_kl": (stats[0] / stats[1].clamp(min=1.0)).item()})


common = dict(
    output_dir=f"/checkpoint/scientific-reasoning/gaetan/ckpt/opsft2/{EXPE}",
    logging_first_step=True,
    report_to="wandb",
    run_name=EXPE,
    save_strategy="no",
    seed=cli.seed,
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
common["num_train_epochs"] = cli.n_epochs

if cli.trainer == "opsft":
    # max_completion_length bounds generation; max_length (above) bounds the teacher-forced pass.
    # One generation per prompt: the OPSFT advantages come from the SFT target gradient, not from a
    # group baseline, so a group of 1 is fine and halves the generation cost per prompt.
    config = OPSFTConfig(
        **common,
        max_completion_length=1024,
        num_generations=1,
        beta=cli.beta,
        use_vllm=cli.use_vllm,
        vllm_gpu_memory_utilization=cli.vllm_gpu_mem,
        # Prompts are filtered to MAX_PROMPT_TOKENS, so this is the tightest window that always fits.
        vllm_max_model_length=MAX_PROMPT_TOKENS + 1024,
        score_norm_probes=cli.score_norm_probes,
        score_projection_steps=cli.score_projection_steps,
        score_norm_damping=cli.score_norm_damping,
        # Each projection step is a full backward, so the activations of an uncheckpointed one are
        # paid j times over -- and next to vLLM's reserved fraction there is no room for them.
        gradient_checkpointing=cli.score_projection_steps > 0,
    )
else:
    config = SFTConfig(**common)

lean_eval = LeanEval(tok, eval_set)
# Registered after lean_eval so it runs second and can reuse the completions lean_eval just made.
on_policy_kl = OnPolicyKL(tok, lean_eval)
callbacks = [VibeCheck(tok=tok), lean_eval, on_policy_kl]

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
        callbacks=callbacks,
    )
else:
    trainer = SFTTrainer(
        model=MODEL,
        args=config,
        train_dataset=train_set,
        eval_dataset=eval_set,
        callbacks=callbacks,
    )
lean_eval.trainer = trainer  # so they can log to wandb
on_policy_kl.trainer = trainer

print(f"Training begins after {time.time() - t} sec")
trainer.train()
