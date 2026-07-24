"""
OPSFT on the SDFT science dataset, with live generation-based accuracy eval every `EVAL_EVERY` steps.

Trains on `science_data` (the new skill) and, at each eval, greedily generates with the *current training model*
(no separate vLLM — that would OOM next to a 7B training job) on three held-out sets:
  - science eval  → new-task accuracy (learning),
  - tooluse eval  → old-task accuracy (tool-use forgetting),
  - gsm8k eval    → old-task accuracy (math forgetting).
The science/tooluse accuracy definitions are ported verbatim from SDFT's eval_science.py / eval_tooluse.py.

Runs under DDP: eval prompts are sharded across processes and results gathered, so cost scales with #GPUs.
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass

import torch
from accelerate.utils import gather_object
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from trl import TrlParser
from trl.experimental.opsft import OPSFTConfig, OPSFTTrainer
from trl.models import unwrap_model_for_generation


os.environ["WANDB_PROJECT"] = "opsft"


@dataclass
class ScriptArguments:
    # Script-level knobs; everything else is OPSFTConfig, overridable on the CLI or via `--config <yaml>`.
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    sdft_data: str = "/home/gaetan/projects/SDFT/data"
    eval_every: int = 50  # run the accuracy eval every N optimizer steps
    gen_batch_size: int = 8  # generation micro-batch (per process)
    science_eval_size: int | None = 64  # None = all 507
    tooluse_eval_size: int | None = None  # None = all 97
    gsm8k_eval_size: int | None = 128  # None = all 1319 GSM8K test; math-capability (forgetting) probe
    science_max_new_tokens: int = 2048  # long enough for <reasoning>…</reasoning><answer>…</answer>
    tooluse_max_new_tokens: int = 2048
    gsm8k_max_new_tokens: int = 2048
    base_kl: bool = True  # log eval/base_kl = k3 KL(current ‖ base θ₀); reloads the base from disk each eval


script_args, training_args = TrlParser((ScriptArguments, OPSFTConfig)).parse_args_and_config()

MODEL = script_args.model_name
SDFT_DATA = script_args.sdft_data
EVAL_EVERY = script_args.eval_every
GEN_BATCH_SIZE = script_args.gen_batch_size
SCIENCE_EVAL_SIZE = script_args.science_eval_size
TOOLUSE_EVAL_SIZE = script_args.tooluse_eval_size
GSM8K_EVAL_SIZE = script_args.gsm8k_eval_size
SCIENCE_MAX_NEW_TOKENS = script_args.science_max_new_tokens
TOOLUSE_MAX_NEW_TOKENS = script_args.tooluse_max_new_tokens
GSM8K_MAX_NEW_TOKENS = script_args.gsm8k_max_new_tokens
BASE_KL = script_args.base_kl


# ------------------------------------------------------------------------------------------------------------------
# Accuracy scoring — ported verbatim from SDFT/eval_science.py and SDFT/eval_tooluse.py so the metric is identical.
# ------------------------------------------------------------------------------------------------------------------
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def score_science(responses, answers):
    return [1 if extract_xml_answer(r) == a else 0 for r, a in zip(responses, answers)]


def extract_actions(text):
    return re.findall(r"Action:\s*(\w+)", text)


def extract_action_inputs(text):
    json_blocks = re.findall(r"Action Input:\s*({.*?})", text, re.DOTALL)
    combined_dict = {}
    for block in json_blocks:
        try:
            combined_dict.update(json.loads(block))
        except json.JSONDecodeError:
            continue
    return combined_dict


def score_tooluse(responses, golden_answers):
    results = []
    for response, golden_answer in zip(responses, golden_answers):
        pred_actions = extract_actions(response)
        pred_inputs = extract_action_inputs(response)
        gt_actions = [item["Action"] for item in golden_answer]
        gt_inputs = {}
        for item in golden_answer:
            try:
                gt_inputs.update(json.loads(item["Action_Input"]))
            except json.JSONDecodeError:
                pass
        actions_match = Counter(pred_actions) == Counter(gt_actions)
        inputs_match = pred_inputs == gt_inputs
        results.append(1 if (actions_match and inputs_match) else 0)
    return results


GSM8K_SYSTEM_PROMPT = "Solve the math problem step by step. End with the final answer on its own line as '#### <number>'."


def _extract_gsm8k_number(text):
    # Number after the last "####" (else the last number in the text), commas/$ stripped.
    tail = text.split("####")[-1] if "####" in text else text
    matches = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", tail)
    return matches[-1].replace(",", "").replace("$", "") if matches else None


def score_gsm8k(responses, golds):
    results = []
    for r, g in zip(responses, golds):
        pred = _extract_gsm8k_number(r)
        try:
            results.append(1 if pred is not None and float(pred) == float(g) else 0)
        except ValueError:
            results.append(0)
    return results


# ------------------------------------------------------------------------------------------------------------------
# Live eval callback
# ------------------------------------------------------------------------------------------------------------------
class GenerationEvalCallback(TrainerCallback):
    """Every `EVAL_EVERY` steps (and once before training), generate on both eval sets and log accuracy."""

    def __init__(self, trainer, science_items, tooluse_items, gsm8k_items):
        self.trainer = trainer
        self.accelerator = trainer.accelerator
        self.tokenizer = trainer.processing_class
        # Pre-format prompts once: science/gsm8k prompts are chat messages, tooluse prompts are raw strings.
        self.science_items = [
            {"text": self._render(ex["prompt"]), "gold": ex["answer"]} for ex in science_items
        ]
        self.tooluse_items = [
            {"text": self._render([{"role": "user", "content": ex["prompt"]}]), "gold": ex["golden_answer"]}
            for ex in tooluse_items
        ]
        self.gsm8k_items = [
            {"text": self._render(ex["prompt"]), "gold": ex["answer"]} for ex in gsm8k_items
        ]

    def _render(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def _generate(self, texts, max_new_tokens):
        tok = self.tokenizer
        original_side = tok.padding_side
        tok.padding_side = "left"  # left-pad for decoder-only generation
        responses = []
        with unwrap_model_for_generation(self.trainer.model_wrapped, self.accelerator) as model:
            model.eval()
            for i in range(0, len(texts), GEN_BATCH_SIZE):
                batch = texts[i : i + GEN_BATCH_SIZE]
                enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.accelerator.device)
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
                gen = out[:, enc["input_ids"].shape[1] :]  # keep only the newly generated tokens
                responses.extend(tok.batch_decode(gen, skip_special_tokens=True))
            model.train()
        tok.padding_side = original_side
        return responses

    def _run(self, items, max_new_tokens, score_fn):
        # Shard prompts across processes, generate the local shard, gather everything back, score on all ranks.
        with self.accelerator.split_between_processes(items) as shard:
            responses = self._generate([it["text"] for it in shard], max_new_tokens)
            local = [{"response": r, "gold": it["gold"]} for it, r in zip(shard, responses)]
        gathered = gather_object(local)
        scores = score_fn([g["response"] for g in gathered], [g["gold"] for g in gathered])
        return sum(scores) / max(len(scores), 1)

    @torch.no_grad()
    def _token_logps(self, model, input_ids, attention_mask, comp_ids):
        # Teacher-forced per-token log-probs of the completion tokens under `model`. logits_to_keep+1 then drop the
        # last position, mirroring OPSFTTrainer._get_per_token_logps_and_entropies, so we keep the logits that
        # *predict* each completion token.
        logits_to_keep = comp_ids.shape[1]
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1, use_cache=False
        ).logits
        logits = logits[:, :-1, :][:, -logits_to_keep:, :]
        return torch.log_softmax(logits, dim=-1).gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def _base_kl(self):
        # k3 KL(current ‖ base θ₀) over the current model's greedy completions on the science prompts. The base is
        # reloaded from disk each eval (so it never occupies GPU memory during the training step / double-backward),
        # forwarded alongside the current model on prompt+completion, and the per-token estimator is averaged.
        tok = self.tokenizer
        pad_id = tok.pad_token_id
        base = AutoModelForCausalLM.from_pretrained(MODEL).to(self.accelerator.device)
        base.eval()
        original_side = tok.padding_side
        tok.padding_side = "left"
        local = []
        with self.accelerator.split_between_processes(self.science_items) as shard:
            with unwrap_model_for_generation(self.trainer.model_wrapped, self.accelerator) as cur:
                cur.eval()
                for i in range(0, len(shard), GEN_BATCH_SIZE):
                    texts = [it["text"] for it in shard[i : i + GEN_BATCH_SIZE]]
                    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
                        self.accelerator.device
                    )
                    gen = cur.generate(
                        **enc, max_new_tokens=SCIENCE_MAX_NEW_TOKENS, do_sample=False, pad_token_id=pad_id
                    )
                    comp_ids = gen[:, enc["input_ids"].shape[1] :]
                    comp_mask = (comp_ids != pad_id).float()
                    attn = torch.cat([enc["attention_mask"], (comp_ids != pad_id).long()], dim=1)
                    delta = self._token_logps(base, gen, attn, comp_ids) - self._token_logps(cur, gen, attn, comp_ids)
                    k3 = torch.exp(delta) - delta - 1  # k3 estimator of KL(current ‖ base), per token
                    per_seq = (k3 * comp_mask).sum(-1) / comp_mask.sum(-1).clamp(min=1.0)
                    local.extend(per_seq.tolist())
                cur.train()
        tok.padding_side = original_side
        del base
        gathered = gather_object(local)
        return sum(gathered) / max(len(gathered), 1)

    def _evaluate_and_log(self, step):
        science_acc = self._run(self.science_items, SCIENCE_MAX_NEW_TOKENS, score_science)
        tooluse_acc = self._run(self.tooluse_items, TOOLUSE_MAX_NEW_TOKENS, score_tooluse)
        gsm8k_acc = self._run(self.gsm8k_items, GSM8K_MAX_NEW_TOKENS, score_gsm8k)
        base_kl = self._base_kl() if BASE_KL else None
        if self.accelerator.is_main_process:
            line = (
                f"[step {step}] science_acc={science_acc:.4f}  tooluse_acc={tooluse_acc:.4f}  "
                f"gsm8k_acc={gsm8k_acc:.4f}"
            )
            if base_kl is not None:
                line += f"  base_kl={base_kl:.4f}"
            print(line, flush=True)
            import wandb

            # Log against our own `eval/step` axis with NO explicit `step=`: the Trainer free-runs wandb's global step
            # ahead of `global_step`, so an explicit `step=` below wandb's current step is silently dropped.
            if wandb.run is not None:
                logs = {
                    "eval/science_accuracy": science_acc,
                    "eval/tooluse_accuracy": tooluse_acc,
                    "eval/gsm8k_accuracy": gsm8k_acc,
                    "eval/step": step,
                }
                if base_kl is not None:
                    logs["eval/base_kl"] = base_kl
                wandb.log(logs)

    def on_train_begin(self, args, state, control, **kwargs):
        if self.accelerator.is_main_process:
            import wandb

            # Bind all eval/* metrics to the custom `eval/step` x-axis so they render even though we log them
            # out-of-order relative to wandb's free-running global step.
            if wandb.run is not None:
                wandb.define_metric("eval/step")
                wandb.define_metric("eval/*", step_metric="eval/step")
        self._evaluate_and_log(state.global_step)  # baseline (step 0) — the reference point for forgetting

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % EVAL_EVERY == 0:
            self._evaluate_and_log(state.global_step)


# ------------------------------------------------------------------------------------------------------------------
# Data & training
# ------------------------------------------------------------------------------------------------------------------
# Drop the longest-prompt samples so each dataset's max prompt length is halved — this shrinks the worst-case
# sequence length that drives OPSFT's eager-attention double-backward memory, without truncating any kept sample.
# tooluse_eval is left untouched: halving it would drop ~59% of the forgetting probe.
_filter_tokenizer = AutoTokenizer.from_pretrained(MODEL)


def filter_long_prompts(ds, prompt_text, name):
    lengths = [len(_filter_tokenizer(prompt_text(ex), add_special_tokens=False)["input_ids"]) for ex in ds]
    threshold = max(lengths) // 2
    kept = [i for i, length in enumerate(lengths) if length <= threshold]
    if os.environ.get("RANK", "0") == "0":
        print(
            f"[filter] {name}: max prompt {max(lengths)} tok -> keep <= {threshold}; "
            f"kept {len(kept)}/{len(ds)} (dropped {len(ds) - len(kept)})",
            flush=True,
        )
    return ds.select(kept)


science_train = load_from_disk(f"{SDFT_DATA}/science_data/train_data")
science_train = science_train.map(
    # OPSFT (x, y) pair: x = the chat prompt, y = the reference answer that defines the SFT direction u.
    lambda ex: {"prompt": ex["messages"], "completion": [{"role": "assistant", "content": ex["output_text"]}]},
    remove_columns=["messages", "output_text"],
)
science_train = filter_long_prompts(
    science_train, lambda ex: "\n".join(m["content"] for m in ex["prompt"]), "science_train"
)

science_eval = load_from_disk(f"{SDFT_DATA}/science_data/eval_data")
science_eval = filter_long_prompts(
    science_eval, lambda ex: "\n".join(m["content"] for m in ex["prompt"]), "science_eval"
)
if SCIENCE_EVAL_SIZE is not None:
    science_eval = science_eval.select(range(SCIENCE_EVAL_SIZE))
tooluse_eval = load_from_disk(f"{SDFT_DATA}/tooluse_data/eval_data")
if TOOLUSE_EVAL_SIZE is not None:
    tooluse_eval = tooluse_eval.select(range(TOOLUSE_EVAL_SIZE))

# GSM8K math-forgetting probe: prompt with a neutral "#### <number>" instruction and score the final number.
gsm8k_eval = load_dataset("openai/gsm8k", "main", split="test")
if GSM8K_EVAL_SIZE is not None:
    gsm8k_eval = gsm8k_eval.select(range(GSM8K_EVAL_SIZE))
gsm8k_eval = gsm8k_eval.map(
    lambda ex: {
        "prompt": [
            {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ],
        "answer": ex["answer"].split("####")[-1].strip(),
    },
    remove_columns=["question", "answer"],
)

model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
trainer = OPSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=science_train,
)
trainer.add_callback(GenerationEvalCallback(trainer, science_eval, tooluse_eval, gsm8k_eval))
trainer.train()
