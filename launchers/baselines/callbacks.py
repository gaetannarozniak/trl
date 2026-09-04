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

"""Evaluation callbacks: Lean compile rate and a qualitative sample.

Both run on explicit step intervals rather than on every log. The previous implementation hooked
`on_log`, which with `logging_steps=1` fired every step and consumed ~52% of an 11-hour run.
"""

import time

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from trainer import log_exact


class _IntervalCallback(TrainerCallback):
    """Fires every `interval` steps, plus once before training when `eval_on_start` is set.

    The step-0 measurement has to come from `on_train_begin`, not `on_step_end`: the latter first
    fires once `global_step` is already 1, so a `global_step == 0` test there can never be true.
    Subclasses implement `_fire`.
    """

    def __init__(self, interval: int):
        self.interval = interval
        self._last_fired = -1

    def _should_fire(self, state) -> bool:
        step = state.global_step
        if step == self._last_fired or self.interval <= 0:
            return False
        return step % self.interval == 0 or step >= state.max_steps

    def _fire(self, args, state) -> None:
        raise NotImplementedError

    def on_train_begin(self, args, state, control, **kwargs):
        # The untrained baseline: without it none of the later numbers have a reference point.
        if getattr(args, "eval_on_start", False):
            self._last_fired = state.global_step
            self._fire(args, state)

    def on_step_end(self, args, state, control, **kwargs):
        if not self._should_fire(state):
            return
        self._last_fired = state.global_step
        self._fire(args, state)


class VibeCheck(_IntervalCallback):
    """Print generated proofs for a couple of held-out theorems.

    Purely qualitative: it is how you notice degenerate repetition or a broken prompt format long
    before the compile rate would tell you.
    """

    def __init__(self, generator, prompts: list[str], args):
        super().__init__(args.vibecheck_steps)
        self.generator = generator
        self.prompts = list(prompts)
        self.max_new_tokens = args.vibecheck_max_new_tokens

    def _fire(self, args, state) -> None:
        # Every rank must enter this: `sync()` can be collective. Only rank 0 prints.
        self.generator.sync(state.global_step)
        generated = self.generator.generate(self.prompts, self.max_new_tokens)

        if not state.is_world_process_zero:
            return
        for prompt, completion in zip(self.prompts, generated.texts, strict=False):
            print(
                f"\n\033[1m[step {state.global_step}] THEOREM:\033[0m\n{prompt}\n\033[1mPROOF:\033[0m{completion}",
                flush=True,
            )


class LeanEval(_IntervalCallback):
    """Generate proofs for held-out theorems and verify them with Lean.

    Work is sharded across ranks: each rank generates and verifies its own slice, then the counts
    are summed. Nothing sits idle waiting on rank 0, which also keeps every rank inside the NCCL
    collective timeout.
    """

    def __init__(self, generator, verifier, dataset, args):
        super().__init__(args.lean_eval_steps)
        self.generator = generator
        self.verifier = verifier
        limit = args.lean_eval_n or len(dataset)
        self.dataset = dataset.select(range(min(limit, len(dataset))))
        self.max_new_tokens = args.lean_max_new_tokens
        self.trainer = None

    def on_train_end(self, args, state, control, **kwargs):
        self.verifier.drain()

    def _fire(self, args, state) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        shard = self.dataset.shard(num_shards=world, index=rank) if world > 1 else self.dataset
        eval_start = time.time()

        if rank == 0:
            print(
                f"\n[lean] step {state.global_step}: {len(self.dataset)} theorems over {world} rank(s)",
                flush=True,
            )

        self.generator.sync(state.global_step)
        start = time.time()
        prompts = list(shard["prompt"])
        generated = self.generator.generate(prompts, self.max_new_tokens)
        generate_seconds = time.time() - start

        # A completion of exactly `max_new_tokens` ran out of budget rather than finishing.
        truncated = [length >= self.max_new_tokens for length in generated.lengths]

        barrier = dist.barrier if dist.is_initialized() else None
        result = self.verifier.submit(prompts, generated.texts, truncated=truncated, barrier=barrier)
        eval_seconds = time.time() - eval_start

        # Sum raw counts across ranks, then divide once: rates must be token/proof-weighted, not an
        # average of per-rank rates, because shards can differ in size.
        counts = torch.tensor(
            [
                result.ok,
                result.error,
                result.truncated,
                result.sorry,
                result.dead,
                result.unverified,
                result.total,
                generated.total_tokens,
                sum(truncated),
            ],
            dtype=torch.long,
            device=self.trainer.accelerator.device,
        )
        if world > 1:
            dist.all_reduce(counts)
        ok, error, trunc, sorry_, dead, unverified, total, gen_tokens, n_trunc = counts.tolist()
        rate = ok / max(total, 1)
        coverage = 1.0 - unverified / max(total, 1)
        mean_length = gen_tokens / max(total, 1)
        # Derived from the reduced totals so it describes the whole eval, not rank 0's slice. The
        # ranks generate concurrently, so node throughput is this times the world size.
        tokens_per_second_per_rank = gen_tokens / world / max(generate_seconds, 1e-9)

        if rank != 0:
            return

        print(
            f"[lean] generate {generate_seconds:6.1f}s  ({len(shard)} proofs/rank, "
            f"{tokens_per_second_per_rank:5.0f} tok/s/rank, mean {mean_length:.0f} tok, "
            f"{n_trunc / max(total, 1):.1%} at max length)\n"
            f"[lean] verify   {result.seconds:6.1f}s  "
            f"({result.seconds / max(result.verified, 1):4.1f}s/proof, {result.verified} verified/rank, "
            f"{self.verifier.num_workers} workers/rank)\n"
            f"[lean] ok={ok}  error={error}  truncated={trunc}  sorry={sorry_}  dead={dead}  "
            f"unverified={unverified}  -> compile rate {rate:.3f}  ({ok}/{total}, "
            f"coverage {coverage:.1%})",
            flush=True,
        )
        for message in result.first_errors[:3]:
            print(f"[lean]   err: {message[:160]}", flush=True)

        if coverage < 1.0:
            print(
                f"[lean] WARNING: wall-clock budget expired, only {coverage:.1%} of proofs were "
                f"verified ({unverified} unverified, counted as failures). The unrun proofs are "
                f"the slow ones rather than a random sample, so this compile rate is a "
                f"load-dependent lower bound and is NOT comparable with other steps. Lower "
                f"`lean_eval_n` or raise `lean_eval_max_seconds`.",
                flush=True,
            )

        log_exact(
            state,
            {
                "eval/lean_compile_rate": rate,
                "eval/lean_error": error / max(total, 1),
                # Failed for lack of token budget rather than for being wrong. If this is large the
                # compile rate is capped by `lean_max_new_tokens`, not by the model.
                "eval/lean_truncated": trunc / max(total, 1),
                "eval/lean_sorry": sorry_ / max(total, 1),
                "eval/lean_dead": dead / max(total, 1),
                # Below 1 means the budget bit and `lean_compile_rate` is only a lower bound.
                "eval/lean_coverage": coverage,
                "eval/gen_mean_tokens": mean_length,
                "eval/gen_at_max_length": n_trunc / max(total, 1),
                "eval/gen_tokens_per_second_per_rank": tokens_per_second_per_rank,
                "time/eval_generation": generate_seconds,
                "time/eval_verification": result.seconds,
                "time/eval_total": eval_seconds,
            },
            fallback=self.trainer.log,
        )
