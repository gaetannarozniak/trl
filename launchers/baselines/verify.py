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

"""Lean proof verification against a prebuilt Mathlib workspace.

Knows nothing about trainers or callbacks: it takes proofs in and returns counts out.

The `submit` / `drain` interface is deliberately queue-shaped even though verification is
currently synchronous, so it can be moved onto a background thread later without touching callers.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field


IMPORT_LINE = re.compile(r"^\s*import\b.*$", re.M)


@dataclass
class VerificationResult:
    """Outcome counts for one batch of proofs.

    - `ok`: compiled with no errors and no holes.
    - `truncated`: generation hit the token budget, so the proof is incomplete by construction.
      Split out from `error` because it measures the *harness*, not the model's mathematics: Lean
      reports it as `unexpected end of input`, indistinguishable in the logs from a genuine syntax
      error. It still counts in `total`, so `compile_rate` stays honest -- excluding it would turn
      the metric into "compile rate among proofs the model chose to finish", which flatters a
      model that rambles on hard theorems and only terminates on easy ones.
    - `error`: Lean rejected a proof the model actually finished.
    - `sorry`: elaborated but contains holes, so it proves nothing.
    - `dead`: the REPL crashed, errored, or timed out. Says more about the harness than the proof.
    - `unverified`: never got a verdict because the wall-clock budget expired. Counted against
      `total` so `compile_rate` remains a lower bound rather than silently improving, but tracked
      separately via `coverage` -- the proofs left unrun are the slow ones, and slow is not the
      same as wrong, so a `coverage` below 1 means the number is load-dependent and not comparable
      with other evaluations.
    """

    ok: int = 0
    error: int = 0
    truncated: int = 0
    sorry: int = 0
    dead: int = 0
    unverified: int = 0
    total: int = 0
    verified: int = 0
    seconds: float = 0.0
    timed_out: bool = False
    first_errors: list[str] = field(default_factory=list)

    @property
    def compile_rate(self) -> float:
        return self.ok / max(self.total, 1)


class LeanVerifier:
    """A pool of Lean REPL workers sharing one preloaded Mathlib environment."""

    def __init__(self, args, rank: int = 0):
        self.workspace = args.lean_workspace
        self.repl_path = args.lean_repl_path
        self.lake_path = args.lake_path
        self.num_workers = args.lean_num_workers
        self.timeout = args.lean_timeout
        self.max_seconds = args.lean_eval_max_seconds
        self.verify_truncated = args.lean_verify_truncated
        self.rank = rank
        self.pool = None
        self.env = None

    def _ensure_pool(self, barrier=None):
        """Build the pool on first use.

        Deferred because each worker loads Mathlib (~20 s), which is wasted if the run dies before
        the first evaluation.

        Constructing the config runs `lake exe cache get`, which unpacks into the shared
        `~/.cache/mathlib`. Several ranks doing that concurrently corrupt each other's `.ltar`
        files and the REPL then dies with `BrokenPipeError`, so rank 0 goes first and the others
        attach to the finished project.
        """
        if self.pool is not None:
            return self.pool

        from lean_interact import Command, LeanREPLConfig, LocalProject
        from lean_interact.pool import LeanServerPool

        start = time.time()

        def make_config():
            return LeanREPLConfig(
                # The workspace is prebuilt, so skip the `lake exe cache get` + `lake build` that
                # LocalProject runs by default: it costs minutes and reaches for the network from a
                # compute node.
                project=LocalProject(directory=self.workspace, auto_build=False),
                local_repl_path=self.repl_path,
                lake_path=self.lake_path,
                build_repl=False,
            )

        config = make_config() if self.rank == 0 else None
        if barrier is not None:
            barrier()
        if config is None:
            config = make_config()

        self.pool = LeanServerPool(config, num_workers=self.num_workers)
        # One session-cached `import Mathlib`; workers replay it on first use, then cache it.
        self.env = self.pool.run(Command(cmd="import Mathlib")).env

        # Force every worker through that replay now. Measured at ~143s for the first proof on a
        # cold worker versus ~2s once warm, so leaving it implicit means the first evaluation pays
        # it inside the timed section -- and, worse, pays it *unevenly* across ranks, which is what
        # pushed one rank past the NCCL collective watchdog and killed a run. Doing it here makes
        # the cost explicit, symmetric, and visible in the log.
        warm_start = time.time()
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            list(
                executor.map(
                    lambda _: self.pool.run(Command(cmd="example : True := trivial", env=self.env)),
                    range(self.num_workers),
                )
            )
        if self.rank == 0:
            print(
                f"[lean] pool ready: {self.num_workers} workers/rank, env={self.env} "
                f"({time.time() - start:.0f}s total, {time.time() - warm_start:.0f}s warming workers)",
                flush=True,
            )
        return self.pool

    def submit(
        self,
        prompts: list[str],
        completions: list[str],
        truncated: list[bool] | None = None,
        barrier=None,
    ) -> VerificationResult:
        """Verify `prompt + completion` for each pair. Blocks until done.

        Args:
            prompts: Theorem statements ending in `:= by`.
            completions: Model-generated proof bodies.
            truncated: Per-proof flag marking generations that hit the token budget. These are
                counted as failures but, unless `verify_truncated` is set, are not sent to Lean: a
                generation cut off mid-proof is syntactically incomplete by construction and
                cannot compile, so elaborating it only occupies a worker. Skipping them shortens
                the critical path considerably -- and since every rank blocks on an `all_reduce`
                afterwards, the slowest shard sets the pace for everyone.
            barrier: Collective used to stagger pool construction across ranks.
        """
        from lean_interact import Command

        pool = self._ensure_pool(barrier=barrier)
        truncated = list(truncated) if truncated is not None else [False] * len(prompts)
        result = VerificationResult(total=len(prompts))

        pending, pending_truncated = [], []
        for prompt, completion, was_truncated in zip(prompts, completions, truncated, strict=True):
            if was_truncated and not self.verify_truncated:
                result.truncated += 1
                continue
            pending.append((prompt, completion))
            pending_truncated.append(was_truncated)

        if not pending:
            return result

        # Mathlib already lives in the session env, so drop the `import` lines but keep the rest of
        # the preamble (`open Real`, `noncomputable section`, ...) the proof needs to elaborate.
        requests = [
            Command(cmd=IMPORT_LINE.sub("", prompt + completion).strip(), env=self.env)
            for prompt, completion in pending
        ]

        start = time.time()
        result.seconds = 0.0
        # Submitted individually rather than via `pool.run_batch` so the wall-clock budget can be
        # enforced: `run_batch` blocks until the last proof returns, which is exactly the
        # behaviour that let one slow rank hold the others in an `all_reduce` past NCCL's 1800s
        # watchdog and kill a run.
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        futures = {
            executor.submit(pool.run, request, timeout=self.timeout): index for index, request in enumerate(requests)
        }
        verdicts: dict[int, object] = {}
        try:
            budget = self.max_seconds if self.max_seconds and self.max_seconds > 0 else None
            for future in as_completed(futures, timeout=budget):
                try:
                    verdicts[futures[future]] = future.result()
                except Exception as exc:  # noqa: BLE001 - a crash is a verdict, not a failure here
                    verdicts[futures[future]] = exc
        except FuturesTimeout:
            result.timed_out = True
        finally:
            # Do not wait: in-flight proofs are already bounded by `self.timeout`, and blocking
            # here would defeat the budget we just enforced.
            executor.shutdown(wait=False, cancel_futures=True)
        result.seconds = time.time() - start
        result.verified = len(verdicts)
        result.unverified = len(requests) - len(verdicts)

        for index, item in verdicts.items():
            was_truncated = pending_truncated[index]
            if not hasattr(item, "env"):
                result.dead += 1
                continue
            errors = [
                message.data
                for message in (getattr(item, "messages", None) or [])
                if getattr(message, "severity", "") == "error"
            ]
            if errors:
                # Classify by cause rather than symptom: Lean reports a cut-off proof as a plain
                # syntax error, indistinguishable from genuinely wrong mathematics.
                if was_truncated:
                    result.truncated += 1
                else:
                    result.error += 1
                    if len(result.first_errors) < 3:
                        result.first_errors.append(str(errors[0]))
            elif getattr(item, "sorries", None):
                result.sorry += 1
            else:
                result.ok += 1
        return result

    def drain(self, timeout: float | None = None) -> None:
        """No-op while verification is synchronous. Kept so callers need not change later."""
        return
