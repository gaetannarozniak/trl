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

"""A single vLLM engine shared by every evaluation callback."""

from typing import NamedTuple

import torch.nn as nn

from trl.generation import VLLMGeneration


class Generated(NamedTuple):
    """Decoded completions plus the token counts needed to reason about cost and truncation.

    `lengths` is the number of generated tokens per completion, before decoding, so a completion
    that hit the budget is exactly `length == max_new_tokens`. Decoded text cannot tell you that:
    a proof truncated mid-token looks like any other short string.
    """

    texts: list[str]
    lengths: list[int]

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)


def restore_hf_call(*models: nn.Module) -> int:
    """Undo vLLM's class-level `__call__` patch for Hugging Face modules.

    vLLM has no native Olmo-3 implementation -- its registry maps `Olmo3ForCausalLM` to
    `("transformers", "TransformersForCausalLM")` -- so it loads the model through its Transformers
    backend. That backend calls `support_torch_compile(...)(cls)` on the *Hugging Face* decoder
    class (`vllm/model_executor/models/transformers/base.py:264`), which replaces
    `cls.__call__` **process-wide**. Every `Olmo3Model` instance is affected, including the
    training policy and the frozen reference, and vLLM's `__call__` immediately reads
    `self.do_not_compile` -- an attribute only its own patched `__init__` sets. The result is
    `AttributeError: 'Olmo3Model' object has no attribute 'do_not_compile'` on the next HF forward
    pass, i.e. as soon as training or evaluation resumes after the engine is built.

    The decorator patches only `__init__` and `__call__`; `forward` is left alone, and its
    `__call__` short-circuits to `self.forward(...)` when `do_not_compile` is set. Marking our own
    modules therefore restores the original behaviour exactly.

    Caveat: that short-circuit bypasses `nn.Module`'s hook machinery for the affected submodules.
    Harmless under DDP and ZeRO-2, where hooks live on the top-level wrapper and parameters are
    unsharded, but ZeRO-3 or FSDP parameter-gathering hooks on inner modules would be skipped.

    Returns:
        Number of modules marked, so callers can log it and notice if the patch ever stops applying.
    """
    patched = 0
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            if type(module).__call__ is not nn.Module.__call__ and not hasattr(module, "do_not_compile"):
                module.do_not_compile = True
                patched += 1
    return patched


class EvalGenerator:
    """Owns one colocated vLLM engine and keeps its weights in sync with training.

    One engine, not one per callback: each colocated engine statically reserves
    `gpu_memory_utilization` of the device, so a second one would double the reservation. Callbacks
    that need different decoding settings pass `sampling_overrides` instead.

    ## Weight synchronisation

    The engine holds its own copy of the weights, loaded once at construction. Without an explicit
    `sync_weights()` every evaluation would score the *initial* checkpoint forever -- and the
    symptom of that bug is a metric that never moves, which is indistinguishable from a model that
    is not learning. `sync()` is therefore called before every generation and is a no-op when the
    weights have not changed since the last call.

    ## Distributed usage

    Colocate mode with `tensor_parallel_size=1` means `generate()` performs no cross-rank
    collectives: every rank runs its own engine over its own shard. `sync_weights()` may still be
    collective under ZeRO-3 or FSDP, so **every rank must call these methods**, never only rank 0.
    """

    def __init__(self, model, accelerator, processing_class, args):
        self.processing_class = processing_class
        self._last_synced_step = -1
        self.generation = VLLMGeneration(
            model=model,
            accelerator=accelerator,
            processing_class=processing_class,
            mode="colocate",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_length=args.vllm_max_model_length,
            max_num_seqs=args.vllm_max_num_seqs,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            attention_backend=args.vllm_attention_backend,
            enable_sleep_mode=args.vllm_enable_sleep_mode,
            trust_remote_code=args.trust_remote_code,
            # Instance defaults are greedy; callbacks override per call.
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_completion_length=args.lean_max_new_tokens,
            logprobs=None,
        )

    def sync(self, step: int) -> None:
        """Push current training weights into the engine, unless already done for this step."""
        if step != self._last_synced_step:
            self.generation.sync_weights()
            self._last_synced_step = step

    def generate(self, prompts: list[str], max_new_tokens: int) -> Generated:
        """Greedily complete `prompts`.

        Args:
            prompts: Raw prompt strings; tokenisation and decoding are handled here because
                `VLLMGeneration` works in token ids.
            max_new_tokens: Generation budget. Can only be lowered relative to the engine's
                `max_model_length`, which is fixed at construction.

        Returns:
            A [`Generated`] with the decoded continuations and their token counts.
        """
        if not prompts:
            return Generated([], [])
        # `datasets` 5.x returns a `Column` from `dataset["col"]`, which the tokenizer rejects.
        prompts = list(prompts)
        prompt_ids = self.processing_class(prompts, add_special_tokens=True)["input_ids"]
        _, completion_ids, _, _ = self.generation.generate(
            prompts=prompt_ids,
            images=None,
            num_generations=1,
            sampling_overrides={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "max_completion_length": max_new_tokens,
                "logprobs": None,
            },
        )
        texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        return Generated(texts, [len(ids) for ids in completion_ids])
