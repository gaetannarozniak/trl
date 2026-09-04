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

"""Dataset registry for the Lean baselines.

`REGISTRY` maps a name to a *zero-argument callable*. Nothing is loaded at import time, so adding
a dataset costs nothing until it is selected -- which keeps startup fast while iterating.

To add a dataset: write a `build_*` function returning a `Dataset` with `prompt` / `completion`
string columns, and register it at the bottom of the file.
"""

import re
from collections.abc import Callable

from datasets import Dataset, DatasetDict, load_dataset


# Lean block comments (`/- ... -/`), stripped before splitting so a docstring mentioning ":= by"
# cannot be mistaken for the start of the proof.
DOCSTRING = re.compile(r"/-.*?-/", re.S)
THEOREM_START = re.compile(r"^\s*(theorem|lemma|example)\b", re.M)

NUMINA_LEAN_PATH = "/storage/home/ilabiad/checkpoint/datasets/huggingface/NuminaMath-LEAN"


def _split_at_proof(example: dict) -> dict:
    """Split a Lean file into (prompt, completion) at the first `:= by`.

    Splits `formal_ground_truth` rather than `formal_statement`: only there does the header match
    the proof body verbatim, so `prompt + completion` is a compilable Lean file.

    Everything above the theorem is kept in the prompt. 36% of rows carry `open Real`,
    `open Finset`, `noncomputable section`, ... and the proof does not elaborate without them.
    """
    body = DOCSTRING.sub("", example["formal_ground_truth"])
    match = THEOREM_START.search(body)
    index = body.find(":= by", match.start()) if match else -1
    if index < 0:
        return {"prompt": None, "completion": None}
    cut = index + len(":= by")
    return {"prompt": body[:cut].strip(), "completion": body[cut:]}


def build_numina_lean(path: str = NUMINA_LEAN_PATH) -> Dataset:
    """Human-written, complete proofs from NuminaMath-LEAN.

    Only `author == "human"` and `ground_truth_type == "complete"` rows are usable: autoformalizer
    rows have no proof at all, `"statement"` rows are a bare `sorry`, and `"with_sorry"` rows still
    contain holes.
    """
    dataset = load_dataset(path)["train"]
    dataset = dataset.filter(lambda e: e["author"] == "human" and e["ground_truth_type"] == "complete")
    dataset = dataset.map(_split_at_proof, remove_columns=dataset.column_names)
    return dataset.filter(lambda e: e["prompt"] is not None and e["completion"].strip())


REGISTRY: dict[str, Callable[[], Dataset]] = {
    "numina_lean": build_numina_lean,
}


def get_dataset(key: str, eval_fraction: float = 0.05, seed: int = 0) -> DatasetDict:
    """Build a registered dataset and split it into train/test.

    Args:
        key: Key into [`REGISTRY`].
        eval_fraction: Fraction held out for evaluation.
        seed: Split seed. Keep fixed across baselines so their eval sets are identical.

    Returns:
        `DatasetDict` with `train` and `test` splits, each having `prompt` / `completion` columns.
    """
    if key not in REGISTRY:
        raise KeyError(f"Unknown dataset {key!r}. Available: {sorted(REGISTRY)}")
    split = REGISTRY[key]().train_test_split(test_size=eval_fraction, seed=seed)
    return DatasetDict(train=split["train"], test=split["test"])
