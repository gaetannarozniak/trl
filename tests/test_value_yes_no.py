# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import pytest
import torch
from torch import nn

from trl.trainer.value_functions import ValueFunction, ValueFunctionConfig


class _StubTokenizer:
    """Tiny stand-in for a tokenizer: only what `ValueFunction.extra_inputs` needs."""

    def __init__(self, vocab, special_ids):
        self._vocab = vocab
        self._id_to_text = {i: t for t, i in vocab.items()}
        self.all_special_ids = special_ids
        self.vocab_size = len(vocab)

    def encode(self, text, add_special_tokens=False):
        return [self._vocab[text]]

    def batch_decode(self, batch, **kwargs):
        return [self._id_to_text[ids[0]] for ids in batch]


def _tokenizer():
    return _StubTokenizer(
        {
            "<pad>": 0,
            " Yes": 1,
            "YES": 2,
            "yes.": 3,
            "yesterday": 4,  # prefix match, by construction
            " no": 5,
            "Nope": 6,
            "NOT": 7,
            " hello": 8,
            "<probe>": 9,
        },
        special_ids=[0],
    )


def test_config_requires_probe_token():
    with pytest.raises(ValueError, match="requires probe_token"):
        ValueFunctionConfig(yes_no_tokens_as_value=True)
    ValueFunctionConfig(probe_token="<probe>", yes_no_tokens_as_value=True)  # no error


def test_yes_no_id_scan():
    vf = ValueFunction(ValueFunctionConfig(probe_token="<probe>", yes_no_tokens_as_value=True))
    assert vf.extra_inputs([], _tokenizer()) == {}
    assert vf._probe_id == 9
    assert set(vf._yes_ids) == {1, 2, 3, 4}
    assert set(vf._no_ids) == {5, 6, 7}


def test_yes_no_id_scan_empty_raises():
    vf = ValueFunction(ValueFunctionConfig(probe_token="<probe>", yes_no_tokens_as_value=True))
    tok = _StubTokenizer({"<pad>": 0, " hello": 1, "<probe>": 2}, special_ids=[0])
    with pytest.raises(ValueError, match="no 'yes' or no 'no'"):
        vf.extra_inputs([], tok)


def test_yes_no_values_math():
    vf = ValueFunction(ValueFunctionConfig(probe_token="<probe>", yes_no_tokens_as_value=True))
    vf._yes_ids, vf._no_ids = [0], [1]
    hidden = torch.zeros(2, 5, 8, requires_grad=True)

    head = nn.Linear(8, 2, bias=False)  # same interface as a policy lm_head
    with torch.no_grad():
        head.weight[0].fill_(1.0)  # yes logit = sum(hidden)
        head.weight[1].fill_(-1.0)  # no logit = -sum(hidden)
    vf._policy_head = head

    hidden_plus = hidden.detach().clone().requires_grad_(True)
    hidden_plus.data[..., 0] = 2.0  # yes logit 2, no logit -2
    values = vf._yes_no_values(hidden_plus)
    assert values.shape == (2, 5)
    assert torch.allclose(values, torch.full((2, 5), 2 * torch.sigmoid(torch.tensor(4.0)) - 1))
    values.sum().backward()
    assert hidden_plus.grad is not None and hidden_plus.grad.abs().sum() > 0

    hidden_minus = hidden.detach().clone()
    hidden_minus[..., 0] = -10.0  # all mass on no
    assert torch.allclose(vf._yes_no_values(hidden_minus), -torch.ones(2, 5), atol=1e-4)
