import re
import unicodedata

from datasets import load_dataset
from wordfreq import zipf_frequency

from trl import TrlParser
from trl.experimental.fd import FDConfig, FDTrainer

print("Finished imports")

_WORD_RE = re.compile(r"[a-zA-ZÀ-ſ]+(?:'[a-zA-ZÀ-ſ]+)?")

def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def french_ratio_reward(completions, **kwargs):
    """Fraction of word tokens whose French zipf-frequency exceeds their English zipf-frequency."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        normalized = _strip_accents(text).lower()
        words = _WORD_RE.findall(normalized)
        if not words:
            rewards.append(0.0)
            continue
        n_french = sum(1 for w in words if zipf_frequency(w, "fr") > zipf_frequency(w, "en"))
        rewards.append(n_french / len(words))
    return rewards

def reward_func(completions, **kwargs):
    return [len(c) for c in completions]

dataset = load_dataset(
    "json",
    # data_files="/checkpoint/scientific-reasoning/gaetan/trl/datasets/miniF2F/valid.jsonl",
    data_files="/checkpoint/scientific-reasoning/gaetan/trl/datasets/numinamath_lean/train.jsonl",
    split="train",
)
eval_dataset = load_dataset(
    "json",
    # data_files="/checkpoint/scientific-reasoning/gaetan/trl/datasets/miniF2F/test32.jsonl",
    data_files="/checkpoint/scientific-reasoning/gaetan/trl/datasets/numinamath_lean/test.jsonl",
    split="train",
)
privileged_context =  "\nReponds en francais. Je veux que toute ta reponse soit en francais, n'ecris pas un seul mot en anglais a part le code lean. C'est important que ce soit ecrit en francais."
dataset = dataset.map(lambda x: {"privileged_context": privileged_context})
eval_dataset = eval_dataset.map(lambda x: {"privileged_context": privileged_context})

parser = TrlParser(FDConfig)
training_args, = parser.parse_args_and_config()

trainer = FDTrainer(
    model="Qwen/Qwen3.5-2B",
    reward_funcs=[french_ratio_reward],
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
