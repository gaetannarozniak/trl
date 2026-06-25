import re
import unicodedata

from wordfreq import zipf_frequency

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

def _extract_predicted_answer(completion_text: str) -> str | None:
    match = re.search(r"####\s*([^\n]+)", completion_text)
    if match:
        return match.group(1).strip().replace(",", "")

    matches = re.findall(r"(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)", completion_text)
    if not matches:
        return None
    return matches[-1].replace("$", "").replace(",", "").strip()


def _normalize_gsm8k_answer(answer_text: str) -> str:
    if "####" not in answer_text:
        return answer_text.strip()
    return answer_text.split("####", 1)[1].strip().replace(",", "")


def _gsm8k_accuracy_reward(completions, solution, **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, solution, strict=True):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        pred = _extract_predicted_answer(content)
        rewards.append(1.0 if pred is not None and pred == gold else 0.0)
    return rewards
