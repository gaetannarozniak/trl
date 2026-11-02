import re


def _normalize(s: str) -> str:
    s = s.split("=")[-1]  # "x = 34" -> "34"
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)  # "\boxed{34}" -> "34"
    s = s.replace("$", "").replace(" ", "").replace(",", "").rstrip(".")
    try:
        return str(float(s))  # "34", "34.0", "034" all match
    except ValueError:
        return s  # non-numeric: compare as text


def reward_function(output: str, ground_truth: str) -> float:
    matches = re.findall(r"Answer\s*:\s*([^\n]+)", output, re.IGNORECASE)
    if not matches:
        return 0.0  # no answer line (or truncated)
    return 1.0 if _normalize(matches[-1]) == _normalize(ground_truth) else 0.0
