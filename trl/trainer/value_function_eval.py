"""Score the critic against Monte-Carlo value estimates measured outside training.

../value_estimation branched 128 rollouts off six Qwen3-8B generations every 512 tokens and averaged their
outcomes, which is the quantity the critic is trying to predict. Comparing against it answers a question the
value loss cannot: the loss only says the critic fits the reward it was shown, not that it tracks the true
probability of success along a trace it never trained on.

The six traces are a correct and an incorrect generation of AIME 2026 problems 1044, 1054 and 1059. None of
them sits in the hard subset the sweep trains on, so this is deliberately an out-of-distribution probe.

Everything here builds the critic's input the way the trainer does and then calls `ValueFunction.get_values`
itself rather than reimplementing it, so the two cannot drift apart silently. What could still drift is the
*input*, and `selfcheck` is what catches that: it rebuilds a live training row through this module's path and
compares against the values the trainer just computed for it.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


REFERENCE = Path("/storage/home/gaetan/projects/value_estimation")
RUN = REFERENCE / "runs/q3_v1"
SCREEN = REFERENCE / "runs/q3_screen/summary.json"
PROBLEMS = REFERENCE / "data/aime_qwen3.json"


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x) ** 0.5
    vy = sum((b - my) ** 2 for b in y) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def load_traces(processing_class, privileged_template: str | None) -> list[dict]:
    """The six reference traces, tokenised the way the trainer tokenises a training prompt.

    `prompt_len` recorded at generation time is asserted against what the chat template produces now, so a
    template or tokenizer change turns into an error here rather than a quietly wrong curve.
    """
    meta = json.loads((RUN / "meta.json").read_text())
    problems = {p["problem_idx"]: p["problem"] for p in json.loads(PROBLEMS.read_text())}
    pass_rate = {r["problem_idx"]: r["pass_rate"] for r in json.loads(SCREEN.read_text())["problems"]}

    monte_carlo = defaultdict(dict)
    with open(RUN / "curve.csv") as f:
        for row in csv.DictReader(f):
            monte_carlo[f"{row['problem_idx']}:{row['sample']}"][int(row["cut"])] = float(row["value"])

    traces = []
    for key, m in meta.items():
        conversation = [{"role": "user", "content": problems[m["problem_idx"]]}]
        prompt_ids = processing_class.apply_chat_template(
            conversation=[conversation], add_generation_prompt=True, tokenize=True, return_dict=True
        )["input_ids"][0]
        if len(prompt_ids) != m["prompt_len"]:
            raise ValueError(
                f"{key}: the prompt tokenises to {len(prompt_ids)} ids but the trace was generated from "
                f"{m['prompt_len']}, so the critic would not see the state the rollouts branched off"
            )
        if privileged_template is not None:
            conversation = conversation + [{"role": "user", "content": privileged_template.format(solution=m["gold"])}]
            prompt_ids = processing_class.apply_chat_template(
                conversation=[conversation], add_generation_prompt=True, tokenize=True, return_dict=True
            )["input_ids"][0]
        traces.append(
            {
                "key": key,
                "prompt_ids": list(prompt_ids),
                "completion_ids": m["token_ids"],
                "correct": bool(m["trace_correct"]),
                "baseline": pass_rate[m["problem_idx"]],  # Rloo is the leave-one-out mean reward
                "monte_carlo": monte_carlo[key],
            }
        )
    return traces


def _values_for(value_function, critic, prompt_ids, completion_ids, baseline, reward, device):
    """One critic forward over prompt+completion, through the trainer's own `get_values`."""
    prompt = torch.tensor(prompt_ids, device=device).unsqueeze(0)
    completion = torch.tensor(completion_ids, device=device).unsqueeze(0)
    input_ids = torch.cat([prompt, completion], dim=1)
    attention_mask = torch.ones_like(input_ids)
    inputs = {
        "completion_ids": completion,
        "completion_mask": torch.ones_like(completion),
        "rewards": torch.tensor([reward], device=device),
        "rloo_baseline": torch.tensor([baseline], device=device),
    }
    if value_function.args.privileged_template is not None:
        # get_values rebuilds input_ids from these, exactly as it does in training
        inputs["critic_prompt_ids"] = prompt
        inputs["critic_prompt_mask"] = torch.ones_like(prompt)
    with torch.no_grad():
        return value_function.get_values(critic, input_ids, attention_mask, completion.size(1), inputs)[0]


def evaluate(value_function, critic, device, step=0) -> tuple[dict[str, float], str]:
    """Correlation and error of the critic against the Monte-Carlo curve, per trace and pooled, and a chart."""
    traces = load_traces(value_function._processing_class, value_function.args.privileged_template)
    metrics, pooled_pred, pooled_mc, by_outcome = {}, [], [], defaultdict(lambda: ([], []))
    drawn = []
    for trace in traces:
        values = _values_for(
            value_function, critic, trace["prompt_ids"], trace["completion_ids"],
            trace["baseline"], float(trace["correct"]), device,
        )
        cuts = sorted(trace["monte_carlo"])
        predicted = [values[cut].item() for cut in cuts]
        target = [trace["monte_carlo"][cut] for cut in cuts]
        metrics[f"eval/r_{trace['key']}"] = _pearson(predicted, target)
        metrics[f"eval/mse_{trace['key']}"] = sum((p - t) ** 2 for p, t in zip(predicted, target)) / len(cuts)
        pooled_pred += predicted
        pooled_mc += target
        group = by_outcome["correct" if trace["correct"] else "incorrect"]
        group[0].extend(predicted)
        group[1].extend(target)
        stride = max(1, values.numel() // 400)  # the critic has a value at every token; the chart needs 400
        drawn.append({**trace, "cuts": cuts, "target": target,
                      "xs": list(range(0, values.numel(), stride)),
                      "ys": values[::stride].tolist()})
    metrics["eval/r"] = _pearson(pooled_pred, pooled_mc)
    metrics["eval/mse"] = sum((p - t) ** 2 for p, t in zip(pooled_pred, pooled_mc)) / len(pooled_pred)
    for outcome, (pred, mc) in by_outcome.items():
        metrics[f"eval/r_{outcome}"] = _pearson(pred, mc)
    return metrics, build_plot(drawn, step)


def selfcheck(value_function, critic, inputs, values) -> float:
    """Largest gap between a training row's values and the same row rebuilt through this module's path.

    The point is not the number but what makes it move: padding, `logits_to_keep`, the privileged prompt and
    the residual baseline all feed both paths, so anything this module gets wrong about how the trainer
    assembles a critic input shows up here instead of in a plausible-looking curve.
    """
    privileged = "critic_prompt_ids" in inputs
    prompt_ids = (inputs["critic_prompt_ids"] if privileged else inputs["prompt_ids"])[0]
    # Strip the left padding: training hides it behind the attention mask, `_values_for` would read it.
    keep = (inputs["critic_prompt_mask"] if privileged else inputs["prompt_mask"])[0].bool()
    length = int(inputs["completion_mask"][0].sum())
    rebuilt = _values_for(
        value_function,
        critic,
        prompt_ids[keep].tolist(),
        inputs["completion_ids"][0, :length].tolist(),
        inputs["rloo_baseline"][0].item(),
        inputs["rewards"][0].item(),
        values.device,
    )
    return (rebuilt[: length + 1] - values[0, : length + 1]).abs().max().item()


CORRECT, WRONG = "#0072B2", "#D55E00"  # Okabe-Ito, as in the advantage chart


def build_plot(drawn: list[dict], step: int) -> str:
    """The critic's value against the Monte-Carlo value it is trying to predict, one panel per problem.

    Markers are the ground truth: 128 rollouts branched off that token, so they carry real error bars. The
    line is the critic, drawn at every token because it costs nothing to evaluate - the question the chart
    answers is whether it has the shape of the curve, not just its average.
    """
    by_problem = defaultdict(list)
    for d in drawn:
        by_problem[d["key"].split(":")[0]].append(d)
    problems = sorted(by_problem)

    pw, ph, left, right, top, bottom = 320, 250, 46, 16, 28, 40
    svg = [f'<svg viewBox="0 0 {pw * len(problems)} {ph}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    for panel, problem in enumerate(problems):
        traces = by_problem[problem]
        ox = panel * pw
        xmax = max(max(d["xs"]) for d in traces) or 1

        def fx(x, ox=ox, xmax=xmax):
            return ox + left + (pw - left - right) * x / xmax

        def fy(y):
            return top + (ph - top - bottom) * (1.06 - y) / 1.12

        for i in range(5):
            y = i / 4
            svg.append(f'<line x1="{fx(0):.1f}" y1="{fy(y):.1f}" x2="{fx(xmax):.1f}" y2="{fy(y):.1f}" stroke="#e5e7eb"/>')
            if panel == 0:
                svg.append(f'<text x="{fx(0) - 6:.1f}" y="{fy(y) + 4:.1f}" text-anchor="end" fill="#6b7280">{y:.2f}</text>')
        for i in range(3):
            x = xmax * i / 2
            svg.append(
                f'<text x="{fx(x):.1f}" y="{ph - bottom + 16}" text-anchor="middle" fill="#6b7280">{x / 1000:.0f}k</text>'
            )
        for d in traces:
            colour = CORRECT if d["correct"] else WRONG
            path = " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(d["xs"], d["ys"]))
            svg.append(f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="1.6" opacity="0.9"/>')
            for cut, value in zip(d["cuts"], d["target"]):
                svg.append(
                    f'<circle cx="{fx(cut):.1f}" cy="{fy(value):.1f}" r="2.6" fill="{colour}" '
                    f'stroke="white" stroke-width="0.7"><title>{d["key"]} cut {cut}: MC {value:.3f}</title></circle>'
                )
            # the outcome itself, at the end of the trace: not an estimate
            svg.append(
                f'<path d="M{fx(max(d["xs"])) - 4:.1f},{fy(float(d["correct"])):.1f} l8,0" stroke="{colour}" stroke-width="2"/>'
            )
        ids = " ".join(d["key"].split(":")[1] for d in traces)
        svg.append(
            f'<text x="{ox + left}" y="16" font-size="11" fill="#111827">problem {problem} - samples {ids}</text>'
        )
    svg.append("</svg>")

    legend = "".join(
        f'<span style="margin-right:18px">{mark} {text}</span>'
        for mark, text in (
            (f'<svg width="24" height="10" style="vertical-align:middle"><line x1="1" y1="5" x2="23" y2="5" stroke="{CORRECT}" stroke-width="2"/></svg>', "critic, correct trace"),
            (f'<svg width="24" height="10" style="vertical-align:middle"><line x1="1" y1="5" x2="23" y2="5" stroke="{WRONG}" stroke-width="2"/></svg>', "critic, incorrect trace"),
            ('<svg width="24" height="10" style="vertical-align:middle"><circle cx="12" cy="5" r="3" fill="#6b7280"/></svg>', "Monte-Carlo ground truth, 128 rollouts"),
        )
    )
    return (
        '<!DOCTYPE html><meta charset="utf-8">'
        '<div style="background:#fff;padding:8px;color:#111827;font:11px ui-sans-serif,system-ui,sans-serif">'
        f'<div style="padding-bottom:4px">step {step} - critic value vs Monte-Carlo, token position on x</div>'
        f'{"".join(svg)}<div style="padding-top:6px;color:#374151">{legend}</div></div>'
    )
