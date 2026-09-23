"""Shared plumbing for the plot scripts: pull a metric from wandb and draw mean +- 1 std over seeds.

Runs are grouped by name with the trailing `|s<seed>` stripped, so the seeds of one configuration are averaged
together. Each seed's history is binned onto a common step grid before averaging, which keeps the comparison
honest when the runs are at different points or were sampled at different steps; a seed that stops early simply
stops contributing to the later bins.
"""

import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wandb  # noqa: E402


PROJECT = os.environ.get("WANDB_PROJECT", "value_function3")
# Okabe-Ito without black and yellow: stays separable under the common colour-vision deficiencies
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
BINS = 200
SMOOTH = 9  # width of the moving average, in bins


def _binned(steps, values, edges):
    """Mean of `values` per bin of `edges`, NaN where the run has no point in the bin."""
    inside = (steps >= edges[0]) & (steps <= edges[-1])  # else everything past the range piles into the last bin
    steps, values = steps[inside], values[inside]
    index = np.clip(np.digitize(steps, edges) - 1, 0, len(edges) - 2)
    total = np.bincount(index, weights=values, minlength=len(edges) - 1)
    count = np.bincount(index, minlength=len(edges) - 1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _smooth(y):
    """Centred moving average that steps over the NaN bins rather than propagating them."""
    if SMOOTH <= 1:
        return y
    kernel = np.ones(SMOOTH)
    present = ~np.isnan(y)
    total = np.convolve(np.where(present, y, 0.0), kernel, mode="same")
    count = np.convolve(present.astype(float), kernel, mode="same")
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _fetch(metric, keep):
    """Return {config: [(steps, values), ...]}, one entry per seed."""
    api = wandb.Api()
    groups = defaultdict(list)
    for run in api.runs(f"{api.default_entity}/{PROJECT}"):
        if not keep(run.name):
            continue
        # x is the training step, not wandb's own counter: the latter runs ~9x ahead and is what the UI calls
        # "Step" only by default, so plotting against it silently misreads every x position
        rows = run.history(keys=[metric, "train/global_step"], samples=100_000, pandas=False)
        points = sorted(
            (r["train/global_step"], r[metric])
            for r in rows
            if r.get(metric) is not None and r.get("train/global_step") is not None
        )
        if not points:
            continue
        steps, values = (np.array(c, dtype=float) for c in zip(*points))
        config = re.sub(r"\|s\d+$", "", run.name)
        groups[config].append((steps, values))
    return groups


def plot(metric, keep, title, out, ylabel=None):
    """Draw one line per configuration, shaded with +- 1 std across its seeds, and write `out`."""
    groups = _fetch(metric, keep)
    if not groups:
        raise SystemExit(f"no run in {PROJECT} matches the filter, or none logged {metric!r}")
    if len(groups) > len(PALETTE):
        raise SystemExit(f"{len(groups)} configurations for {len(PALETTE)} colours; narrow the filter")

    configs = sorted(groups)
    # The shared part of the names is the experiment, not a distinction between lines: state it once in the title.
    # Cut it back to a `|` so the labels start on a whole component rather than mid-token.
    prefix = os.path.commonprefix(configs) if len(configs) > 1 else ""
    prefix = prefix[: prefix.rfind("|") + 1]

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=130)
    for color, config in zip(PALETTE, configs):
        series = groups[config]
        lo = min(steps[0] for steps, _ in series)
        hi = max(steps[-1] for steps, _ in series)
        edges = np.linspace(lo, hi, BINS + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        stacked = np.vstack([_binned(steps, values, edges) for steps, values in series])
        with warnings.catch_warnings():  # a bin no seed reached is all-NaN, and plots as a gap
            warnings.simplefilter("ignore", RuntimeWarning)
            mean, std = np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)
        mean, std = _smooth(mean), _smooth(std)
        label = f"{config[len(prefix) :] or config} (n={len(series)})"
        ax.fill_between(centers, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(centers, mean, color=color, linewidth=1.6, label=label)

    caption = f"{prefix + '  ' if prefix else ''}mean +- 1 std over seeds, {SMOOTH}-bin moving average"
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel(ylabel or metric, fontsize=9)
    ax.set_title(f"{title}\n{caption}", fontsize=9.5, loc="left")
    ax.tick_params(labelsize=8)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out} ({len(configs)} configurations, band is +- 1 std over seeds)")
