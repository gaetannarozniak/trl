"""Critic explained variance per experiment, aggregated over seeds and relaunched runs.

One panel per experiment, each with the `0_` control drawn in grey for reference. Runs are grouped by name with
the trailing `|s<seed>` stripped, so seeds and any relaunch of the same configuration average together.
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
from common import BINS, _binned, _fetch, _smooth

import matplotlib.pyplot as plt  # noqa: E402  (common sets the Agg backend on import)


MAX_STEP = 1000
STEM = re.compile(r"^[0-4]_plr0\|residual\|?")
# one hue per experiment, ramped light to dark along that experiment's ordered hyperparameter
PANELS = {
    "1_": ("1 — critic read layer", "#0072B2"),
    "2_": ("2 — TD(lambda) value target", "#D55E00"),
    "3_": ("3 — privileged prompt and probe tokens", "#009E73"),
    "4_": ("4 — privileged prompt x TD(lambda)", "#CC79A7"),
}


def ramp(base, n):
    """`n` tints of one hue, palest first, so an ordered hyperparameter reads as a gradient.

    The floor is well above white: any paler and the curve is lost against the translucent std bands.
    """
    rgb = [int(base[i : i + 2], 16) for i in (1, 3, 5)]
    return [
        "#%02x%02x%02x" % tuple(round(255 - (255 - c) * (0.55 + 0.45 * i / max(n - 1, 1))) for c in rgb)
        for i in range(n)
    ]


def order_key(label):
    """Sort by the hyperparameter the label carries, not alphabetically: layer8 precedes layer12."""
    numbers = tuple(float(x) for x in re.findall(r"\d+\.?\d*", label))
    return (numbers, len(label), label)

groups = _fetch("train/value_ev", lambda name: re.match(r"[0-4]_", name) is not None)
edges = np.linspace(0, MAX_STEP, BINS + 1)
centers = (edges[:-1] + edges[1:]) / 2


def aggregate(series):
    """Mean and std over the runs of one configuration, binned onto the shared grid and smoothed."""
    stacked = np.vstack([_binned(steps, values, edges) for steps, values in series])
    with warnings.catch_warnings():  # a bin no run reached is all-NaN and plots as a gap
        warnings.simplefilter("ignore", RuntimeWarning)
        return _smooth(np.nanmean(stacked, axis=0)), _smooth(np.nanstd(stacked, axis=0))


control = [s for name, s in groups.items() if name.startswith("0_")]
control = aggregate([run for runs in control for run in runs]) if control else None

fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), dpi=130, sharex=True)
for ax, (prefix, (title, hue)) in zip(axes.flat, PANELS.items()):
    configs = sorted((name for name in groups if name.startswith(prefix)), key=lambda n: order_key(STEM.sub("", n)))
    # every band at zorder 1 and every line at 3, so a later config's band cannot bury an earlier config's line
    if control is not None:
        mean, std = control
        ax.fill_between(centers, mean - std, mean + std, color="#9ca3af", alpha=0.10, linewidth=0, zorder=1)
        ax.plot(centers, mean, color="#6b7280", linewidth=1.4, linestyle="--", label="control (0_)", zorder=2)
    for color, name in zip(ramp(hue, len(configs)), configs):
        mean, std = aggregate(groups[name])
        ax.fill_between(centers, mean - std, mean + std, color=color, alpha=0.10, linewidth=0, zorder=1)
        ax.plot(
            centers,
            mean,
            color=color,
            linewidth=1.8,
            label=f"{STEM.sub('', name) or name} (n={len(groups[name])})",
            zorder=3,
        )
    if not configs:
        ax.text(0.5, 0.5, "no runs yet", transform=ax.transAxes, ha="center", color="#9ca3af")
    ax.axhline(0.0, color="#d1d5db", linewidth=0.8)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xlim(0, MAX_STEP)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7)

for ax in axes[1]:
    ax.set_xlabel("step", fontsize=9)
for ax in axes[:, 0]:
    ax.set_ylabel("train/value_ev", fontsize=9)
fig.suptitle(
    f"Critic explained variance, mean ± 1 std over seeds and relaunches, {BINS}-bin grid to step {MAX_STEP}",
    fontsize=10,
)
fig.tight_layout()

out = sys.argv[1]
Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out} ({sum(len(v) for v in groups.values())} runs across {len(groups)} configurations)")
