"""Critic explained variance across the first sweep, with the policy frozen: pure policy evaluation."""

import sys

from common import plot


plot(
    metric="train/value_r2",
    keep=lambda name: name.startswith("1_plr0|"),
    title="Critic explained variance - frozen policy (lr 0)",
    ylabel="train/value_r2",
    out=sys.argv[1],
)
