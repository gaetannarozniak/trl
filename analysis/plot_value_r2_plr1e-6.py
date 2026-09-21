"""Critic explained variance across the first sweep, for the arms that actually train the policy."""

import sys

from common import plot


plot(
    metric="train/value_r2",
    keep=lambda name: name.startswith("1_plr1e-6|"),
    title="Critic explained variance - policy lr 1e-6",
    ylabel="train/value_r2",
    out=sys.argv[1],
)
