"""Task accuracy across the first sweep, for the arms that actually train the policy.

Shares its colour assignment with the value_r2 plots: the lines are keyed by configuration name, sorted, so a
given arm keeps its colour from one figure to the next.
"""

import sys

from common import plot


plot(
    metric="train/rewards/accuracy_reward/mean",
    keep=lambda name: name.startswith("1_plr1e-6|"),
    title="Accuracy reward - policy lr 1e-6",
    ylabel="train/rewards/accuracy_reward/mean",
    out=sys.argv[1],
)
