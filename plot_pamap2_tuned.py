"""Create the PAMAP2 tuned plot from the completed experiment output.

This avoids retraining after a plotting-only error.
"""

import matplotlib.pyplot as plt
import numpy as np

fractions = ["10%", "20%", "40%"]
accuracies = {
    "10%": [57.72, 59.76, 48.51],
    "20%": [58.57, 55.75, 52.60],
    "40%": [59.51, 58.14, 49.96],
}
means = [np.mean(accuracies[fraction]) for fraction in fractions]
stds = [np.std(accuracies[fraction], ddof=1) for fraction in fractions]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(fractions, means, yerr=stds, capsize=5, color="#2ca02c", hatch="//", edgecolor="black")
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.7, f"{mean:.2f}%", ha="center", fontweight="bold")

ax.set_ylim(0, 100)
ax.set_xlabel("Training Data Percentage", fontweight="bold")
ax.set_ylabel("Test Accuracy (%)", fontweight="bold")
ax.set_title("PAMAP2 Tuned CNN: 512-Sample Windows")
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("pamap2_tuned_accuracy_plot.png", dpi=300, bbox_inches="tight")
print("PAMAP2 tuned means:", dict(zip(fractions, map(lambda value: f"{value:.2f}%", means))))
print("Plot saved as pamap2_tuned_accuracy_plot.png")
