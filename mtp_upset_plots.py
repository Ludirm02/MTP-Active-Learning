"""
4-way confused-sample overlap, per (dataset, fraction) -- 12 true Venn
diagrams (ellipse-based, 4 sets), using venny4py instead of UpSet.

Requires: pip install venny4py

Reads: confused_clusters/confused_<dataset>_<frac>_<metric>.csv (same
source as the UpSet version -- no retraining, no re-clustering). Valid
because all four metrics share the same sample_index space (see
mtp_uncertainty_scatter.py for that fix).

Output: venn_plots/venn_<dataset>_<frac>.png, one per (dataset, fraction).

NOTE: a true 4-circle Venn diagram is visually denser than the UpSet
version -- with 4 overlapping ellipses there are up to 15 distinct regions
to read, which is why UpSet plots are generally recommended for 4+ sets.
This script is here because you specifically want the traditional Venn
look; if it reads as too cluttered once you see it, the UpSet version
(mtp_upset_plots.py) is still there as the more readable alternative.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from venny4py.venny4py import venny4py

CONFUSED_DIR = Path("confused_clusters")
VENN_DIR = Path("venn_plots")
VENN_DIR.mkdir(exist_ok=True)

DATASETS = ["hhar", "ms", "pamap2", "ucihar"]
FRACTIONS = [10, 20, 40]
METRICS = ["entropy", "least_confidence", "margin", "bald"]
METRIC_LABELS = {
    "entropy": "Entropy",
    "least_confidence": "Least Confidence",
    "margin": "Margin",
    "bald": "BALD",
}

total_done = 0
for dataset in DATASETS:
    for frac in FRACTIONS:
        sets = {}
        missing = False
        for metric in METRICS:
            path = CONFUSED_DIR / f"confused_{dataset}_{frac}_{metric}.csv"
            if not path.exists():
                print(f"  Missing {path} -- skipping {dataset.upper()} {frac}%.")
                missing = True
                break
            df = pd.read_csv(path)
            sets[METRIC_LABELS[metric]] = set(df["sample_index"])
        if missing:
            continue

        fig, ax = plt.subplots(figsize=(7, 6.5))
        venny4py(sets=sets, asax=ax, size=4)
        ax.set_title(f"Confused-Sample Overlap — {dataset.upper()} {frac}% train data", fontsize=12)

        out_path = VENN_DIR / f"venn_{dataset}_{frac}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
        total_done += 1

print(f"\nDone. {total_done} Venn diagrams in {VENN_DIR}/.")