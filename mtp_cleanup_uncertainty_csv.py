"""
Full K-Means clustering analysis for entropy / least confidence / margin /
BALD, across all 4 datasets x 3 training fractions. Zero retraining -- reads
only uncertainty_csv/uncertainty_<dataset>_<frac>pct_seed<seed>.csv, which
now contains all four metrics (entropy, least_confidence, margin, bald) for
the SAME samples -- BALD was moved into this same file by the updated
mtp_uncertainty_scatter.py, so set-level comparisons across all four metrics
are now valid (no more separate bald_*.csv with a different sample space).

For each (dataset, fraction, metric):
  1. Sweep K = 2..8, compute the silhouette score for each K.
  2. Pick the K with the highest silhouette score.
  3. Cluster labels at that K partition ALL samples into K groups.
  4. Identify the most-confused cluster: highest mean value for
     entropy/least_confidence/bald, LOWEST mean value for margin.
  5. Save that cluster's rows to confused_clusters/confused_<dataset>_<frac>_<metric>.csv
  6. Save a silhouette-vs-K plot to silhouette_scores/silhouette_<dataset>_<frac>_<metric>.png
  7. Save a full cluster scatter plot (ALL samples, unrounded values,
     colored by cluster 0..K-1) to cluster_scatter_plots/clusters_<dataset>_<frac>_<metric>.png
     x-axis = Sample Index, y-axis = the metric's raw value.

Best seed per (dataset, fraction) is read from models/*.pt checkpoint
filenames (already selected by mtp_random_split.py) -- no accuracy is
recomputed here.

Output counts: 4 datasets x 3 fractions x 4 metrics = 48 of each
(confused CSVs, silhouette plots, cluster scatter plots).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

MODEL_DIR = Path("models")
CSV_DIR = Path("uncertainty_csv")
CONFUSED_DIR = Path("confused_clusters")
SILHOUETTE_DIR = Path("silhouette_scores")
SCATTER_DIR = Path("cluster_scatter_plots")

for d in (CONFUSED_DIR, SILHOUETTE_DIR, SCATTER_DIR):
    d.mkdir(exist_ok=True)

K_RANGE = range(2, 9)  # 2 to 8 inclusive

# All four metrics now live in the SAME file -- only the column differs.
METRIC_CONFIG = {
    "entropy": {"column": "entropy", "higher_is_more_uncertain": True, "label": "Entropy"},
    "least_confidence": {"column": "least_confidence", "higher_is_more_uncertain": True, "label": "Least Confidence"},
    "margin": {"column": "margin", "higher_is_more_uncertain": False, "label": "Margin"},  # smaller = more uncertain
    "bald": {"column": "bald", "higher_is_more_uncertain": True, "label": "BALD"},
}

CLUSTER_COLORS = plt.cm.tab10.colors


def find_best_seeds():
    """Parses models/*.pt filenames to recover the best-accuracy seed per
    (dataset, fraction), as already chosen by mtp_random_split.py."""
    pattern = re.compile(r"^(?P<dataset>[a-z0-9]+)_frac(?P<frac>\d+)_seed(?P<seed>\d+)\.pt$")
    best_seeds = {}
    for ckpt in MODEL_DIR.glob("*.pt"):
        m = pattern.match(ckpt.name)
        if not m:
            print(f"  Skipping unrecognized checkpoint filename: {ckpt.name}")
            continue
        best_seeds[(m.group("dataset"), int(m.group("frac")))] = int(m.group("seed"))
    return best_seeds


def sweep_k(values):
    values_2d = values.reshape(-1, 1)
    k_list, sil_list = [], []
    best_k, best_score, best_labels = None, -1.0, None
    for k in K_RANGE:
        if k >= len(values):
            break
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(values_2d)
        score = silhouette_score(values_2d, labels)
        k_list.append(k)
        sil_list.append(score)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    return k_list, sil_list, best_k, best_labels


def plot_silhouette(dataset, frac, metric_label, k_list, sil_list, best_k, out_path):
    plt.figure(figsize=(6, 4.5))
    plt.plot(k_list, sil_list, marker="o")
    best_idx = k_list.index(best_k)
    plt.scatter([best_k], [sil_list[best_idx]], color="red", s=110, zorder=5,
                label=f"Selected K={best_k}")
    plt.xlabel("K (number of clusters)")
    plt.ylabel("Silhouette Score")
    plt.title(f"{metric_label} Silhouette Scores\n{dataset.upper()} {frac}% train data")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cluster_scatter(dataset, frac, metric_label, values, labels, best_k, out_path):
    plt.figure(figsize=(11, 5))
    idx_all = np.arange(len(values))
    for cluster_id in range(best_k):
        mask = labels == cluster_id
        plt.scatter(
            idx_all[mask], values[mask], s=4, alpha=0.4,
            color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
            label=f"Cluster {cluster_id} (n={mask.sum()})",
        )
    plt.xlabel("Sample Index")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} — K={best_k} clusters — {dataset.upper()} {frac}% train data")
    plt.legend(markerscale=3, fontsize=8, loc="upper center",
               ncol=min(best_k, 5), bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    best_seeds = find_best_seeds()
    if not best_seeds:
        print(f"No checkpoints found under {MODEL_DIR}/ -- nothing to do.")
        return

    total_done = 0
    for (dataset, frac), seed in sorted(best_seeds.items()):
        print(f"\n{dataset.upper()} {frac}% (seed {seed})")

        csv_path = CSV_DIR / f"uncertainty_{dataset}_{frac}pct_seed{seed}.csv"
        if not csv_path.exists():
            print(f"  Missing {csv_path} -- skipping this dataset/fraction entirely.")
            continue
        df = pd.read_csv(csv_path)

        for metric, cfg in METRIC_CONFIG.items():
            if cfg["column"] not in df.columns:
                print(f"  {metric}: column '{cfg['column']}' not in {csv_path.name} -- skipping "
                      f"(this file may be from before BALD was merged in -- rerun mtp_uncertainty_scatter.py).")
                continue

            values = df[cfg["column"]].to_numpy()
            k_list, sil_list, best_k, best_labels = sweep_k(values)
            if best_k is None:
                print(f"  {metric}: not enough samples to cluster, skipping.")
                continue

            cluster_means = pd.Series(values).groupby(best_labels).mean()
            confused_cluster = (
                cluster_means.idxmax() if cfg["higher_is_more_uncertain"] else cluster_means.idxmin()
            )
            confused_mask = best_labels == confused_cluster
            confused_path = CONFUSED_DIR / f"confused_{dataset}_{frac}_{metric}.csv"
            df[["sample_index", cfg["column"]]][confused_mask].to_csv(confused_path, index=False)

            sil_path = SILHOUETTE_DIR / f"silhouette_{dataset}_{frac}_{metric}.png"
            plot_silhouette(dataset, frac, cfg["label"], k_list, sil_list, best_k, sil_path)

            scatter_path = SCATTER_DIR / f"clusters_{dataset}_{frac}_{metric}.png"
            plot_cluster_scatter(dataset, frac, cfg["label"], values, best_labels, best_k, scatter_path)

            print(
                f"  {metric}: best K={best_k}, confused cluster #{confused_cluster} "
                f"({confused_mask.sum()}/{len(df)} samples) -> "
                f"{confused_path.name}, {sil_path.name}, {scatter_path.name}"
            )
            total_done += 1

    print(f"\nDone. {total_done} (dataset,fraction,metric) combinations processed.")
    print(f"  {CONFUSED_DIR}/   -> confused-cluster CSVs")
    print(f"  {SILHOUETTE_DIR}/ -> silhouette-vs-K plots")
    print(f"  {SCATTER_DIR}/    -> full cluster scatter plots")


if __name__ == "__main__":
    main()