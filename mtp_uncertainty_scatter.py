"""
Uncertainty scatter plots for the guide's exact request:
  "For example, for entropy, 10% training data, for HHAR ... plot the scatter
   plot (with values being the Entropy value itself). Similarly for other
   metrics, training sizes and datasets."

SEPARATE deliverable from mtp_active_learning.py's AL-round comparison.
Does not touch the AL loop, K-Means selection, or retraining -- K-Means /
Silhouette-based K selection belongs only to the active-learning candidate
selection stage, not to this scatter-plot deliverable.

NESTING NOTE: for a given seed, the RNG used to pick the labeled subset is
seeded by (seed, activity label) only -- NOT by the fraction. That means the
same shuffled order is reused across 10%/20%/40%, so labeled_10% is
automatically a PREFIX of labeled_20%, which is a prefix of labeled_40%.
This gives a clean "as training data grows, how does uncertainty change"
comparison per seed, without needing any extra code for it.

For each (dataset, fraction), this script:
  1. Runs all 6 seeds: splits raw signal into held-out TEST (untouched),
     LABELED (nested prefix at the target fraction), and UNLABELED POOL.
  2. Trains one model per seed on LABELED.
  3. Scores the UNLABELED POOL (never the test set) with that seed's model.
  4. Computes entropy, least confidence, margin per pool sample.
  5. Saves the raw values to CSV (sample_index, entropy, least_confidence,
     margin) per (dataset, fraction, seed) -- 4*3*6 = 72 CSV files.
  6. Produces ONE combined scatter plot per (dataset, fraction, metric) with
     all 6 seeds overlaid (distinct colors, legend) -- 4*3*3 = 36 PNGs total,
     not 216, so the deliverable stays manageable while keeping all seed
     variation visible.

COST: 4 datasets x 3 fractions x 6 seeds = 72 model trainings.
"""

import copy
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_DIR = Path("al_mtp_data")
CSV_DIR = Path("uncertainty_csv")
CSV_DIR.mkdir(exist_ok=True)

DATASETS = ["ms", "hhar", "pamap2", "ucihar"]
WINDOW_CONFIG = {
    "ms": (128, 64),
    "hhar": (128, 64),
    "pamap2": (512, 256),
    "ucihar": (128, 64),
}

TEST_FRACTION = 0.20
LABELED_FRACTIONS = [0.10, 0.20, 0.40]
SEEDS = [32, 42, 72, 13, 100, 7]

MAX_EPOCHS = 60
PATIENCE = 15
BATCH_SIZE = 128
VAL_SHARE_OF_LABELED = 0.15

SEED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    numeric = frame.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").fillna(0.0)
    if {"x", "y", "z"}.issubset(numeric.columns):
        numeric["mag"] = np.sqrt(numeric["x"] ** 2 + numeric["y"] ** 2 + numeric["z"] ** 2)
    return numeric


def make_windows(values, label, window_size, stride):
    n_features = values.shape[1] if values.ndim == 2 else 0
    if len(values) < window_size:
        return np.empty((0, window_size, n_features), dtype=np.float32), np.empty(0, dtype=np.int64)
    windows = [values[s : s + window_size] for s in range(0, len(values) - window_size + 1, stride)]
    return np.asarray(windows, dtype=np.float32), np.full(len(windows), label, dtype=np.int64)


def random_contiguous_split(values, label, fraction, window_size, seed):
    rng = np.random.default_rng(seed)
    n_samples = len(values)
    block_len = int(round(fraction * n_samples))
    buffer = window_size
    max_start = n_samples - block_len
    if max_start <= 0:
        raise ValueError("Requested fraction larger than file length.")
    block_start = int(rng.integers(0, max_start + 1))
    block_end = block_start + block_len

    block = values[block_start:block_end]
    before = values[: max(0, block_start - buffer)]
    after = values[block_end + buffer :]

    stride = max(1, window_size // 4)
    x_block, y_block = make_windows(block, label, window_size, stride)
    x_before, y_before = make_windows(before, label, window_size, stride)
    x_after, y_after = make_windows(after, label, window_size, stride)

    rem_x = [a for a in (x_before, x_after) if len(a)]
    rem_y = [a for a in (y_before, y_after) if len(a)]
    if rem_x:
        return x_block, y_block, np.concatenate(rem_x), np.concatenate(rem_y)
    return x_block, y_block, np.empty((0, window_size, values.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)


def normalize_per_window(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


class HAR_CNN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.30), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x.permute(0, 2, 1)))


def train_model(train_x, train_y, n_features, n_classes, seed):
    set_seed(seed)
    tr_x, val_x, tr_y, val_y = train_test_split(
        train_x, train_y, test_size=VAL_SHARE_OF_LABELED, random_state=seed,
        stratify=train_y if len(np.unique(train_y)) > 1 else None,
    )
    tr_x_n, val_x_n = normalize_per_window(tr_x), normalize_per_window(val_x)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(tr_x_n), torch.from_numpy(tr_y)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x_n), torch.from_numpy(val_y)), batch_size=BATCH_SIZE)

    model = HAR_CNN(n_features, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_loss, best_weights, stale = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb.to(device)), yb.to(device)).item() * len(yb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        if val_loss < best_loss - 0.001:
            best_loss, best_weights, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return model


def score_pool(model, pool_x):
    pool_x_n = normalize_per_window(pool_x)
    loader = DataLoader(TensorDataset(torch.from_numpy(pool_x_n)), batch_size=BATCH_SIZE)
    all_probs = []
    with torch.no_grad():
        for (xb,) in loader:
            all_probs.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def save_csv(dataset_name, fraction, seed, entropy_vals, lc_vals, margin_vals):
    fname = CSV_DIR / f"uncertainty_{dataset_name}_{int(fraction*100)}pct_seed{seed}.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "entropy", "least_confidence", "margin"])
        for i, (e, lc, m) in enumerate(zip(entropy_vals, lc_vals, margin_vals)):
            writer.writerow([i, e, lc, m])
    return fname


# --- Main loop ---
for dataset_name in DATASETS:
    print(f"\n{'='*50}\nDATASET: {dataset_name.upper()}\n{'='*50}")
    window_size, _ = WINDOW_CONFIG[dataset_name]
    data_dir = BASE_DIR / dataset_name / "data"
    csv_paths = sorted(
        p for p in data_dir.glob("*.csv")
        if p.name.lower() not in {"std.csv", "mean.csv", "metadata.csv", "labels.csv"}
    )
    if not csv_paths:
        print(f"Skipping {dataset_name} - no CSVs found.")
        continue
    n_classes = len(csv_paths)

    raw_by_activity = []
    for label, path in enumerate(csv_paths):
        values = numeric_features(pd.read_csv(path)).to_numpy(dtype=np.float32)
        raw_by_activity.append((label, values))
    n_features = raw_by_activity[0][1].shape[-1]

    for fraction in LABELED_FRACTIONS:
        print(f"\n-- {int(fraction*100)}% labeled --")
        # Collect all seeds' results here before plotting, so each metric's
        # plot for this (dataset, fraction) can overlay all 6 seeds at once.
        seed_results = {}

        for seed in SEEDS:
            labeled_x_list, labeled_y_list = [], []
            pool_x_list, pool_y_list = [], []

            for label, values in raw_by_activity:
                # Test block: seeded by (seed, label) only -- same across fractions.
                _, _, x_rest, y_rest = random_contiguous_split(values, label, TEST_FRACTION, window_size, seed=seed + label)

                labeled_fraction_of_rest = min(fraction / (1 - TEST_FRACTION), 0.95)
                n_rest = len(y_rest)
                n_labeled = max(1, int(round(labeled_fraction_of_rest * n_rest)))

                # Seeded by (seed, label) only -- NOT fraction -- so the same
                # permutation is reused at 10%/20%/40%, making labeled_10% a
                # prefix of labeled_20%, a prefix of labeled_40%, per seed.
                rng = np.random.default_rng(seed + label + 999)
                perm = rng.permutation(n_rest)
                labeled_idx, pool_idx = perm[:n_labeled], perm[n_labeled:]

                labeled_x_list.append(x_rest[labeled_idx])
                labeled_y_list.append(y_rest[labeled_idx])
                pool_x_list.append(x_rest[pool_idx])
                pool_y_list.append(y_rest[pool_idx])

            labeled_x = np.concatenate(labeled_x_list)
            labeled_y = np.concatenate(labeled_y_list)
            pool_x = np.concatenate(pool_x_list)
            pool_y = np.concatenate(pool_y_list)

            model = train_model(labeled_x, labeled_y, n_features, n_classes, seed=seed)
            probs = score_pool(model, pool_x)

            entropy_vals = -np.sum(probs * np.log(probs + 1e-12), axis=1)
            lc_vals = 1 - np.max(probs, axis=1)
            sorted_p = np.sort(probs, axis=1)
            margin_vals = sorted_p[:, -1] - sorted_p[:, -2]

            csv_path = save_csv(dataset_name, fraction, seed, entropy_vals, lc_vals, margin_vals)
            seed_results[seed] = {"entropy": entropy_vals, "least_confidence": lc_vals, "margin": margin_vals}
            print(f"  seed {seed}: labeled={len(labeled_y)}, pool={len(pool_y)}, saved {csv_path}")

        # One combined plot per metric, all 6 seeds overlaid.
        for metric_name in ["entropy", "least_confidence", "margin"]:
            plt.figure(figsize=(11, 5))
            for i, seed in enumerate(SEEDS):
                values = seed_results[seed][metric_name]
                plt.scatter(
                    np.arange(len(values)), values, s=4, alpha=0.3,
                    color=SEED_COLORS[i % len(SEED_COLORS)], label=f"seed {seed}",
                )
            plt.xlabel("Sample index (unlabeled pool)")
            plt.ylabel(metric_name.replace("_", " ").title())
            plt.title(
                f"{metric_name.replace('_', ' ').title()} — {dataset_name.upper()} "
                f"({int(fraction*100)}% training data, all 6 seeds)"
            )
            plt.legend(markerscale=3, fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.15))
            plt.tight_layout()
            fname = f"uncertainty_{dataset_name}_{int(fraction*100)}pct_{metric_name}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved {fname}")

print("\nAll 72 CSVs and 36 combined scatter plots complete.")