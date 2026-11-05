"""
Uncertainty scatter data for the guide's request, NOW INCLUDING BALD in the
same CSV as entropy/LC/margin, computed from the SAME trained model on the
SAME unlabeled pool.

WHY THIS CHANGED FROM THE EARLIER VERSION:
------------------------------------------------------------------------
BALD was previously computed by a separate script that reconstructed
mtp_random_split.py's TEST set and loaded a separately-trained checkpoint.
That meant BALD's sample_index did NOT refer to the same physical window as
entropy/LC/margin's sample_index for the "same" dataset/fraction -- making
any 4-way intersection across metrics invalid. This version computes BALD
right here, on the exact same pool_x used for entropy/LC/margin, with the
exact same model instance already in memory -- so all four metrics now
share one sample_index space and can be validly intersected.

For each (dataset, fraction), this script:
  1. Splits raw signal 3 ways per activity: held-out TEST (untouched),
     LABELED (at the target fraction), UNLABELED POOL (the rest). Leak-safe
     contiguous-block splitting, same as mtp_random_split.py.
  2. Trains ONE model on LABELED.
  3. Single forward pass (dropout off) on the UNLABELED POOL -> entropy,
     least confidence, margin.
  4. MC-Dropout (N stochastic passes, dropout kept active, BatchNorm fixed
     at eval) on the SAME UNLABELED POOL, SAME model -> BALD.
  5. Saves all four metrics to one CSV:
     uncertainty_csv/uncertainty_<dataset>_<fraction>pct_seed<seed>.csv
     columns: sample_index, entropy, least_confidence, margin, bald

6 seeds per (dataset, fraction); a later cleanup step keeps only the
best-accuracy seed's file (same as before -- unchanged from prior version).
"""

import copy
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
MC_PASSES = 30  # stochastic forward passes for BALD

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
    """Single forward pass, dropout OFF -- used for entropy/LC/margin."""
    pool_x_n = normalize_per_window(pool_x)
    loader = DataLoader(TensorDataset(torch.from_numpy(pool_x_n)), batch_size=BATCH_SIZE)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            all_probs.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def enable_dropout_only(model):
    """Keep BatchNorm in eval (fixed running stats) but turn Dropout back
    to train mode -- the standard MC-Dropout setup, so BALD isn't polluted
    by BatchNorm using batch statistics instead of its learned running ones."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def mc_dropout_probabilities(model, pool_x, n_passes):
    """Returns shape (n_passes, n_samples, n_classes), scored on the SAME
    pool_x used for entropy/LC/margin above, with the SAME model."""
    pool_x_n = normalize_per_window(pool_x)
    loader = DataLoader(TensorDataset(torch.from_numpy(pool_x_n)), batch_size=BATCH_SIZE, shuffle=False)
    enable_dropout_only(model)
    all_passes = []
    with torch.no_grad():
        for _ in range(n_passes):
            pass_probs = []
            for (xb,) in loader:
                probs = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
                pass_probs.append(probs)
            all_passes.append(np.concatenate(pass_probs, axis=0))
    return np.stack(all_passes, axis=0)


def entropy_from_probabilities(probs):
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)


def compute_bald(mc_probs):
    """BALD(x) = H(mean_t p_t) - mean_t H(p_t). Higher = more epistemic
    uncertainty (the model's passes disagree with each other)."""
    mean_probs = np.mean(mc_probs, axis=0)
    predictive_entropy = entropy_from_probabilities(mean_probs)
    expected_entropy = np.mean(entropy_from_probabilities(mc_probs), axis=0)
    return np.maximum(predictive_entropy - expected_entropy, 0.0)


def save_csv(dataset_name, fraction, seed, entropy_vals, lc_vals, margin_vals, bald_vals):
    fname = CSV_DIR / f"uncertainty_{dataset_name}_{int(fraction*100)}pct_seed{seed}.csv"
    with open(fname, "w", newline="") as f:
        f.write("sample_index,entropy,least_confidence,margin,bald\n")
        for i, (e, lc, m, b) in enumerate(zip(entropy_vals, lc_vals, margin_vals, bald_vals)):
            f.write(f"{i},{e},{lc},{m},{b}\n")
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

        for seed in SEEDS:
            labeled_x_list, labeled_y_list = [], []
            pool_x_list, pool_y_list = [], []

            for label, values in raw_by_activity:
                _, _, x_rest, y_rest = random_contiguous_split(values, label, TEST_FRACTION, window_size, seed=seed + label)
                labeled_fraction_of_rest = min(fraction / (1 - TEST_FRACTION), 0.95)
                n_rest = len(y_rest)
                n_labeled = max(1, int(round(labeled_fraction_of_rest * n_rest)))
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

            # Single pass (dropout off) -- entropy, LC, margin.
            probs = score_pool(model, pool_x)
            entropy_vals = -np.sum(probs * np.log(probs + 1e-12), axis=1)
            lc_vals = 1 - np.max(probs, axis=1)
            sorted_p = np.sort(probs, axis=1)
            margin_vals = sorted_p[:, -1] - sorted_p[:, -2]

            # MC-Dropout on the SAME pool, SAME model -- BALD.
            mc_probs = mc_dropout_probabilities(model, pool_x, MC_PASSES)
            bald_vals = compute_bald(mc_probs)

            csv_path = save_csv(dataset_name, fraction, seed, entropy_vals, lc_vals, margin_vals, bald_vals)
            print(
                f"  seed {seed}: labeled={len(labeled_y)}, pool={len(pool_y)}, "
                f"mean bald={bald_vals.mean():.4f}, saved {csv_path}"
            )

print("\nAll CSVs (entropy, least_confidence, margin, bald in one file each) complete.")
print("Run mtp_cleanup_uncertainty_csv.py next to keep only the best-seed file per (dataset, fraction).")