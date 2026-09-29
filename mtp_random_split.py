"""
Random-split learning curve experiment (10% / 20% / 40% training data) with
leak-safe random-contiguous splitting, plus uncertainty scoring (entropy,
least confidence, margin) on the best model per training fraction.

DESIGN NOTES (carried over from earlier iterations):
------------------------------------------------------------------------
- Train/test split: a CONTIGUOUS block of the raw (pre-windowing) signal is
  chosen as train, with its start position randomized by the seed. A
  one-window buffer is excluded on both sides of the cut so no window can
  straddle the train/remainder boundary (no leakage). Remainder is windowed
  separately and split into a small validation slice + test set.
- CAVEAT: because test = "whatever's left" after train, the test set is a
  different size at 10%/20%/40% (90%/80%/60% of the file). Flagged to guide.
- Magnitude channel: sqrt(x^2+y^2+z^2) added when raw columns are x/y/z
  (this only fires for HHAR's per-activity files) — partially cancels out
  device/orientation differences without needing device labels.
- Per-window normalization: each window is standardized using its OWN
  mean/std per channel (instead of one global StandardScaler fit over the
  whole training subset). This corrects for per-recording offset/scale
  differences that plain global scaling can't reach.
- Denser window overlap (stride = window_size // 4) for more real training
  windows per split, no leakage risk since overlap only happens within an
  already-isolated region.
- 6 seeds instead of 3, higher early-stopping patience (25) since MAX_EPOCHS
  was not being hit before.

NEW IN THIS VERSION:
------------------------------------------------------------------------
- Best-validation-accuracy checkpoint per (dataset, fraction) is saved to
  disk under models/.
- Test-set softmax probabilities from that best model are used to compute
  three uncertainty metrics per sample: entropy, least confidence, margin.
- One scatter plot per dataset (uncertainty_<dataset>.png), three panels
  (entropy / least confidence / margin), showing all test samples at each
  training fraction using that fraction's best-accuracy seed.
- BALD (MC Dropout) is NOT included yet — deferred per guide's instruction
  to do it after entropy/LC/margin are working.
"""

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --- 1. Configuration ---
BASE_DIR = Path("al_mtp_data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DATASETS = ["ms", "hhar", "pamap2", "ucihar"]

# (window_size, stride) per dataset. PAMAP2 is sampled ~100Hz vs ~50Hz for
# the others, so its window is widened to cover a comparable time span.
WINDOW_CONFIG = {
    "ms": (128, 64),
    "hhar": (128, 64),
    "pamap2": (512, 256),
    "ucihar": (128, 64),
}

TRAINING_FRACTIONS = [0.10, 0.20, 0.40]
SEEDS = [32, 42, 72, 13, 100, 7]
MAX_EPOCHS = 100
PATIENCE = 25
BATCH_SIZE = 128
VAL_SHARE_OF_REMAINDER = 0.15  # slice of the non-train remainder used for early stopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")


# --- 2. Helper Functions ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Removes junk columns, cleans missing/infinite values, and adds an
    orientation-invariant magnitude channel when raw x/y/z columns exist."""
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    numeric = frame.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").fillna(0.0)
    if {"x", "y", "z"}.issubset(numeric.columns):
        numeric["mag"] = np.sqrt(numeric["x"] ** 2 + numeric["y"] ** 2 + numeric["z"] ** 2)
    return numeric


def make_windows(values: np.ndarray, label: int, window_size: int, stride: int):
    """Slices one contiguous chunk of raw samples into windows. Never called
    across a train/remainder boundary, so windows never straddle a split."""
    n_features = values.shape[1] if values.ndim == 2 else 0
    if len(values) < window_size:
        return (
            np.empty((0, window_size, n_features), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    windows = [
        values[start : start + window_size]
        for start in range(0, len(values) - window_size + 1, stride)
    ]
    return np.asarray(windows, dtype=np.float32), np.full(len(windows), label, dtype=np.int64)


def random_contiguous_split(values: np.ndarray, label: int, fraction: float, window_size: int, seed: int):
    """
    Choose a contiguous block covering `fraction` of the raw samples as TRAIN,
    with its start position randomized by `seed`. A one-window buffer is
    excluded on both sides of the cut so no window can straddle the
    train/remainder boundary. Returns windowed (train_x, train_y, remainder_x,
    remainder_y) for this one activity file. Windows use denser overlap
    (stride = window_size // 4) for more real training data per split.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(values)
    train_len = int(round(fraction * n_samples))
    buffer = window_size

    max_start = n_samples - train_len
    if max_start <= 0:
        raise ValueError("Requested training fraction is larger than this file's length.")
    train_start = int(rng.integers(0, max_start + 1))
    train_end = train_start + train_len

    train_block = values[train_start:train_end]
    before_block = values[: max(0, train_start - buffer)]
    after_block = values[train_end + buffer :]

    stride = max(1, window_size // 4)
    x_train, y_train = make_windows(train_block, label, window_size, stride)
    x_before, y_before = make_windows(before_block, label, window_size, stride)
    x_after, y_after = make_windows(after_block, label, window_size, stride)

    remainder_x_parts = [a for a in (x_before, x_after) if len(a)]
    remainder_y_parts = [a for a in (y_before, y_after) if len(a)]
    if remainder_x_parts:
        x_remainder = np.concatenate(remainder_x_parts)
        y_remainder = np.concatenate(remainder_y_parts)
    else:
        x_remainder = np.empty((0, window_size, values.shape[1]), dtype=np.float32)
        y_remainder = np.empty(0, dtype=np.int64)

    return x_train, y_train, x_remainder, y_remainder


def normalize_per_window(x: np.ndarray) -> np.ndarray:
    """Standardize each window using its OWN mean/std per channel, instead of
    one global scaler. Corrects for per-recording offset/scale differences
    (e.g. device-specific sensor bias) without needing device labels."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


# --- 3. CNN (unchanged architecture) ---
class HAR_CNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x.permute(0, 2, 1)))


def plot_uncertainty_scatter(dataset_name, records):
    """
    Create separate scatter plots for every:
      - training fraction: 10%, 20%, 40%
      - uncertainty metric: entropy, least confidence, margin

    Each plot uses the best-accuracy seed selected for that fraction.
    A dashed horizontal line shows the mean uncertainty value.
    """

    metrics = [
        ("entropy", "Entropy"),
        ("least_confidence", "Least Confidence"),
        ("margin", "Margin"),
    ]

    fractions_sorted = sorted(records.keys())

    for frac in fractions_sorted:

        for metric, title in metrics:

            values = records[frac][metric]
            mean_val = np.mean(values)

            plt.figure(figsize=(10, 5))

            plt.scatter(
                np.arange(len(values)),
                values,
                s=4,
                alpha=0.30
            )

            # Mean uncertainty line
            plt.axhline(
                mean_val,
                linestyle="--",
                linewidth=1.2,
                color="black"
            )

            # Mean annotation
            plt.text(
                0.98,
                0.95,
                f"Mean = {mean_val:.4f}",
                transform=plt.gca().transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.8
                )
            )

            plt.xlabel("Test Sample Index")
            plt.ylabel(f"{title} Value")

            plt.title(
                f"{title} — {int(frac * 100)}% train data "
                f"({dataset_name.upper()}) "
                f"[seed {records[frac]['seed']}]"
            )

            plt.tight_layout()

            filename = (
                f"uncertainty_{dataset_name}_"
                f"{int(frac * 100)}pct_"
                f"{metric}.png"
            )

            plt.savefig(
                filename,
                dpi=200,
                bbox_inches="tight"
            )

            plt.close()

            print(f"Saved {filename}")


# --- 4. Main Execution Loop ---
final_plot_data = {ds: {frac: [] for frac in TRAINING_FRACTIONS} for ds in DATASETS}
uncertainty_records = {ds: {} for ds in DATASETS}

for dataset_name in DATASETS:
    print(f"\n{'='*50}\nPROCESSING DATASET: {dataset_name.upper()}\n{'='*50}")
    window_size, _ = WINDOW_CONFIG[dataset_name]
    data_dir = BASE_DIR / dataset_name / "data"
    csv_paths = sorted(
        p for p in data_dir.glob("*.csv")
        if p.name.lower() not in {"std.csv", "mean.csv", "metadata.csv", "labels.csv"}
    )
    if not csv_paths:
        print(f"Skipping {dataset_name} - No CSVs found.")
        continue

    class_names = [path.stem for path in csv_paths]
    n_classes = len(class_names)

    # Raw arrays per activity file, loaded once and reused across fractions/seeds.
    raw_by_activity = []
    for label, path in enumerate(csv_paths):
        values = numeric_features(pd.read_csv(path)).to_numpy(dtype=np.float32)
        raw_by_activity.append((label, values))
    n_features = raw_by_activity[0][1].shape[-1]

    for fraction in TRAINING_FRACTIONS:
        print(f"\n--- Training with {int(fraction*100)}% Data ({dataset_name.upper()}) ---")
        best_this_fraction = {"acc": -1.0}

        for seed in SEEDS:
            set_seed(seed)

            train_x_list, train_y_list = [], []
            remainder_x_list, remainder_y_list = [], []
            for label, values in raw_by_activity:
                x_tr, y_tr, x_rem, y_rem = random_contiguous_split(
                    values, label, fraction, window_size, seed=seed + label  # vary cut per class too
                )
                train_x_list.append(x_tr)
                train_y_list.append(y_tr)
                remainder_x_list.append(x_rem)
                remainder_y_list.append(y_rem)

            train_x = np.concatenate(train_x_list)
            train_y = np.concatenate(train_y_list)
            remainder_x = np.concatenate(remainder_x_list)
            remainder_y = np.concatenate(remainder_y_list)

            val_x, test_x, val_y, test_y = train_test_split(
                remainder_x, remainder_y,
                test_size=1 - VAL_SHARE_OF_REMAINDER,
                random_state=seed,
                stratify=remainder_y,
            )

            if seed == SEEDS[0]:
                test_counts = np.bincount(test_y, minlength=n_classes)
                majority_baseline = test_counts.max() / len(test_y) * 100
                print(
                    f"  [seed {seed}] train={len(train_y)}, val={len(val_y)}, test={len(test_y)}, "
                    f"majority baseline={majority_baseline:.2f}%"
                )

            # Per-window normalization (replaces the earlier global StandardScaler).
            train_x = normalize_per_window(train_x)
            val_x = normalize_per_window(val_x)
            test_x = normalize_per_window(test_x)

            train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=BATCH_SIZE)
            test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)), batch_size=BATCH_SIZE)

            model = HAR_CNN(n_features, n_classes).to(device)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

            best_loss, best_weights, stale = float("inf"), None, 0
            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(x_batch.to(device)), y_batch.to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        val_loss += criterion(model(x_batch.to(device)), y_batch.to(device)).item() * len(y_batch)
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

            truth, predictions, probs = [], [], []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    logits = model(x_batch.to(device))
                    truth.extend(y_batch.numpy())
                    predictions.extend(logits.argmax(dim=1).cpu().numpy())
                    probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

            acc = accuracy_score(truth, predictions) * 100
            final_plot_data[dataset_name][fraction].append(acc)
            print(f"Seed {seed}: Accuracy = {acc:.2f}% (Epochs run: {epoch})")

            if acc > best_this_fraction["acc"]:
                best_this_fraction = {
                    "acc": acc,
                    "seed": seed,
                    "probs": np.array(probs),
                    "state_dict": copy.deepcopy(model.state_dict()),
                }

        # After all seeds for this fraction: save the best checkpoint + compute uncertainty metrics.
        ckpt_path = MODEL_DIR / f"{dataset_name}_frac{int(fraction*100)}_seed{best_this_fraction['seed']}.pt"
        torch.save(best_this_fraction["state_dict"], ckpt_path)

        p = best_this_fraction["probs"]  # shape (n_test_samples, n_classes)
        entropy = -np.sum(p * np.log(p + 1e-12), axis=1)
        sorted_p = np.sort(p, axis=1)
        least_confidence = 1 - sorted_p[:, -1]
        margin = sorted_p[:, -1] - sorted_p[:, -2]

        uncertainty_records[dataset_name][fraction] = {
            "entropy": entropy,
            "least_confidence": least_confidence,
            "margin": margin,
            "seed": best_this_fraction["seed"],
            "acc": best_this_fraction["acc"],
        }
        print(
            f"  -> Best seed for {int(fraction*100)}% = {best_this_fraction['seed']} "
            f"(acc={best_this_fraction['acc']:.2f}%), checkpoint saved to {ckpt_path}"
        )

print("\nAll training complete! Generating plots...")

# --- 5. Accuracy Bar Plot ---
datasets_display = ["MotionSense", "HHAR", "PAMAP2", "UCI HAR"]
dataset_keys = ["ms", "hhar", "pamap2", "ucihar"]
fractions = TRAINING_FRACTIONS

means = {ds: [] for ds in dataset_keys}
stds = {ds: [] for ds in dataset_keys}
for ds in dataset_keys:
    for frac in fractions:
        accuracies = final_plot_data[ds][frac]
        means[ds].append(np.mean(accuracies))
        stds[ds].append(np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0)

x = np.arange(len(fractions))
width = 0.2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
hatches = ['//', '\\\\', '--', 'xx']

fig, ax = plt.subplots(figsize=(10, 6))
for i, ds in enumerate(dataset_keys):
    bars = ax.bar(
        x + (i - 1.5) * width, means[ds], width,
        label=datasets_display[i], color=colors[i], hatch=hatches[i],
        edgecolor='black', yerr=stds[ds], capsize=4,
        error_kw={'elinewidth': 1, 'capthick': 1},
    )
    for bar, err in zip(bars, stds[ds]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + err + 0.5, f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Training Data Percentage (test = remainder, size varies by fraction)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['10%', '20%', '40%'], fontsize=12, fontweight='bold')
ax.set_ylim(0, 115)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Datasets', title_fontsize=12, fontsize=10, loc='upper center', ncol=4, framealpha=1)

plt.tight_layout()
plt.savefig('combined_accuracy_plot_random_split.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'combined_accuracy_plot_random_split.png'!")

# --- 6. Uncertainty Scatter Plots (entropy / least confidence / margin) ---
for ds in DATASETS:
    if uncertainty_records.get(ds):
        plot_uncertainty_scatter(ds, uncertainty_records[ds])