"""Leakage-safe 1D-CNN experiments for the CSV-based UCI HAR notebook.

Run from UCHIAR1.ipynb with: %run corrected_uchiar_experiment.py
"""

from pathlib import Path
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ---------- Configuration ----------
DATA_DIR = Path("al_mtp_data/ucihar/data")
WINDOW_SIZE = 128
# Non-overlapping windows prevent the same raw rows from appearing in both train and test.
WINDOW_STRIDE = 128
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.10
# These are percentages of the complete window dataset, not of a changing split.
TRAINING_FRACTIONS = (0.10, 0.20, 0.40)
SEEDS = (32, 46, 72)
MAX_EPOCHS = 75
PATIENCE = 12
BATCH_SIZE = 128


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only sensor columns and make occasional missing samples safe for a CNN."""
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] == 0:
        raise ValueError("No numeric sensor columns were found.")
    return numeric.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").fillna(0.0)


def make_windows_for_activity(values: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray]:
    """Create windows without crossing an activity-file boundary."""
    windows = [
        values[start : start + WINDOW_SIZE]
        for start in range(0, len(values) - WINDOW_SIZE + 1, WINDOW_STRIDE)
    ]
    if not windows:
        return (
            np.empty((0, WINDOW_SIZE, values.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    return np.asarray(windows, dtype=np.float32), np.full(len(windows), label, dtype=np.int64)


def scale_windows(train_x: np.ndarray, *other_sets: np.ndarray) -> list[np.ndarray]:
    """Fit feature scaling exclusively on the training windows."""
    scaler = StandardScaler()
    n_features = train_x.shape[-1]
    transformed = [
        scaler.fit_transform(train_x.reshape(-1, n_features)).reshape(train_x.shape).astype(np.float32)
    ]
    for values in other_sets:
        transformed.append(
            scaler.transform(values.reshape(-1, n_features)).reshape(values.shape).astype(np.float32)
        )
    return transformed


class BetterCNN(nn.Module):
    """Small regularised CNN with global pooling instead of an oversized flatten layer."""

    def __init__(self, n_features: int, n_classes: int) -> None:
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
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.30), nn.Linear(256, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x.permute(0, 2, 1)))


def run_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    excluded_names = {"std.csv", "mean.csv", "metadata.csv", "labels.csv"}
    csv_paths = sorted(p for p in DATA_DIR.glob("*.csv") if p.name.lower() not in excluded_names)
    if not csv_paths:
        raise FileNotFoundError(f"No activity CSV files found in {DATA_DIR.resolve()}")

    class_names = [path.stem for path in csv_paths]
    window_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    expected_features: int | None = None

    for label, path in enumerate(csv_paths):
        values = numeric_features(pd.read_csv(path)).to_numpy(dtype=np.float32)
        if expected_features is None:
            expected_features = values.shape[1]
        if values.shape[1] != expected_features:
            raise ValueError(f"{path.name} has {values.shape[1]} features; expected {expected_features}.")
        x_part, y_part = make_windows_for_activity(values, label)
        if len(x_part):
            window_parts.append(x_part)
            label_parts.append(y_part)
        print(f"{path.name}: {len(values)} rows -> {len(x_part)} windows")

    if not window_parts:
        raise ValueError("No complete windows were created. Lower WINDOW_SIZE or check the input CSV files.")
    x_windows = np.concatenate(window_parts)
    y_windows = np.concatenate(label_parts)
    class_counts = np.bincount(y_windows, minlength=len(class_names))
    if np.min(class_counts) < 3:
        raise ValueError(f"At least 3 windows per class are required; class counts are {class_counts.tolist()}.")

    print(f"\nDataset: {len(x_windows)} windows; window shape={x_windows.shape[1:]}; classes={class_names}")
    all_indices = np.arange(len(y_windows))
    train_val_idx, test_idx = train_test_split(
        all_indices, test_size=TEST_FRACTION, stratify=y_windows, random_state=2026
    )
    train_pool_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VALIDATION_FRACTION / (1 - TEST_FRACTION),
        stratify=y_windows[train_val_idx],
        random_state=2027,
    )
    print(f"Fixed split: train pool={len(train_pool_idx)}, validation={len(val_idx)}, test={len(test_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    def loader(x: np.ndarray, y: np.ndarray, shuffle: bool = False) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=device.type == "cuda")

    def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module):
        model.eval()
        losses, truth, predictions = [], [], []
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                logits = model(x_batch.to(device))
                losses.append(criterion(logits, y_batch.to(device)).item() * len(y_batch))
                truth.extend(y_batch.numpy())
                predictions.extend(logits.argmax(dim=1).cpu().numpy())
        return sum(losses) / len(data_loader.dataset), accuracy_score(truth, predictions), np.asarray(truth), np.asarray(predictions)

    def fit_and_score(train_idx: np.ndarray, seed: int):
        set_seed(seed)
        train_x, val_x, test_x = scale_windows(x_windows[train_idx], x_windows[val_idx], x_windows[test_idx])
        train_loader = loader(train_x, y_windows[train_idx], shuffle=True)
        val_loader = loader(val_x, y_windows[val_idx])
        test_loader = loader(test_x, y_windows[test_idx])
        model = BetterCNN(x_windows.shape[-1], len(class_names)).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
        best_loss, best_weights, stale = float("inf"), None, 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()  # Essential: validation puts the model into eval mode every epoch.
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(x_batch.to(device)), y_batch.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            val_loss, _, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)
            if val_loss < best_loss - 1e-4:
                best_loss, best_weights, stale = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break

        if best_weights is None:
            raise RuntimeError("Training did not produce a model checkpoint.")
        model.load_state_dict(best_weights)
        test_loss, test_accuracy, truth, predictions = evaluate(model, test_loader, criterion)
        return test_accuracy, test_loss, truth, predictions, epoch

    results = []
    final_prediction = None
    for fraction in TRAINING_FRACTIONS:
        requested = round(fraction * len(y_windows))
        if requested >= len(train_pool_idx):
            raise ValueError(f"{fraction:.0%} requires {requested} windows, but the fixed train pool has only {len(train_pool_idx)}.")
        for seed in SEEDS:
            chosen_idx, _ = train_test_split(
                train_pool_idx, train_size=requested, stratify=y_windows[train_pool_idx], random_state=seed
            )
            accuracy, loss, truth, predictions, epochs = fit_and_score(chosen_idx, seed)
            results.append(
                {
                    "training_fraction": fraction,
                    "seed": seed,
                    "train_windows": len(chosen_idx),
                    "epochs": epochs,
                    "test_loss": loss,
                    "test_accuracy": accuracy,
                }
            )
            print(f"train={fraction:.0%}, seed={seed}: test accuracy={accuracy:.2%} ({epochs} epochs)")
            if fraction == max(TRAINING_FRACTIONS) and seed == SEEDS[-1]:
                final_prediction = (truth, predictions)

    results_df = pd.DataFrame(results)
    summary = results_df.groupby("training_fraction", as_index=False).agg(
        mean_accuracy=("test_accuracy", "mean"),
        std_accuracy=("test_accuracy", "std"),
        mean_epochs=("epochs", "mean"),
    )
    summary["training_fraction"] = summary["training_fraction"].map(lambda value: f"{value:.0%}")
    print(results_df.to_string(index=False))
    print(summary.to_string(index=False))

    if final_prediction is not None:
        truth, predictions = final_prediction
        print(classification_report(truth, predictions, target_names=class_names, zero_division=0))
        ConfusionMatrixDisplay.from_predictions(
            truth, predictions, display_labels=class_names, xticks_rotation=45, cmap="Blues"
        )
        plt.tight_layout()
        plt.show()
    return results_df, summary


results_df, summary = run_experiments()


import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# UCHIAR RESULTS
# --------------------------------------------------

training_sizes = [10, 20, 40]
seeds = [32, 46, 72]

accuracy = {
    10: [96.13, 95.58, 93.92],
    20: [96.69, 96.69, 97.24],
    40: [97.24, 97.79, 97.79]
}

# Calculate mean and standard deviation
means = [
    np.mean(accuracy[size])
    for size in training_sizes
]

stds = [
    np.std(accuracy[size], ddof=1)
    for size in training_sizes
]

# --------------------------------------------------
# PLOT
# --------------------------------------------------

x = np.arange(len(training_sizes))
width = 0.22

plt.figure(figsize=(10, 6))

# Plot each random seed separately
for i, seed in enumerate(seeds):

    values = [
        accuracy[size][i]
        for size in training_sizes
    ]

    bars = plt.bar(
        x + (i - 1) * width,
        values,
        width,
        label=f"Seed {seed}"
    )

    # Accuracy value above each bar
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

# Mean accuracy + standard deviation
plt.errorbar(
    x,
    means,
    yerr=stds,
    fmt="ko",
    capsize=6,
    markersize=5,
    label="Mean ± Std"
)

# --------------------------------------------------
# LABELS
# --------------------------------------------------

plt.xlabel("Training Data Percentage", fontsize=13)
plt.ylabel("Accuracy (%)", fontsize=13)

plt.title(
    "UCHIAR Accuracy for Different Training Data Percentages",
    fontsize=14
)

plt.xticks(
    x,
    ["10%", "20%", "40%"]
)

plt.ylim(85, 100)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "uchiar_accuracy_all_seeds.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()