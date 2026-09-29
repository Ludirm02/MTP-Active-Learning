"""
BALD uncertainty scatter plots for the existing HAR experiment.

Purpose
-------
Compute ONLY BALD uncertainty for the SAME:
    - 4 datasets: MotionSense, HHAR, PAMAP2, UCI HAR
    - 3 training fractions: 10%, 20%, 40%
    - best-accuracy seed/checkpoint already produced by mtp_random_split.py

This script does NOT:
    - retrain any model
    - recompute entropy / least-confidence / margin
    - run K-Means
    - run active learning
    - change the existing uncertainty results

It loads the saved best-model checkpoints from:
    models/<dataset>_frac<10|20|40>_seed<S>.pt

It reconstructs the SAME test split used by mtp_random_split.py and
computes BALD with MC Dropout.

BALD formula:
    BALD(x) = H(mean_t p_t) - mean_t H(p_t)

Higher BALD = higher epistemic uncertainty / model disagreement.

Output:
    12 PNG files:
      bald_uncertainty_ms_10pct.png
      bald_uncertainty_ms_20pct.png
      bald_uncertainty_ms_40pct.png
      ...and similarly for hhar, pamap2, ucihar.

Each plot contains:
    - x-axis: Test Sample Index
    - y-axis: BALD Value
    - one dot per test sample
    - dashed horizontal mean-BALD line
    - Mean = ... annotation
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration -- kept consistent with mtp_random_split.py
# ============================================================

BASE_DIR = Path("al_mtp_data")
MODEL_DIR = Path("models")

CSV_DIR = Path("uncertainty_csv")
CSV_DIR.mkdir(exist_ok=True)

DATASETS = ["ms", "hhar", "pamap2", "ucihar"]
TRAINING_FRACTIONS = [0.10, 0.20, 0.40]

WINDOW_CONFIG = {
    "ms": (128, 64),
    "hhar": (128, 64),
    "pamap2": (512, 256),
    "ucihar": (128, 64),
}

# These are the same six seeds used by mtp_random_split.py.
SEEDS = [32, 42, 72, 13, 100, 7]

VAL_SHARE_OF_REMAINDER = 0.15
BATCH_SIZE = 128

# Number of stochastic forward passes for BALD.
# More passes give a more stable Monte-Carlo estimate but take longer.
MC_PASSES = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running BALD on {device}")


# ============================================================
# 2. Same preprocessing helpers as mtp_random_split.py
# ============================================================

def numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[
        :, ~frame.columns.astype(str).str.startswith("Unnamed")
    ]

    numeric = frame.select_dtypes(include=[np.number]).copy()

    numeric = (
        numeric
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .fillna(0.0)
    )

    if {"x", "y", "z"}.issubset(numeric.columns):
        numeric["mag"] = np.sqrt(
            numeric["x"] ** 2
            + numeric["y"] ** 2
            + numeric["z"] ** 2
        )

    return numeric


def make_windows(
    values: np.ndarray,
    label: int,
    window_size: int,
    stride: int,
):
    n_features = values.shape[1] if values.ndim == 2 else 0

    if len(values) < window_size:
        return (
            np.empty(
                (0, window_size, n_features),
                dtype=np.float32,
            ),
            np.empty(0, dtype=np.int64),
        )

    windows = [
        values[start:start + window_size]
        for start in range(
            0,
            len(values) - window_size + 1,
            stride,
        )
    ]

    return (
        np.asarray(windows, dtype=np.float32),
        np.full(len(windows), label, dtype=np.int64),
    )


def random_contiguous_split(
    values: np.ndarray,
    label: int,
    fraction: float,
    window_size: int,
    seed: int,
):
    """
    EXACT split logic used by mtp_random_split.py.

    A contiguous raw-signal training block is selected, with a
    window-sized buffer around the boundary. The remainder is
    windowed separately.
    """

    rng = np.random.default_rng(seed)

    n_samples = len(values)
    train_len = int(round(fraction * n_samples))
    buffer = window_size

    max_start = n_samples - train_len

    if max_start <= 0:
        raise ValueError(
            "Requested training fraction is larger than this file's length."
        )

    train_start = int(
        rng.integers(0, max_start + 1)
    )
    train_end = train_start + train_len

    train_block = values[train_start:train_end]

    before_block = values[
        :max(0, train_start - buffer)
    ]

    after_block = values[
        train_end + buffer:
    ]

    stride = max(1, window_size // 4)

    x_train, y_train = make_windows(
        train_block,
        label,
        window_size,
        stride,
    )

    x_before, y_before = make_windows(
        before_block,
        label,
        window_size,
        stride,
    )

    x_after, y_after = make_windows(
        after_block,
        label,
        window_size,
        stride,
    )

    remainder_x_parts = [
        x for x in (x_before, x_after)
        if len(x)
    ]

    remainder_y_parts = [
        y for y in (y_before, y_after)
        if len(y)
    ]

    if remainder_x_parts:
        x_remainder = np.concatenate(
            remainder_x_parts
        )
        y_remainder = np.concatenate(
            remainder_y_parts
        )
    else:
        x_remainder = np.empty(
            (
                0,
                window_size,
                values.shape[1],
            ),
            dtype=np.float32,
        )
        y_remainder = np.empty(
            0,
            dtype=np.int64,
        )

    return (
        x_train,
        y_train,
        x_remainder,
        y_remainder,
    )


def normalize_per_window(x: np.ndarray) -> np.ndarray:
    mean = x.mean(
        axis=1,
        keepdims=True,
    )

    std = x.std(
        axis=1,
        keepdims=True,
    ) + 1e-8

    return (
        (x - mean) / std
    ).astype(np.float32)


# ============================================================
# 3. SAME CNN architecture used by mtp_random_split.py
# ============================================================

class HAR_CNN(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                n_features,
                64,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                128,
                256,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(
                256,
                n_classes,
            ),
        )

    def forward(self, x):
        return self.classifier(
            self.features(
                x.permute(0, 2, 1)
            )
        )


# ============================================================
# 4. MC Dropout prediction
# ============================================================

def enable_dropout_only(model):
    """
    Keep the model in eval mode so BatchNorm statistics remain fixed,
    but turn Dropout layers back to train mode.

    This is the standard MC-Dropout setup needed for BALD.
    """
    model.eval()

    for module in model.modules():
        if isinstance(
            module,
            (
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.Dropout3d,
            ),
        ):
            module.train()


def mc_dropout_probabilities(
    model,
    x,
    n_passes=30,
):
    """
    Returns:
        probs: shape (n_passes, n_samples, n_classes)
    """

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x)
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_passes = []

    enable_dropout_only(model)

    with torch.no_grad():
        for pass_idx in range(n_passes):
            pass_probs = []

            for (xb,) in loader:
                logits = model(
                    xb.to(device)
                )

                probs = torch.softmax(
                    logits,
                    dim=1,
                ).cpu().numpy()

                pass_probs.append(probs)

            pass_probs = np.concatenate(
                pass_probs,
                axis=0,
            )

            all_passes.append(pass_probs)

    return np.stack(
        all_passes,
        axis=0,
    )


# ============================================================
# 5. BALD calculation
# ============================================================

def entropy_from_probabilities(probs):
    """
    Entropy of categorical probability distributions.

    probs shape:
        (..., n_classes)
    """
    probs = np.clip(
        probs,
        1e-12,
        1.0,
    )

    return -np.sum(
        probs * np.log(probs),
        axis=-1,
    )


def compute_bald(mc_probs):
    """
    BALD(x) =
        H(mean_t p_t)
        - mean_t H(p_t)

    mc_probs:
        shape = (MC passes, samples, classes)

    returns:
        one BALD value per sample
    """

    mean_probs = np.mean(
        mc_probs,
        axis=0,
    )

    predictive_entropy = entropy_from_probabilities(
        mean_probs
    )

    individual_entropies = entropy_from_probabilities(
        mc_probs
    )

    expected_entropy = np.mean(
        individual_entropies,
        axis=0,
    )

    bald = (
        predictive_entropy
        - expected_entropy
    )

    # Small negative values can occur from floating-point error.
    return np.maximum(
        bald,
        0.0,
    )


# ============================================================
# 6. Reconstruct the SAME test set
# ============================================================

def reconstruct_test_set(
    dataset_name,
    fraction,
    best_seed,
):
    """
    Reconstruct exactly the test data used to create the checkpoint.

    mtp_random_split.py:
      1. creates the contiguous training block using seed + label
      2. takes the remainder
      3. performs train_test_split(remainder, ...)
         with random_state=seed
         and test_size=1-VAL_SHARE_OF_REMAINDER

    This function repeats those exact operations.
    """

    window_size, _ = WINDOW_CONFIG[
        dataset_name
    ]

    data_dir = (
        BASE_DIR
        / dataset_name
        / "data"
    )

    csv_paths = sorted(
        p for p in data_dir.glob("*.csv")
        if p.name.lower()
        not in {
            "std.csv",
            "mean.csv",
            "metadata.csv",
            "labels.csv",
        }
    )

    if not csv_paths:
        raise FileNotFoundError(
            f"No dataset CSVs found in {data_dir}"
        )

    remainder_x_list = []
    remainder_y_list = []

    for label, path in enumerate(
        csv_paths
    ):
        values = numeric_features(
            pd.read_csv(path)
        ).to_numpy(
            dtype=np.float32
        )

        _, _, x_rem, y_rem = (
            random_contiguous_split(
                values,
                label,
                fraction,
                window_size,
                seed=best_seed + label,
            )
        )

        remainder_x_list.append(x_rem)
        remainder_y_list.append(y_rem)

    remainder_x = np.concatenate(
        remainder_x_list
    )

    remainder_y = np.concatenate(
        remainder_y_list
    )

    # EXACT same split used by mtp_random_split.py.
    _, test_x, _, test_y = train_test_split(
        remainder_x,
        remainder_y,
        test_size=1 - VAL_SHARE_OF_REMAINDER,
        random_state=best_seed,
        stratify=remainder_y,
    )

    test_x = normalize_per_window(
        test_x
    )

    return (
        test_x,
        test_y,
        len(csv_paths),
        test_x.shape[-1],
    )


# ============================================================
# 7. Read best seed from checkpoint filename
# ============================================================

def find_checkpoint(
    dataset_name,
    fraction,
):
    """
    Finds:
        models/<dataset>_frac<percentage>_seed<S>.pt

    The filename itself tells us which seed was the best seed,
    because mtp_random_split.py saved only that seed's checkpoint.
    """

    percentage = int(
        round(fraction * 100)
    )

    matches = sorted(
        MODEL_DIR.glob(
            f"{dataset_name}_frac{percentage}_seed*.pt"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for "
            f"{dataset_name} {percentage}% in "
            f"{MODEL_DIR}/"
        )

    # Normally there should be exactly one checkpoint per
    # dataset/fraction because the existing script saves the
    # best seed only.
    if len(matches) > 1:
        print(
            f"WARNING: found {len(matches)} checkpoints for "
            f"{dataset_name} {percentage}%. "
            f"Using the first sorted match: {matches[0].name}"
        )

    checkpoint = matches[0]

    # Expected format:
    # dataset_frac10_seed32.pt
    stem = checkpoint.stem
    seed_text = stem.split("_seed")[-1]

    try:
        best_seed = int(seed_text)
    except ValueError:
        raise ValueError(
            f"Could not determine seed from checkpoint name: "
            f"{checkpoint.name}"
        )

    if best_seed not in SEEDS:
        print(
            f"WARNING: checkpoint seed {best_seed} is not in "
            f"the configured seed list {SEEDS}."
        )

    return checkpoint, best_seed


# ============================================================
# 8. Plot one BALD scatter plot
# ============================================================

def save_bald_csv(dataset_name, fraction, bald_values, best_seed):
    percentage = int(round(fraction * 100))

    fname = (
        CSV_DIR
        / f"bald_{dataset_name}_{percentage}pct_seed{best_seed}.csv"
    )

    df = pd.DataFrame({
        "sample_index": np.arange(len(bald_values)),
        "bald": bald_values,
    })

    df.to_csv(fname, index=False)

    print(f"Saved {fname}")

def plot_bald(
    dataset_name,
    fraction,
    bald_values,
    best_seed,
):
    percentage = int(
        round(fraction * 100)
    )

    mean_bald = float(
        np.mean(bald_values)
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.scatter(
        np.arange(
            len(bald_values)
        ),
        bald_values,
        s=4,
        alpha=0.30,
    )

    # Mean BALD line.
    plt.axhline(
        mean_bald,
        linestyle="--",
        linewidth=1.2,
        color="black",
    )

    # Mean annotation.
    plt.text(
        0.98,
        0.95,
        f"Mean = {mean_bald:.4f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
        ),
    )

    plt.xlabel(
        "Test Sample Index"
    )

    plt.ylabel(
        "BALD Value"
    )

    plt.title(
        f"BALD — {percentage}% train data "
        f"({dataset_name.upper()}) "
        f"[seed {best_seed}]"
    )

    plt.tight_layout()

    filename = (
        f"bald_uncertainty_"
        f"{dataset_name}_"
        f"{percentage}pct.png"
    )

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved {filename} | "
        f"n={len(bald_values)} | "
        f"mean={mean_bald:.6f} | "
        f"seed={best_seed}"
    )


# ============================================================
# 9. Main
# ============================================================

def main():

    MODEL_DIR.mkdir(
        exist_ok=True
    )

    total_plots = 0

    for dataset_name in DATASETS:

        print(
            f"\n{'=' * 60}\n"
            f"DATASET: {dataset_name.upper()}\n"
            f"{'=' * 60}"
        )

        for fraction in TRAINING_FRACTIONS:

            percentage = int(
                round(fraction * 100)
            )

            print(
                f"\n--- BALD: "
                f"{dataset_name.upper()} "
                f"{percentage}% ---"
            )

            # ------------------------------------------------
            # Load the EXISTING best checkpoint.
            # No retraining.
            # ------------------------------------------------
            checkpoint_path, best_seed = (
                find_checkpoint(
                    dataset_name,
                    fraction,
                )
            )

            print(
                f"Loading checkpoint: "
                f"{checkpoint_path}"
            )

            # ------------------------------------------------
            # Reconstruct the SAME test set.
            # ------------------------------------------------
            test_x, test_y, n_classes, n_features = (
                reconstruct_test_set(
                    dataset_name,
                    fraction,
                    best_seed,
                )
            )

            print(
                f"Test samples: {len(test_y)} | "
                f"Features: {n_features} | "
                f"Classes: {n_classes}"
            )

            # ------------------------------------------------
            # Create the same CNN architecture and load the
            # already-trained weights.
            # ------------------------------------------------
            model = HAR_CNN(
                n_features,
                n_classes,
            ).to(device)

            state_dict = torch.load(
                checkpoint_path,
                map_location=device,
            )

            model.load_state_dict(
                state_dict
            )

            # ------------------------------------------------
            # MC Dropout.
            # ------------------------------------------------
            print(
                f"Running {MC_PASSES} stochastic "
                f"forward passes..."
            )

            mc_probs = (
                mc_dropout_probabilities(
                    model,
                    test_x,
                    n_passes=MC_PASSES,
                )
            )

            # ------------------------------------------------
            # Compute BALD for every test sample.
            # ------------------------------------------------
            bald_values = compute_bald(
                mc_probs
            )

            print(
                f"BALD range: "
                f"{bald_values.min():.6f} "
                f"to "
                f"{bald_values.max():.6f}"
            )

            # ------------------------------------------------
            # save bald values to csv
            # ------------------------------------------------
            save_bald_csv(
                dataset_name,
                fraction,
                bald_values,
                best_seed,
            )


            plot_bald(dataset_name,
                      fraction,
                      bald_values,
                      best_seed,
                      )

            total_plots += 1

    print(
        f"\n{'=' * 60}\n"
        f"BALD computation complete.\n"
        f"Generated {total_plots} BALD scatter plots.\n"
        f"{'=' * 60}"
    )


if __name__ == "__main__":
    main()