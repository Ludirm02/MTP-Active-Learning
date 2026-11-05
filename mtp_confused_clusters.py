
# # """
# # Confused-cluster extraction via K-Means (silhouette-optimal K), built on top
# # of the ALREADY-COMPUTED uncertainty CSVs -- no retraining, no re-scoring.

# # WHAT THIS DOES:
# # ------------------------------------------------------------------------
# # 1. Scans models/ for checkpoint filenames of the form
# #    <dataset>_frac<fraction>_seed<seed>.pt (saved by mtp_random_split.py,
# #    which already picked the best-accuracy seed per dataset/fraction). This
# #    tells us, for free, which seed is "best" for each (dataset, fraction) --
# #    no accuracy needs to be recomputed.
# # 2. For that (dataset, fraction, best seed), loads the matching CSV from
# #    uncertainty_csv/uncertainty_<dataset>_<fraction>pct_seed<seed>.csv
# #    (saved by the uncertainty-scatter script) -- columns: sample_index,
# #    entropy, least_confidence, margin.
# # 3. For each of the three metrics, sweeps K = 2..8, picks the K that
# #    maximises the silhouette score, and clusters that metric's values.
# # 4. Identifies the "most confused" cluster: highest mean value for entropy
# #    and least_confidence (higher = more uncertain), LOWEST mean value for
# #    margin (smaller margin = more uncertain -- this is the opposite
# #    direction, and easy to get backwards, so it's handled explicitly below).
# # 5. Saves just that cluster's rows to confused_<dataset>_<fraction>_<metric>.csv.

# # BALD is intentionally NOT included yet, per instruction -- add it once its
# # CSV format/location is available; the loop below is written so a 4th
# # metric can be dropped in later with one line.

# # COST: zero model training. Only reads existing CSVs and runs K-Means on
# # single columns of already-computed numbers -- this should run in seconds.
# # """

# #  #
# #  #
# import re
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# MODEL_DIR = Path("models")
# CSV_DIR = Path("uncertainty_csv")
# OUTPUT_DIR = Path("confused_clusters")
# OUTPUT_DIR.mkdir(exist_ok=True)

# METRICS = ["bald"]
# # For entropy/LC, higher value = more uncertain -> take the highest-mean cluster.
# # For margin, smaller value = more uncertain -> take the lowest-mean cluster.
# #
# #HIGHER_IS_MORE_UNCERTAIN = {"entropy": True, "least_confidence": True, "margin": False}
# HIGHER_IS_MORE_UNCERTAIN = {
#     "bald": True
# }


# K_RANGE = range(2, 9)  # 2 to 8 inclusive


# def find_best_seeds():
#     """Parses models/*.pt filenames to recover the best-accuracy seed per
#     (dataset, fraction), as already chosen by mtp_random_split.py."""
#     pattern = re.compile(r"^(?P<dataset>[a-z0-9]+)_frac(?P<frac>\d+)_seed(?P<seed>\d+)\.pt$")
#     best_seeds = {}
#     for ckpt in MODEL_DIR.glob("*.pt"):
#         m = pattern.match(ckpt.name)
#         if not m:
#             print(f"  Skipping unrecognized checkpoint filename: {ckpt.name}")
#             continue
#         dataset = m.group("dataset")
#         frac = int(m.group("frac"))
#         seed = int(m.group("seed"))
#         best_seeds[(dataset, frac)] = seed
#     return best_seeds


# def find_confused_cluster(values, higher_is_more_uncertain):
#     """Sweeps K=2..8, picks the silhouette-best K, returns a boolean mask
#     over `values` marking membership in the most-confused cluster, plus the
#     chosen K and that cluster's mean value (for logging)."""
#     values_2d = values.reshape(-1, 1)
#     best_k, best_score, best_labels = None, -1.0, None
#     for k in K_RANGE:
#         if k >= len(values):
#             break
#         labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(values_2d)
#         score = silhouette_score(values_2d, labels)
#         if score > best_score:
#             best_k, best_score, best_labels = k, score, labels

#     cluster_means = pd.Series(values).groupby(best_labels).mean()
#     confused_cluster = cluster_means.idxmax() if higher_is_more_uncertain else cluster_means.idxmin()
#     mask = best_labels == confused_cluster
#     return mask, best_k, cluster_means[confused_cluster]


# def main():
#     best_seeds = find_best_seeds()
#     if not best_seeds:
#         print(f"No checkpoints found under {MODEL_DIR}/ -- nothing to do.")
#         return

#     for (dataset, frac), seed in sorted(best_seeds.items()):
#         csv_path = CSV_DIR / f"bald_{dataset}_{frac}pct_seed{seed}.csv"
#         if not csv_path.exists():
#             print(f"  Missing CSV for {dataset} {frac}% (seed {seed}): {csv_path} -- skipping.")
#             continue

#         df = pd.read_csv(csv_path)
#         print(f"\n{dataset.upper()} {frac}% (best seed {seed}, n={len(df)}):")

#         for metric in METRICS:
#             values = df[metric].to_numpy()
#             higher = HIGHER_IS_MORE_UNCERTAIN[metric]
#             mask, best_k, confused_mean = find_confused_cluster(values, higher)

#             confused_df = df[mask]
#             out_path = OUTPUT_DIR / f"confused_{dataset}_{frac}_{metric}.csv"
#             confused_df.to_csv(out_path, index=False)

#             print(
#                 f"  {metric}: best K={best_k}, confused cluster mean={confused_mean:.4f}, "
#                 f"{mask.sum()}/{len(df)} samples -> {out_path}"
#             )


# if __name__ == "__main__":
#     main()


# """
# BALD uncertainty scatter plots for the existing HAR experiment.

# Purpose
# -------
# Compute ONLY BALD uncertainty for the SAME:
#     - 4 datasets: MotionSense, HHAR, PAMAP2, UCI HAR
#     - 3 training fractions: 10%, 20%, 40%
#     - best-accuracy seed/checkpoint already produced by mtp_random_split.py

# This script does NOT:
#     - retrain any model
#     - recompute entropy / least-confidence / margin
#     - run K-Means
#     - run active learning
#     - change the existing uncertainty results

# It loads the saved best-model checkpoints from:
#     models/<dataset>_frac<10|20|40>_seed<S>.pt

# It reconstructs the SAME test split used by mtp_random_split.py and
# computes BALD with MC Dropout.

# BALD formula:
#     BALD(x) = H(mean_t p_t) - mean_t H(p_t)

# Higher BALD = higher epistemic uncertainty / model disagreement.

# Output:
#     12 PNG files:
#       bald_uncertainty_ms_10pct.png
#       bald_uncertainty_ms_20pct.png
#       bald_uncertainty_ms_40pct.png
#       ...and similarly for hhar, pamap2, ucihar.

# Each plot contains:
#     - x-axis: Test Sample Index
#     - y-axis: BALD Value
#     - one dot per test sample
#     - dashed horizontal mean-BALD line
#     - Mean = ... annotation
# """

# # import math
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd
# # import torch
# # import torch.nn as nn
# # from sklearn.model_selection import train_test_split
# # from torch.utils.data import DataLoader, TensorDataset
# # import matplotlib.pyplot as plt


# # # ============================================================
# # # 1. Configuration -- kept consistent with mtp_random_split.py
# # # ============================================================

# # BASE_DIR = Path("al_mtp_data")
# # MODEL_DIR = Path("models")

# # DATASETS = ["ms", "hhar", "pamap2", "ucihar"]
# # TRAINING_FRACTIONS = [0.10, 0.20, 0.40]

# # WINDOW_CONFIG = {
# #     "ms": (128, 64),
# #     "hhar": (128, 64),
# #     "pamap2": (512, 256),
# #     "ucihar": (128, 64),
# # }

# # # These are the same six seeds used by mtp_random_split.py.
# # SEEDS = [32, 42, 72, 13, 100, 7]

# # VAL_SHARE_OF_REMAINDER = 0.15
# # BATCH_SIZE = 128

# # # Number of stochastic forward passes for BALD.
# # # More passes give a more stable Monte-Carlo estimate but take longer.
# # MC_PASSES = 30

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"Running BALD on {device}")


# # # ============================================================
# # # 2. Same preprocessing helpers as mtp_random_split.py
# # # ============================================================

# # def numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
# #     frame = frame.loc[
# #         :, ~frame.columns.astype(str).str.startswith("Unnamed")
# #     ]

# #     numeric = frame.select_dtypes(include=[np.number]).copy()

# #     numeric = (
# #         numeric
# #         .replace([np.inf, -np.inf], np.nan)
# #         .interpolate(limit_direction="both")
# #         .fillna(0.0)
# #     )

# #     if {"x", "y", "z"}.issubset(numeric.columns):
# #         numeric["mag"] = np.sqrt(
# #             numeric["x"] ** 2
# #             + numeric["y"] ** 2
# #             + numeric["z"] ** 2
# #         )

# #     return numeric


# # def make_windows(
# #     values: np.ndarray,
# #     label: int,
# #     window_size: int,
# #     stride: int,
# # ):
# #     n_features = values.shape[1] if values.ndim == 2 else 0

# #     if len(values) < window_size:
# #         return (
# #             np.empty(
# #                 (0, window_size, n_features),
# #                 dtype=np.float32,
# #             ),
# #             np.empty(0, dtype=np.int64),
# #         )

# #     windows = [
# #         values[start:start + window_size]
# #         for start in range(
# #             0,
# #             len(values) - window_size + 1,
# #             stride,
# #         )
# #     ]

# #     return (
# #         np.asarray(windows, dtype=np.float32),
# #         np.full(len(windows), label, dtype=np.int64),
# #     )


# # def random_contiguous_split(
# #     values: np.ndarray,
# #     label: int,
# #     fraction: float,
# #     window_size: int,
# #     seed: int,
# # ):
# #     """
# #     EXACT split logic used by mtp_random_split.py.

# #     A contiguous raw-signal training block is selected, with a
# #     window-sized buffer around the boundary. The remainder is
# #     windowed separately.
# #     """

# #     rng = np.random.default_rng(seed)

# #     n_samples = len(values)
# #     train_len = int(round(fraction * n_samples))
# #     buffer = window_size

# #     max_start = n_samples - train_len

# #     if max_start <= 0:
# #         raise ValueError(
# #             "Requested training fraction is larger than this file's length."
# #         )

# #     train_start = int(
# #         rng.integers(0, max_start + 1)
# #     )
# #     train_end = train_start + train_len

# #     train_block = values[train_start:train_end]

# #     before_block = values[
# #         :max(0, train_start - buffer)
# #     ]

# #     after_block = values[
# #         train_end + buffer:
# #     ]

# #     stride = max(1, window_size // 4)

# #     x_train, y_train = make_windows(
# #         train_block,
# #         label,
# #         window_size,
# #         stride,
# #     )

# #     x_before, y_before = make_windows(
# #         before_block,
# #         label,
# #         window_size,
# #         stride,
# #     )

# #     x_after, y_after = make_windows(
# #         after_block,
# #         label,
# #         window_size,
# #         stride,
# #     )

# #     remainder_x_parts = [
# #         x for x in (x_before, x_after)
# #         if len(x)
# #     ]

# #     remainder_y_parts = [
# #         y for y in (y_before, y_after)
# #         if len(y)
# #     ]

# #     if remainder_x_parts:
# #         x_remainder = np.concatenate(
# #             remainder_x_parts
# #         )
# #         y_remainder = np.concatenate(
# #             remainder_y_parts
# #         )
# #     else:
# #         x_remainder = np.empty(
# #             (
# #                 0,
# #                 window_size,
# #                 values.shape[1],
# #             ),
# #             dtype=np.float32,
# #         )
# #         y_remainder = np.empty(
# #             0,
# #             dtype=np.int64,
# #         )

# #     return (
# #         x_train,
# #         y_train,
# #         x_remainder,
# #         y_remainder,
# #     )


# # def normalize_per_window(x: np.ndarray) -> np.ndarray:
# #     mean = x.mean(
# #         axis=1,
# #         keepdims=True,
# #     )

# #     std = x.std(
# #         axis=1,
# #         keepdims=True,
# #     ) + 1e-8

# #     return (
# #         (x - mean) / std
# #     ).astype(np.float32)


# # # ============================================================
# # # 3. SAME CNN architecture used by mtp_random_split.py
# # # ============================================================

# # class HAR_CNN(nn.Module):
# #     def __init__(
# #         self,
# #         n_features: int,
# #         n_classes: int,
# #     ):
# #         super().__init__()

# #         self.features = nn.Sequential(
# #             nn.Conv1d(
# #                 n_features,
# #                 64,
# #                 kernel_size=7,
# #                 padding=3,
# #             ),
# #             nn.BatchNorm1d(64),
# #             nn.ReLU(),
# #             nn.MaxPool1d(2),

# #             nn.Conv1d(
# #                 64,
# #                 128,
# #                 kernel_size=5,
# #                 padding=2,
# #             ),
# #             nn.BatchNorm1d(128),
# #             nn.ReLU(),
# #             nn.MaxPool1d(2),

# #             nn.Conv1d(
# #                 128,
# #                 256,
# #                 kernel_size=3,
# #                 padding=1,
# #             ),
# #             nn.BatchNorm1d(256),
# #             nn.ReLU(),
# #             nn.AdaptiveAvgPool1d(1),
# #         )

# #         self.classifier = nn.Sequential(
# #             nn.Flatten(),
# #             nn.Dropout(0.30),
# #             nn.Linear(
# #                 256,
# #                 n_classes,
# #             ),
# #         )

# #     def forward(self, x):
# #         return self.classifier(
# #             self.features(
# #                 x.permute(0, 2, 1)
# #             )
# #         )


# # # ============================================================
# # # 4. MC Dropout prediction
# # # ============================================================

# # def enable_dropout_only(model):
# #     """
# #     Keep the model in eval mode so BatchNorm statistics remain fixed,
# #     but turn Dropout layers back to train mode.

# #     This is the standard MC-Dropout setup needed for BALD.
# #     """
# #     model.eval()

# #     for module in model.modules():
# #         if isinstance(
# #             module,
# #             (
# #                 nn.Dropout,
# #                 nn.Dropout1d,
# #                 nn.Dropout2d,
# #                 nn.Dropout3d,
# #             ),
# #         ):
# #             module.train()


# # def mc_dropout_probabilities(
# #     model,
# #     x,
# #     n_passes=30,
# # ):
# #     """
# #     Returns:
# #         probs: shape (n_passes, n_samples, n_classes)
# #     """

# #     loader = DataLoader(
# #         TensorDataset(
# #             torch.from_numpy(x)
# #         ),
# #         batch_size=BATCH_SIZE,
# #         shuffle=False,
# #     )

# #     all_passes = []

# #     enable_dropout_only(model)

# #     with torch.no_grad():
# #         for pass_idx in range(n_passes):
# #             pass_probs = []

# #             for (xb,) in loader:
# #                 logits = model(
# #                     xb.to(device)
# #                 )

# #                 probs = torch.softmax(
# #                     logits,
# #                     dim=1,
# #                 ).cpu().numpy()

# #                 pass_probs.append(probs)

# #             pass_probs = np.concatenate(
# #                 pass_probs,
# #                 axis=0,
# #             )

# #             all_passes.append(pass_probs)

# #     return np.stack(
# #         all_passes,
# #         axis=0,
# #     )


# # # ============================================================
# # # 5. BALD calculation
# # # ============================================================

# # def entropy_from_probabilities(probs):
# #     """
# #     Entropy of categorical probability distributions.

# #     probs shape:
# #         (..., n_classes)
# #     """
# #     probs = np.clip(
# #         probs,
# #         1e-12,
# #         1.0,
# #     )

# #     return -np.sum(
# #         probs * np.log(probs),
# #         axis=-1,
# #     )


# # def compute_bald(mc_probs):
# #     """
# #     BALD(x) =
# #         H(mean_t p_t)
# #         - mean_t H(p_t)

# #     mc_probs:
# #         shape = (MC passes, samples, classes)

# #     returns:
# #         one BALD value per sample
# #     """

# #     mean_probs = np.mean(
# #         mc_probs,
# #         axis=0,
# #     )

# #     predictive_entropy = entropy_from_probabilities(
# #         mean_probs
# #     )

# #     individual_entropies = entropy_from_probabilities(
# #         mc_probs
# #     )

# #     expected_entropy = np.mean(
# #         individual_entropies,
# #         axis=0,
# #     )

# #     bald = (
# #         predictive_entropy
# #         - expected_entropy
# #     )

# #     # Small negative values can occur from floating-point error.
# #     return np.maximum(
# #         bald,
# #         0.0,
# #     )


# # # ============================================================
# # # 6. Reconstruct the SAME test set
# # # ============================================================

# # def reconstruct_test_set(
# #     dataset_name,
# #     fraction,
# #     best_seed,
# # ):
# #     """
# #     Reconstruct exactly the test data used to create the checkpoint.

# #     mtp_random_split.py:
# #       1. creates the contiguous training block using seed + label
# #       2. takes the remainder
# #       3. performs train_test_split(remainder, ...)
# #          with random_state=seed
# #          and test_size=1-VAL_SHARE_OF_REMAINDER

# #     This function repeats those exact operations.
# #     """

# #     window_size, _ = WINDOW_CONFIG[
# #         dataset_name
# #     ]

# #     data_dir = (
# #         BASE_DIR
# #         / dataset_name
# #         / "data"
# #     )

# #     csv_paths = sorted(
# #         p for p in data_dir.glob("*.csv")
# #         if p.name.lower()
# #         not in {
# #             "std.csv",
# #             "mean.csv",
# #             "metadata.csv",
# #             "labels.csv",
# #         }
# #     )

# #     if not csv_paths:
# #         raise FileNotFoundError(
# #             f"No dataset CSVs found in {data_dir}"
# #         )

# #     remainder_x_list = []
# #     remainder_y_list = []

# #     for label, path in enumerate(
# #         csv_paths
# #     ):
# #         values = numeric_features(
# #             pd.read_csv(path)
# #         ).to_numpy(
# #             dtype=np.float32
# #         )

# #         _, _, x_rem, y_rem = (
# #             random_contiguous_split(
# #                 values,
# #                 label,
# #                 fraction,
# #                 window_size,
# #                 seed=best_seed + label,
# #             )
# #         )

# #         remainder_x_list.append(x_rem)
# #         remainder_y_list.append(y_rem)

# #     remainder_x = np.concatenate(
# #         remainder_x_list
# #     )

# #     remainder_y = np.concatenate(
# #         remainder_y_list
# #     )

# #     # EXACT same split used by mtp_random_split.py.
# #     _, test_x, _, test_y = train_test_split(
# #         remainder_x,
# #         remainder_y,
# #         test_size=1 - VAL_SHARE_OF_REMAINDER,
# #         random_state=best_seed,
# #         stratify=remainder_y,
# #     )

# #     test_x = normalize_per_window(
# #         test_x
# #     )

# #     return (
# #         test_x,
# #         test_y,
# #         len(csv_paths),
# #         test_x.shape[-1],
# #     )


# # # ============================================================
# # # 7. Read best seed from checkpoint filename
# # # ============================================================

# # def find_checkpoint(
# #     dataset_name,
# #     fraction,
# # ):
# #     """
# #     Finds:
# #         models/<dataset>_frac<percentage>_seed<S>.pt

# #     The filename itself tells us which seed was the best seed,
# #     because mtp_random_split.py saved only that seed's checkpoint.
# #     """

# #     percentage = int(
# #         round(fraction * 100)
# #     )

# #     matches = sorted(
# #         MODEL_DIR.glob(
# #             f"{dataset_name}_frac{percentage}_seed*.pt"
# #         )
# #     )

# #     if not matches:
# #         raise FileNotFoundError(
# #             f"No checkpoint found for "
# #             f"{dataset_name} {percentage}% in "
# #             f"{MODEL_DIR}/"
# #         )

# #     # Normally there should be exactly one checkpoint per
# #     # dataset/fraction because the existing script saves the
# #     # best seed only.
# #     if len(matches) > 1:
# #         print(
# #             f"WARNING: found {len(matches)} checkpoints for "
# #             f"{dataset_name} {percentage}%. "
# #             f"Using the first sorted match: {matches[0].name}"
# #         )

# #     checkpoint = matches[0]

# #     # Expected format:
# #     # dataset_frac10_seed32.pt
# #     stem = checkpoint.stem
# #     seed_text = stem.split("_seed")[-1]

# #     try:
# #         best_seed = int(seed_text)
# #     except ValueError:
# #         raise ValueError(
# #             f"Could not determine seed from checkpoint name: "
# #             f"{checkpoint.name}"
# #         )

# #     if best_seed not in SEEDS:
# #         print(
# #             f"WARNING: checkpoint seed {best_seed} is not in "
# #             f"the configured seed list {SEEDS}."
# #         )

# #     return checkpoint, best_seed


# # # ============================================================
# # # 8. Plot one BALD scatter plot
# # # ============================================================

# # def plot_bald(
# #     dataset_name,
# #     fraction,
# #     bald_values,
# #     best_seed,
# # ):
# #     percentage = int(
# #         round(fraction * 100)
# #     )

# #     mean_bald = float(
# #         np.mean(bald_values)
# #     )

# #     plt.figure(
# #         figsize=(10, 5)
# #     )

# #     plt.scatter(
# #         np.arange(
# #             len(bald_values)
# #         ),
# #         bald_values,
# #         s=4,
# #         alpha=0.30,
# #     )

# #     # Mean BALD line.
# #     plt.axhline(
# #         mean_bald,
# #         linestyle="--",
# #         linewidth=1.2,
# #         color="black",
# #     )

# #     # Mean annotation.
# #     plt.text(
# #         0.98,
# #         0.95,
# #         f"Mean = {mean_bald:.4f}",
# #         transform=plt.gca().transAxes,
# #         ha="right",
# #         va="top",
# #         fontsize=9,
# #         bbox=dict(
# #             boxstyle="round",
# #             facecolor="white",
# #             alpha=0.8,
# #         ),
# #     )

# #     plt.xlabel(
# #         "Test Sample Index"
# #     )

# #     plt.ylabel(
# #         "BALD Value"
# #     )

# #     plt.title(
# #         f"BALD — {percentage}% train data "
# #         f"({dataset_name.upper()}) "
# #         f"[seed {best_seed}]"
# #     )

# #     plt.tight_layout()

# #     filename = (
# #         f"bald_uncertainty_"
# #         f"{dataset_name}_"
# #         f"{percentage}pct.png"
# #     )

# #     plt.savefig(
# #         filename,
# #         dpi=200,
# #         bbox_inches="tight",
# #     )

# #     plt.close()

# #     print(
# #         f"Saved {filename} | "
# #         f"n={len(bald_values)} | "
# #         f"mean={mean_bald:.6f} | "
# #         f"seed={best_seed}"
# #     )


# # # ============================================================
# # # 9. Main
# # # ============================================================

# # def main():

# #     MODEL_DIR.mkdir(
# #         exist_ok=True
# #     )

# #     total_plots = 0

# #     for dataset_name in DATASETS:

# #         print(
# #             f"\n{'=' * 60}\n"
# #             f"DATASET: {dataset_name.upper()}\n"
# #             f"{'=' * 60}"
# #         )

# #         for fraction in TRAINING_FRACTIONS:

# #             percentage = int(
# #                 round(fraction * 100)
# #             )

# #             print(
# #                 f"\n--- BALD: "
# #                 f"{dataset_name.upper()} "
# #                 f"{percentage}% ---"
# #             )

# #             # ------------------------------------------------
# #             # Load the EXISTING best checkpoint.
# #             # No retraining.
# #             # ------------------------------------------------
# #             checkpoint_path, best_seed = (
# #                 find_checkpoint(
# #                     dataset_name,
# #                     fraction,
# #                 )
# #             )

# #             print(
# #                 f"Loading checkpoint: "
# #                 f"{checkpoint_path}"
# #             )

# #             # ------------------------------------------------
# #             # Reconstruct the SAME test set.
# #             # ------------------------------------------------
# #             test_x, test_y, n_classes, n_features = (
# #                 reconstruct_test_set(
# #                     dataset_name,
# #                     fraction,
# #                     best_seed,
# #                 )
# #             )

# #             print(
# #                 f"Test samples: {len(test_y)} | "
# #                 f"Features: {n_features} | "
# #                 f"Classes: {n_classes}"
# #             )

# #             # ------------------------------------------------
# #             # Create the same CNN architecture and load the
# #             # already-trained weights.
# #             # ------------------------------------------------
# #             model = HAR_CNN(
# #                 n_features,
# #                 n_classes,
# #             ).to(device)

# #             state_dict = torch.load(
# #                 checkpoint_path,
# #                 map_location=device,
# #             )

# #             model.load_state_dict(
# #                 state_dict
# #             )

# #             # ------------------------------------------------
# #             # MC Dropout.
# #             # ------------------------------------------------
# #             print(
# #                 f"Running {MC_PASSES} stochastic "
# #                 f"forward passes..."
# #             )

# #             mc_probs = (
# #                 mc_dropout_probabilities(
# #                     model,
# #                     test_x,
# #                     n_passes=MC_PASSES,
# #                 )
# #             )

# #             # ------------------------------------------------
# #             # Compute BALD for every test sample.
# #             # ------------------------------------------------
# #             bald_values = compute_bald(
# #                 mc_probs
# #             )

# #             print(
# #                 f"BALD range: "
# #                 f"{bald_values.min():.6f} "
# #                 f"to "
# #                 f"{bald_values.max():.6f}"
# #             )

# #             # ------------------------------------------------
# #             # Plot.
# #             # ------------------------------------------------
# #             plot_bald(
# #                 dataset_name,
# #                 fraction,
# #                 bald_values,
# #                 best_seed,
# #             )

# #             total_plots += 1

# #     print(
# #         f"\n{'=' * 60}\n"
# #         f"BALD computation complete.\n"
# #         f"Generated {total_plots} BALD scatter plots.\n"
# #         f"{'=' * 60}"
# #     )


# # if __name__ == "__main__":
# #     main()


"""
Full K-Means clustering analysis for entropy / least confidence / margin /
BALD, across all 4 datasets x 3 training fractions. Zero retraining -- reads
only the already-computed CSVs:
  - uncertainty_csv/uncertainty_<dataset>_<frac>pct_seed<seed>.csv
    (entropy, least_confidence, margin -- from the unlabeled pool)
  - uncertainty_csv/bald_<dataset>_<frac>pct_seed<seed>.csv
    (bald -- from mtp_random_split.py's test set)

CAVEAT (documented, not fixed here): the BALD CSVs and the entropy/LC/margin
CSVs come from two different underlying sample sets (different splits,
different trained models), even for the "same" seed number. sample_index N
in one file is NOT the same physical window as sample_index N in the other.
Each metric's clustering below is internally valid; cross-metric sample-level
comparisons are not.

For each (dataset, fraction, metric):
  1. Sweep K = 2..8, compute the silhouette score for each K.
  2. Pick the K with the highest silhouette score.
  3. Cluster labels at that K partition ALL samples into K groups
     (0, 1, ..., K-1).
  4. Identify the most-confused cluster: highest mean value for
     entropy/least_confidence/bald, LOWEST mean value for margin.
  5. Save that cluster's rows to confused_clusters/confused_<dataset>_<frac>_<metric>.csv
  6. Save a silhouette-vs-K plot to silhouette_scores/silhouette_<dataset>_<frac>_<metric>.png
  7. Save a full cluster scatter plot (ALL samples, unrounded values,
     colored by cluster 0..K-1) to
     cluster_scatter_plots/clusters_<dataset>_<frac>_<metric>.png
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

METRIC_CONFIG = {
    "entropy": {
        "file_prefix": "uncertainty",
        "column": "entropy",
        "higher_is_more_uncertain": True,
        "label": "Entropy",
    },
    "least_confidence": {
        "file_prefix": "uncertainty",
        "column": "least_confidence",
        "higher_is_more_uncertain": True,
        "label": "Least Confidence",
    },
    "margin": {
        "file_prefix": "uncertainty",
        "column": "margin",
        "higher_is_more_uncertain": False,  # smaller margin = more uncertain
        "label": "Margin",
    },
    "bald": {
        "file_prefix": "bald",
        "column": "bald",
        "higher_is_more_uncertain": True,
        "label": "BALD",
    },
}

CLUSTER_COLORS = plt.cm.tab10.colors  # up to 10 distinct colors; K maxes out at 8


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
    """Sweeps K=2..8. Returns (k_list, silhouette_list, best_k, best_labels)."""
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

        # Entropy/LC/margin share one file per (dataset, fraction, seed).
        unc_path = CSV_DIR / f"uncertainty_{dataset}_{frac}pct_seed{seed}.csv"
        unc_df = pd.read_csv(unc_path) if unc_path.exists() else None
        if unc_df is None:
            print(f"  Missing {unc_path} -- entropy/LC/margin skipped for this combo.")

        # BALD comes from a separate file (different underlying samples -- see caveat above).
        bald_path = CSV_DIR / f"bald_{dataset}_{frac}pct_seed{seed}.csv"
        bald_df = pd.read_csv(bald_path) if bald_path.exists() else None
        if bald_df is None:
            print(f"  Missing {bald_path} -- BALD skipped for this combo.")

        for metric, cfg in METRIC_CONFIG.items():
            df = unc_df if cfg["file_prefix"] == "uncertainty" else bald_df
            if df is None:
                continue

            values = df[cfg["column"]].to_numpy()
            k_list, sil_list, best_k, best_labels = sweep_k(values)
            if best_k is None:
                print(f"  {metric}: not enough samples to cluster, skipping.")
                continue

            # --- Confused-cluster CSV ---
            cluster_means = pd.Series(values).groupby(best_labels).mean()
            confused_cluster = (
                cluster_means.idxmax() if cfg["higher_is_more_uncertain"] else cluster_means.idxmin()
            )
            confused_mask = best_labels == confused_cluster
            confused_path = CONFUSED_DIR / f"confused_{dataset}_{frac}_{metric}.csv"
            df[confused_mask].to_csv(confused_path, index=False)

            # --- Silhouette-vs-K plot ---
            sil_path = SILHOUETTE_DIR / f"silhouette_{dataset}_{frac}_{metric}.png"
            plot_silhouette(dataset, frac, cfg["label"], k_list, sil_list, best_k, sil_path)

            # --- Full cluster scatter plot (all samples, all clusters, unrounded) ---
            scatter_path = SCATTER_DIR / f"clusters_{dataset}_{frac}_{metric}.png"
            plot_cluster_scatter(dataset, frac, cfg["label"], values, best_labels, best_k, scatter_path)

            print(
                f"  {metric}: best K={best_k}, confused cluster #{confused_cluster} "
                f"({confused_mask.sum()}/{len(df)} samples) -> "
                f"{confused_path.name}, {sil_path.name}, {scatter_path.name}"
            )
            total_done += 1

    print(f"\nDone. {total_done} (dataset,fraction,metric) combinations processed.")
    print(f"  {CONFUSED_DIR}/         -> confused-cluster CSVs")
    print(f"  {SILHOUETTE_DIR}/       -> silhouette-vs-K plots")
    print(f"  {SCATTER_DIR}/          -> full cluster scatter plots")


if __name__ == "__main__":
    main()