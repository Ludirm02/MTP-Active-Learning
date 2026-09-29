"""
Active Learning comparison experiment for HAR (Stages 4-7 of the project plan).

Builds on the same preprocessing/model as mtp_random_split.py, but adds what
that script did NOT do: a genuine unlabeled pool separate from a held-out
test set, top-N uncertain candidate selection, K-Means diversity selection
within that candidate pool, an iterative label-and-retrain loop, and a
comparison of 4 query strategies (Random, Entropy, Least Confidence, Margin)
at equal labeling budgets.

WHY A SEPARATE UNLABELED POOL MATTERS:
------------------------------------------------------------------------
Computing uncertainty on the same set used to report final accuracy is not
real active learning (you already have those labels). Here, each dataset
is split three ways per seed:
  - TEST       (held out, untouched, used ONLY to report accuracy at each round)
  - LABELED    (starts at ~10% of total data; grows each AL round)
  - UNLABELED  (the pool queried from; shrinks each AL round as samples move
                into LABELED)
Both extra splits reuse the same leak-safe contiguous-block splitting logic
as mtp_random_split.py (random start position per seed, buffer against
boundary leakage), applied twice in sequence.

PROTOCOL (confirmed with user):
------------------------------------------------------------------------
- Round budget = 10% of the ORIGINAL unlabeled pool size, fixed across rounds.
- 2 AL rounds (so 3 accuracy points per strategy: initial, round 1, round 2).
- Candidate pool for diversity selection = 3x the round budget, ranked by
  uncertainty, then clustered into up to 5 groups (K-Means) and sampled
  proportionally to cluster size so no single confused region dominates.
- Strategies: random (baseline), entropy, least confidence, margin.
- Random strategy skips uncertainty scoring and clustering entirely.

COST NOTE: this trains many more models than mtp_random_split.py
(datasets x strategies x rounds x seeds). SEEDS is reduced to 2 by default.
Consider testing on one dataset first by trimming DATASETS below.
"""

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --- 1. Configuration ---
BASE_DIR = Path("al_mtp_data")
DATASETS = ["ms", "hhar", "pamap2", "ucihar"]  # trim this list for a quick test run

WINDOW_CONFIG = {
    "ms": (128, 64),
    "hhar": (128, 64),
    "pamap2": (512, 256),
    "ucihar": (128, 64),
}

TEST_FRACTION = 0.20            # held-out, untouched test block (of total raw signal)
INITIAL_LABELED_FRACTION = 0.10  # of total raw signal (matches earlier baseline's "10%")
ROUND_BUDGET_FRACTION = 0.10     # of the ORIGINAL unlabeled pool size, per round
N_ROUNDS = 2
STRATEGIES = ["random", "entropy", "least_confidence", "margin"]
SEEDS = [32, 42]  # reduced for compute cost; raise once a full run is confirmed to work

TOP_K_MULTIPLIER = 3
N_CLUSTERS = 5

MAX_EPOCHS = 60      # reduced from the baseline script's 100, since this trains far more models
PATIENCE = 15
BATCH_SIZE = 128
VAL_SHARE_OF_LABELED = 0.15  # slice of the labeled set held out for early stopping each round

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")


# --- 2. Helper Functions (shared logic with mtp_random_split.py) ---
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


def make_windows(values: np.ndarray, label: int, window_size: int, stride: int):
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
    """Same leak-safe logic as mtp_random_split.py: a random-position
    contiguous block of `fraction` of the raw samples, buffered on both
    sides, windowed separately from the remainder."""
    rng = np.random.default_rng(seed)
    n_samples = len(values)
    block_len = int(round(fraction * n_samples))
    buffer = window_size

    max_start = n_samples - block_len
    if max_start <= 0:
        raise ValueError("Requested fraction is larger than this file's length.")
    block_start = int(rng.integers(0, max_start + 1))
    block_end = block_start + block_len

    block = values[block_start:block_end]
    before = values[: max(0, block_start - buffer)]
    after = values[block_end + buffer :]

    stride = max(1, window_size // 4)
    x_block, y_block = make_windows(block, label, window_size, stride)
    x_before, y_before = make_windows(before, label, window_size, stride)
    x_after, y_after = make_windows(after, label, window_size, stride)

    rem_x_parts = [a for a in (x_before, x_after) if len(a)]
    rem_y_parts = [a for a in (y_before, y_after) if len(a)]
    if rem_x_parts:
        x_remainder = np.concatenate(rem_x_parts)
        y_remainder = np.concatenate(rem_y_parts)
    else:
        x_remainder = np.empty((0, window_size, values.shape[1]), dtype=np.float32)
        y_remainder = np.empty(0, dtype=np.int64)

    return x_block, y_block, x_remainder, y_remainder


def normalize_per_window(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


class HAR_CNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.30), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x.permute(0, 2, 1)))


def train_and_eval(train_x, train_y, test_x, test_y, n_features, n_classes, seed):
    """Trains one model on (train_x, train_y), returns (test_accuracy, model)."""
    set_seed(seed)
    train_x_s, val_x_s, train_y_s, val_y_s = train_test_split(
        train_x, train_y, test_size=VAL_SHARE_OF_LABELED, random_state=seed,
        stratify=train_y if len(np.unique(train_y)) > 1 else None,
    )
    train_x_n = normalize_per_window(train_x_s)
    val_x_n = normalize_per_window(val_x_s)
    test_x_n = normalize_per_window(test_x)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x_n), torch.from_numpy(train_y_s)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x_n), torch.from_numpy(val_y_s)), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x_n), torch.from_numpy(test_y)), batch_size=BATCH_SIZE)

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
    truth, preds = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            truth.extend(yb.numpy())
            preds.extend(logits.argmax(dim=1).cpu().numpy())
    return accuracy_score(truth, preds) * 100, model


def score_pool(model, pool_x, n_features):
    pool_x_n = normalize_per_window(pool_x)
    loader = DataLoader(TensorDataset(torch.from_numpy(pool_x_n)), batch_size=BATCH_SIZE)
    all_probs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            probs = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def select_indices(strategy, pool_x, probs, round_budget, seed):
    """Returns indices (into pool_x) to move from unlabeled pool to labeled set."""
    n_pool = len(pool_x)
    round_budget = min(round_budget, n_pool)
    rng = np.random.default_rng(seed)

    if strategy == "random":
        return rng.choice(n_pool, size=round_budget, replace=False)

    if strategy == "entropy":
        scores = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        higher_is_more_uncertain = True
    elif strategy == "least_confidence":
        scores = 1 - np.max(probs, axis=1)
        higher_is_more_uncertain = True
    elif strategy == "margin":
        sorted_p = np.sort(probs, axis=1)
        scores = sorted_p[:, -1] - sorted_p[:, -2]
        higher_is_more_uncertain = False  # smaller margin = more uncertain
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    candidate_size = min(n_pool, round_budget * TOP_K_MULTIPLIER)
    order = np.argsort(scores)
    candidate_idx = order[-candidate_size:] if higher_is_more_uncertain else order[:candidate_size]

    k = min(N_CLUSTERS, len(candidate_idx))
    if k < 2:
        return candidate_idx[:round_budget]

    candidate_flat = pool_x[candidate_idx].reshape(len(candidate_idx), -1)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(candidate_flat)

    counts = np.bincount(cluster_labels, minlength=k)
    proportions = counts / counts.sum()
    alloc = np.floor(proportions * round_budget).astype(int)
    remainder = round_budget - alloc.sum()
    frac_order = np.argsort(-(proportions * round_budget - alloc))
    for i in range(remainder):
        alloc[frac_order[i % k]] += 1

    selected = []
    for c in range(k):
        members = np.flatnonzero(cluster_labels == c)
        take = min(alloc[c], len(members))
        if take > 0:
            chosen = rng.choice(members, size=take, replace=False)
            selected.extend(candidate_idx[chosen])
    return np.array(selected[:round_budget])


# --- 3. Main experiment loop ---
# results[dataset][strategy] = list of (seed, round, labeled_size, accuracy)
results = {ds: {s: [] for s in STRATEGIES} for ds in DATASETS}

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

    for seed in SEEDS:
        print(f"\n-- seed {seed} --")
        test_x_list, test_y_list = [], []
        labeled_x_list, labeled_y_list = [], []
        pool_x_list, pool_y_list = [], []

        for label, values in raw_by_activity:
            # Step A: carve out the held-out test block first.
            x_test, y_test, x_rest, y_rest = random_contiguous_split(
                values, label, TEST_FRACTION, window_size, seed=seed + label
            )
            test_x_list.append(x_test)
            test_y_list.append(y_test)

            # Step B: from the rest, carve out the initial labeled block.
            labeled_fraction_of_rest = INITIAL_LABELED_FRACTION / (1 - TEST_FRACTION)
            # random_contiguous_split expects raw samples, not windows, so
            # reconstruct an approximate raw slice size proportionally.
            rest_raw_len = int(len(values) * (1 - TEST_FRACTION))
            # Re-derive "rest" as a raw array is not directly available post-windowing,
            # so instead split windows directly here (safe: x_rest windows already
            # come from a contiguous, buffered region with no overlap into test).
            n_rest = len(y_rest)
            n_labeled = max(1, int(round(labeled_fraction_of_rest * n_rest)))
            rng = np.random.default_rng(seed + label + 999)
            perm = rng.permutation(n_rest)
            labeled_idx = perm[:n_labeled]
            pool_idx = perm[n_labeled:]
            labeled_x_list.append(x_rest[labeled_idx])
            labeled_y_list.append(y_rest[labeled_idx])
            pool_x_list.append(x_rest[pool_idx])
            pool_y_list.append(y_rest[pool_idx])

        test_x = np.concatenate(test_x_list)
        test_y = np.concatenate(test_y_list)
        base_labeled_x = np.concatenate(labeled_x_list)
        base_labeled_y = np.concatenate(labeled_y_list)
        base_pool_x = np.concatenate(pool_x_list)
        base_pool_y = np.concatenate(pool_y_list)
        original_pool_size = len(base_pool_y)
        round_budget = max(1, int(round(ROUND_BUDGET_FRACTION * original_pool_size)))
        print(f"  labeled={len(base_labeled_y)}, pool={original_pool_size}, test={len(test_y)}, round_budget={round_budget}")

        for strategy in STRATEGIES:
            labeled_x, labeled_y = base_labeled_x.copy(), base_labeled_y.copy()
            pool_x, pool_y = base_pool_x.copy(), base_pool_y.copy()

            for round_idx in range(N_ROUNDS + 1):
                acc, model = train_and_eval(labeled_x, labeled_y, test_x, test_y, n_features, n_classes, seed)
                results[dataset_name][strategy].append((seed, round_idx, len(labeled_y), acc))
                print(f"  [{strategy}] round {round_idx}: labeled={len(labeled_y)}, test_acc={acc:.2f}%")

                if round_idx == N_ROUNDS or len(pool_y) == 0:
                    break

                if strategy == "random":
                    probs = None
                else:
                    probs = score_pool(model, pool_x, n_features)
                sel_idx = select_indices(strategy, pool_x, probs, round_budget, seed=seed + round_idx)

                labeled_x = np.concatenate([labeled_x, pool_x[sel_idx]])
                labeled_y = np.concatenate([labeled_y, pool_y[sel_idx]])
                mask = np.ones(len(pool_y), dtype=bool)
                mask[sel_idx] = False
                pool_x, pool_y = pool_x[mask], pool_y[mask]

print("\nAll active-learning runs complete. Generating plots...")

# --- 4. Plots ---
STRATEGY_COLORS = {"random": "#888888", "entropy": "#1f77b4", "least_confidence": "#ff7f0e", "margin": "#2ca02c"}
STRATEGY_LABELS = {"random": "Random", "entropy": "Entropy", "least_confidence": "Least Confidence", "margin": "Margin"}


def aggregate(dataset_name, strategy):
    """Returns (labeled_sizes_per_round, mean_acc_per_round) averaged over seeds."""
    rows = results[dataset_name][strategy]
    by_round = {}
    for seed, round_idx, labeled_size, acc in rows:
        by_round.setdefault(round_idx, {"sizes": [], "accs": []})
        by_round[round_idx]["sizes"].append(labeled_size)
        by_round[round_idx]["accs"].append(acc)
    rounds_sorted = sorted(by_round.keys())
    sizes = [np.mean(by_round[r]["sizes"]) for r in rounds_sorted]
    accs = [np.mean(by_round[r]["accs"]) for r in rounds_sorted]
    return sizes, accs


# Plot 1: AL curve per dataset (accuracy vs labeled size, one line per strategy)
for ds in DATASETS:
    if not any(results[ds].values()):
        continue
    plt.figure(figsize=(7, 5))
    for strategy in STRATEGIES:
        sizes, accs = aggregate(ds, strategy)
        if sizes:
            plt.plot(sizes, accs, marker="o", label=STRATEGY_LABELS[strategy], color=STRATEGY_COLORS[strategy])
    plt.xlabel("Labeled samples")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Active Learning Curve — {ds.upper()}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"al_curve_{ds}.png", dpi=200)
    plt.close()
    print(f"Saved al_curve_{ds}.png")

# Plot 2: Final-round accuracy bar comparison per dataset
for ds in DATASETS:
    if not any(results[ds].values()):
        continue
    final_accs = []
    for strategy in STRATEGIES:
        _, accs = aggregate(ds, strategy)
        final_accs.append(accs[-1] if accs else 0)
    plt.figure(figsize=(6, 5))
    bars = plt.bar([STRATEGY_LABELS[s] for s in STRATEGIES], final_accs,
                   color=[STRATEGY_COLORS[s] for s in STRATEGIES])
    for bar, val in zip(bars, final_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}", ha="center", fontsize=9)
    plt.ylabel("Final Test Accuracy (%)")
    plt.title(f"Final-Round Accuracy — {ds.upper()}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"al_final_accuracy_{ds}.png", dpi=200)
    plt.close()
    print(f"Saved al_final_accuracy_{ds}.png")

# Plot 3: Improvement-over-random per dataset
improvement_by_dataset = {}
for ds in DATASETS:
    if not any(results[ds].values()):
        continue
    _, random_accs = aggregate(ds, "random")
    if not random_accs:
        continue
    random_final = random_accs[-1]
    improvements = []
    for strategy in STRATEGIES[1:]:
        _, accs = aggregate(ds, strategy)
        improvements.append((accs[-1] - random_final) if accs else 0)
    improvement_by_dataset[ds] = improvements
    plt.figure(figsize=(6, 5))
    bars = plt.bar([STRATEGY_LABELS[s] for s in STRATEGIES[1:]], improvements,
                   color=[STRATEGY_COLORS[s] for s in STRATEGIES[1:]])
    for bar, val in zip(bars, improvements):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:+.1f}", ha="center",
                  va="bottom" if val >= 0 else "top", fontsize=9)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Accuracy gain over Random (pp)")
    plt.title(f"Improvement over Random — {ds.upper()}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"al_improvement_{ds}.png", dpi=200)
    plt.close()
    print(f"Saved al_improvement_{ds}.png")

# Plot 4: Cross-dataset summary of improvement over random
if improvement_by_dataset:
    strategies_no_random = STRATEGIES[1:]
    x = np.arange(len(strategies_no_random))
    width = 0.2
    plt.figure(figsize=(8, 5))
    for i, ds in enumerate(improvement_by_dataset):
        plt.bar(x + (i - len(improvement_by_dataset) / 2) * width, improvement_by_dataset[ds],
                width, label=ds.upper())
    plt.xticks(x, [STRATEGY_LABELS[s] for s in strategies_no_random])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Accuracy gain over Random (pp)")
    plt.title("Cross-Dataset: Improvement over Random by Strategy")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("al_cross_dataset_summary.png", dpi=200)
    plt.close()
    print("Saved al_cross_dataset_summary.png")

print("\nDone.")