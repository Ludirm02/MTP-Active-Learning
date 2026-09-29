import os
import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --- 1. Configuration ---
BASE_DIR = Path("al_mtp_data")
DATASETS = ["ms", "hhar", "pamap2", "ucihar"]
TRAINING_FRACTIONS = [0.10, 0.20, 0.40]
SEEDS = [32, 42, 72]
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 128

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
    """Removes junk columns and cleans missing/infinite values."""
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    numeric = frame.select_dtypes(include=[np.number]).copy()
    return numeric.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").fillna(0.0)

def make_windows(values: np.ndarray, label: int, window_size: int):
    """Slices time-series data into NON-OVERLAPPING windows."""
    windows = [values[start : start + window_size] for start in range(0, len(values) - window_size + 1, window_size)]
    if not windows:
        return np.empty((0, window_size, values.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.asarray(windows, dtype=np.float32), np.full(len(windows), label, dtype=np.int64)

def nested_stratified_subsets(labels: np.ndarray, total_windows: int, seed: int):
    """Return class-balanced subsets where 10% is inside 20%, inside 40%."""
    rng = np.random.default_rng(seed)
    classes, class_counts = np.unique(labels, return_counts=True)
    shuffled_by_class = {
        class_id: rng.permutation(np.flatnonzero(labels == class_id))
        for class_id in classes
    }
    subsets, previous = {}, set()

    for fraction in TRAINING_FRACTIONS:
        requested = min(round(fraction * total_windows), len(labels))
        ideal = requested * class_counts / len(labels)
        quotas = np.floor(ideal).astype(int)
        for position in np.argsort(ideal - quotas)[::-1][: requested - quotas.sum()]:
            quotas[position] += 1
        selected = np.concatenate([
            shuffled_by_class[class_id][:quota]
            for class_id, quota in zip(classes, quotas)
        ])
        rng.shuffle(selected)
        if not previous.issubset(set(selected)):
            raise RuntimeError("Training subsets must be nested.")
        subsets[fraction] = selected
        previous = set(selected)
    return subsets

# --- 3. Stronger 1D CNN ---
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
            nn.AdaptiveAvgPool1d(1), # This makes it independent of window size!
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x.permute(0, 2, 1)))

# --- 4. Main Execution Loop ---
final_plot_data = {ds: {frac: [] for frac in TRAINING_FRACTIONS} for ds in DATASETS}

for dataset_name in DATASETS:
    print(f"\n{'='*50}\nPROCESSING DATASET: {dataset_name.upper()}\n{'='*50}")
    
    # Dynamic Window Size: PAMAP2 needs a longer window due to 100Hz sampling
    WS = 512 if dataset_name == "pamap2" else 128
    
    data_dir = BASE_DIR / dataset_name / "data"
    csv_paths = sorted(p for p in data_dir.glob("*.csv") if p.name.lower() not in {"std.csv", "mean.csv", "metadata.csv", "labels.csv"})
    
    if not csv_paths:
        print(f"Skipping {dataset_name} - No CSVs found.")
        continue

    class_names = [path.stem for path in csv_paths]
    n_classes = len(class_names)
    
    all_x, all_y = [], []

    # Create non-overlapping windows
    for label, path in enumerate(csv_paths):
        values = numeric_features(pd.read_csv(path)).to_numpy(dtype=np.float32)
        x_part, y_part = make_windows(values, label, WS)
        if len(x_part):
            all_x.append(x_part)
            all_y.append(y_part)

    X_full = np.concatenate(all_x)
    y_full = np.concatenate(all_y)
    total_windows = len(y_full)

    # ZERO-LEAKAGE GLOBAL SPLITTING
    # 1. Split 20% off for testing
    X_train_val, test_x, y_train_val, test_y = train_test_split(X_full, y_full, test_size=0.20, stratify=y_full, random_state=2026)
    
    # 2. Split 10% (of the total) off for validation. (0.10 / 0.80 = 0.125)
    train_pool_x, val_x, train_pool_y, val_y = train_test_split(X_train_val, y_train_val, test_size=0.125, stratify=y_train_val, random_state=2027)

    # Scale the features (fit ONLY on train_pool)
    scaler = StandardScaler()
    n_features = train_pool_x.shape[-1]
    scaler.fit(train_pool_x.reshape(-1, n_features))
    
    train_pool_x = scaler.transform(train_pool_x.reshape(-1, n_features)).reshape(train_pool_x.shape).astype(np.float32)
    val_x = scaler.transform(val_x.reshape(-1, n_features)).reshape(val_x.shape).astype(np.float32)
    test_x = scaler.transform(test_x.reshape(-1, n_features)).reshape(test_x.shape).astype(np.float32)

    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)), batch_size=BATCH_SIZE)
    
    subsets_by_seed = {
        seed: nested_stratified_subsets(train_pool_y, total_windows, seed)
        for seed in SEEDS
    }
    
    test_counts = np.bincount(test_y, minlength=n_classes)
    majority_baseline = test_counts.max() / len(test_y) * 100
    print(f"Majority-class baseline: {majority_baseline:.2f}%")

    for fraction in TRAINING_FRACTIONS:
        print(f"\n--- Training with {int(fraction*100)}% Data ({dataset_name.upper()}) ---")
        
        for seed in SEEDS:
            set_seed(seed)
            chosen_idx = subsets_by_seed[seed][fraction]
            
            train_x_sub = train_pool_x[chosen_idx]
            train_y_sub = train_pool_y[chosen_idx]
            
            train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x_sub), torch.from_numpy(train_y_sub)), batch_size=BATCH_SIZE, shuffle=True)

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
                val_loss = 0
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
            truth, predictions = [], []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    logits = model(x_batch.to(device))
                    truth.extend(y_batch.numpy())
                    predictions.extend(logits.argmax(dim=1).cpu().numpy())
            
            acc = accuracy_score(truth, predictions) * 100
            final_plot_data[dataset_name][fraction].append(acc)
            print(f"Seed {seed}: Accuracy = {acc:.2f}% (Epochs: {epoch})")

print("\nAll training complete! Generating the plot...")

# --- 5. Plotting ---
datasets_display = ["MotionSense", "HHAR", "PAMAP2", "UCI HAR"]
dataset_keys = ["ms", "hhar", "pamap2", "ucihar"]
fractions = [0.10, 0.20, 0.40]

means = {ds: [] for ds in dataset_keys}
stds = {ds: [] for ds in dataset_keys}

for ds in dataset_keys:
    for frac in fractions:
        accuracies = final_plot_data[ds][frac]
        means[ds].append(np.mean(accuracies))
        stds[ds].append(np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0)

x = np.arange(len(fractions))
width = 0.2  
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
hatches = ['//', '\\\\', '--', 'xx']

fig, ax = plt.subplots(figsize=(10, 6), edgecolor='black', linewidth=1)

for i, ds in enumerate(dataset_keys):
    bars = ax.bar(
        x + (i - 1.5) * width, 
        means[ds], 
        width, 
        label=datasets_display[i],
        color=colors[i], 
        hatch=hatches[i], 
        edgecolor='black',
        yerr=stds[ds], 
        capsize=4,
        error_kw={'elinewidth': 1, 'capthick': 1}
    )
    
    for bar, err in zip(bars, stds[ds]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height + err + 0.5, 
            f'{height:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=9, 
            fontweight='bold'
        )

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Training Data Percentage', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['10%', '20%', '40%'], fontsize=12, fontweight='bold')
ax.set_ylim(0, 105)

ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Datasets', title_fontsize=12, fontsize=10, loc='upper center', ncol=4, framealpha=1)

plt.tight_layout()
plt.savefig('combined_accuracy_plot_final.png', dpi=300, bbox_inches='tight')
print("\nPlot saved successfully as 'combined_accuracy_plot_final.png'!")