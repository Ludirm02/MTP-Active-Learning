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

# --- Configuration ---
BASE_DIR = Path("al_mtp_data")
DATASETS = ["ms", "hhar", "pamap2", "ucihar"]
WINDOW_SIZE = 128
WINDOW_STRIDE = 128 # Prevents data leakage
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.10
TRAINING_FRACTIONS = [0.10, 0.20, 0.40]
SEEDS = [32, 42, 72]
MAX_EPOCHS = 75
PATIENCE = 12
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# --- Helper Functions (From your corrected script) ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    numeric = frame.select_dtypes(include=[np.number]).copy()
    return numeric.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").fillna(0.0)

def make_windows_for_activity(values: np.ndarray, label: int):
    windows = [values[start : start + WINDOW_SIZE] for start in range(0, len(values) - WINDOW_SIZE + 1, WINDOW_STRIDE)]
    if not windows:
        return np.empty((0, WINDOW_SIZE, values.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.asarray(windows, dtype=np.float32), np.full(len(windows), label, dtype=np.int64)

def scale_windows(train_x, *other_sets):
    scaler = StandardScaler()
    n_features = train_x.shape[-1]
    transformed = [scaler.fit_transform(train_x.reshape(-1, n_features)).reshape(train_x.shape).astype(np.float32)]
    for values in other_sets:
        transformed.append(scaler.transform(values.reshape(-1, n_features)).reshape(values.shape).astype(np.float32))
    return transformed

class BetterCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.30), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x.permute(0, 2, 1)))

# --- Main Execution Loop ---
final_plot_data = {ds: {frac: [] for frac in TRAINING_FRACTIONS} for ds in DATASETS}

for dataset_name in DATASETS:
    print(f"\n{'='*40}\nPROCESSING DATASET: {dataset_name.upper()}\n{'='*40}")
    data_dir = BASE_DIR / dataset_name / "data"
    csv_paths = sorted(p for p in data_dir.glob("*.csv") if p.name.lower() not in {"std.csv", "mean.csv", "metadata.csv", "labels.csv"})
    
    if not csv_paths:
        print(f"Skipping {dataset_name} - No CSVs found.")
        continue

    class_names = [path.stem for path in csv_paths]
    window_parts, label_parts = [], []
    expected_features = None

    for label, path in enumerate(csv_paths):
        values = numeric_features(pd.read_csv(path)).to_numpy(dtype=np.float32)
        if expected_features is None: expected_features = values.shape[1]
        x_part, y_part = make_windows_for_activity(values, label)
        if len(x_part):
            window_parts.append(x_part)
            label_parts.append(y_part)

    x_windows = np.concatenate(window_parts)
    y_windows = np.concatenate(label_parts)
    
    all_indices = np.arange(len(y_windows))
    train_val_idx, test_idx = train_test_split(all_indices, test_size=TEST_FRACTION, stratify=y_windows, random_state=2026)
    train_pool_idx, val_idx = train_test_split(train_val_idx, test_size=VALIDATION_FRACTION / (1 - TEST_FRACTION), stratify=y_windows[train_val_idx], random_state=2027)

    for fraction in TRAINING_FRACTIONS:
        print(f"\n--- Training with {int(fraction*100)}% Data ---")
        requested = round(fraction * len(y_windows))
        
        for seed in SEEDS:
            set_seed(seed)
            chosen_idx, _ = train_test_split(train_pool_idx, train_size=requested, stratify=y_windows[train_pool_idx], random_state=seed)
            
            train_x, val_x, test_x = scale_windows(x_windows[chosen_idx], x_windows[val_idx], x_windows[test_idx])
            
            train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(y_windows[chosen_idx])), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(y_windows[val_idx])), batch_size=BATCH_SIZE)
            test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x), torch.from_numpy(y_windows[test_idx])), batch_size=BATCH_SIZE)

            model = BetterCNN(x_windows.shape[-1], len(class_names)).to(device)
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        val_loss += criterion(model(x_batch.to(device)), y_batch.to(device)).item() * len(y_batch)
                val_loss /= len(val_loader.dataset)
                scheduler.step(val_loss)

                if val_loss < best_loss - 1e-4:
                    best_loss, best_weights, stale = val_loss, copy.deepcopy(model.state_dict()), 0
                else:
                    stale += 1
                    if stale >= PATIENCE: break

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
            print(f"Seed {seed}: Accuracy = {acc:.2f}%")

print("\nAll training complete! Ready to plot.")