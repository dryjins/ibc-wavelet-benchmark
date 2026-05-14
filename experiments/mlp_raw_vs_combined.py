"""
Raw + MLP vs Combined + MLP Comparison
Issue 1 Response: Is the 83% MLP result due to feature fusion or model capacity?

Protocol: Subject-wise evaluation with enrollment
- 5-fold StratifiedGroupKFold (subject_id as group)
- For each test subject: 5 samples added to training (enrollment), remaining samples form query set
- Same MLP architecture and hyperparameters as Combined + MLP in the paper
- Seeds: 42, 123, 456, 789, 999 (matching paper)

Compares:
  (A) Raw spectrum (256-dim) + MLP
  (B) Combined (raw 256 + DWT stats 28 = 284-dim) + MLP
"""

import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, "/home/dryjins/works/ibc-wavelet-benchmark")
from features import get_feature

# =============================================================================
# Config
# =============================================================================
SEEDS = [42]  # start with one seed for quick check
N_ENROLL = 5
EPOCHS = 1000
BATCH_SIZE = 64
LR = 5e-4
WD = 1e-3
HIDDEN = 256
DROPOUT = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROC_CSV = "data/processed/ibc_processed.csv"
LABS_CSV = "data/labels_filtered.csv"
OUT_JSON = "results/mlp_raw_vs_combined.json"

# Reproducibility
random.seed(SEEDS[0])
np.random.seed(SEEDS[0])
torch.manual_seed(SEEDS[0])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEEDS[0])

print(f"Device: {DEVICE}")

# =============================================================================
# Load data
# =============================================================================
X_raw = pd.read_csv(PROC_CSV).values.astype(np.float32)   # (N, 256)
y = pd.read_csv(LABS_CSV)["subject_id"].values.astype(int)
N, D_raw = X_raw.shape
print(f"Spectra: {X_raw.shape}, Labels: {y.shape}, Subjects: {len(np.unique(y))}")

# DWT stats (db4, level=2)
dwt_feat = get_feature("dwt_db4_l2")
X_dwt_all = dwt_feat.fit(X_raw, y).transform(X_raw)   # (N, 28)
D_dwt = X_dwt_all.shape[1]
print(f"DWT stats: {X_dwt_all.shape}")

# =============================================================================
# MLP architecture
# =============================================================================
class MLPNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)

# =============================================================================
# Training function
# =============================================================================
def train_mlp(X_train_raw, y_train, X_query_raw, y_query,
              X_dwt_train, X_dwt_query,
              n_classes, use_dwt=False):

    # Encode labels
    classes = np.unique(y_train)
    to_index = {int(c): i for i, c in enumerate(classes)}
    to_label = {i: int(c) for i, c in enumerate(classes)}
    def encode(labels):
        return np.vectorize(lambda t: to_index[int(t)])(labels)
    def decode(indices):
        return np.vectorize(lambda i: to_label[int(i)])(indices)

    y_train_enc = encode(y_train)
    val_mask = np.isin(y_query, classes)
    X_q, y_q = X_query_raw[val_mask], y_query[val_mask]
    y_q_enc = encode(y_q)
    X_q_dwt = X_dwt_query[val_mask]

    # Build feature matrix
    if use_dwt:
        X_tr = np.hstack([X_train_raw, X_dwt_train])
        X_va = np.hstack([X_q, X_q_dwt])
        in_dim = D_raw + D_dwt
    else:
        X_tr = X_train_raw
        X_va = X_q
        in_dim = D_raw

    # Standardize
    scaler = StandardScaler()
    X_trn = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    # Tensors
    Xt = torch.from_numpy(X_trn).float().to(DEVICE)
    yt = torch.from_numpy(y_train_enc).long().to(DEVICE)
    Xv = torch.from_numpy(X_va_s).float().to(DEVICE)
    yv = torch.from_numpy(y_q_enc).long().to(DEVICE)

    model = MLPNet(in_dim, n_classes, hidden=HIDDEN, p=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    best_state, best_val_loss = None, float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), yv).item()
            preds = model(Xv).argmax(dim=1).cpu().numpy()
            val_acc = (preds == y_q_enc).mean()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 200 == 0:
            print(f"    Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_preds = model(Xv).argmax(dim=1).cpu().numpy()
        final_acc = (final_preds == y_q_enc).mean()

    return final_acc

# =============================================================================
# Run 5-fold subject-wise evaluation with enrollment
# =============================================================================
results = {
    "raw": {"accuracy": []},
    "combined": {"accuracy": []},
}

for seed in SEEDS:
    print(f"\n=== Seed {seed} ===")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X_raw, y, groups=y)):
        y_train_full = y[train_idx]
        y_test_full = y[test_idx]

        # Enrollment: select 5 samples per test subject
        # enrolled_orig_idx: original dataset indices for enrollment
        enrolled_orig_idx = []
        remaining_orig_idx = []  # for query set

        for sub in np.unique(y_test_full):
            sub_idx_in_test = np.where(y_test_full == sub)[0]
            n_sub = len(sub_idx_in_test)
            if n_sub <= N_ENROLL:
                enrolled_orig_idx.extend(test_idx[sub_idx_in_test].tolist())
                continue
            enrolled = random.sample(list(sub_idx_in_test), N_ENROLL)
            remaining = [i for i in sub_idx_in_test if i not in enrolled]
            enrolled_orig_idx.extend([test_idx[i] for i in enrolled])
            remaining_orig_idx.extend([test_idx[i] for i in remaining])

        # Build train set: original train + enrolled samples
        X_train_raw = np.vstack([X_raw[train_idx], X_raw[enrolled_orig_idx]])
        y_train = np.concatenate([y_train_full, y[enrolled_orig_idx]])

        # Query set: remaining (non-enrolled) test samples
        X_query_raw = X_raw[remaining_orig_idx]
        y_query = y[remaining_orig_idx]

        # DWT for train (original + enrolled) and query (remaining)
        X_dwt_train = X_dwt_all[np.concatenate([train_idx, enrolled_orig_idx])]
        X_dwt_query = X_dwt_all[remaining_orig_idx]

        n_classes = len(np.unique(y_train))
        n_query_subjects = len(np.unique(y_query))
        print(f"  Fold {fold_idx+1}: train={len(X_train_raw)}, query={len(X_query_raw)}, "
              f"n_classes_train={n_classes}, query_subjects={n_query_subjects}")

        # Raw MLP
        acc_raw = train_mlp(
            X_train_raw.copy(), y_train.copy(),
            X_query_raw.copy(), y_query.copy(),
            X_dwt_train.copy(), X_dwt_query.copy(),
            n_classes, use_dwt=False)
        results["raw"]["accuracy"].append(acc_raw)

        # Combined MLP
        acc_comb = train_mlp(
            X_train_raw.copy(), y_train.copy(),
            X_query_raw.copy(), y_query.copy(),
            X_dwt_train.copy(), X_dwt_query.copy(),
            n_classes, use_dwt=True)
        results["combined"]["accuracy"].append(acc_comb)

        print(f"    Raw MLP: {acc_raw:.4f}, Combined MLP: {acc_comb:.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
for feat_name, res in results.items():
    accs = res["accuracy"]
    print(f"{feat_name:10s}: mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
    for i, a in enumerate(accs):
        print(f"              Fold {i+1}: {a:.4f}")

print("\nComparison:")
raw_mean = np.mean(results["raw"]["accuracy"])
comb_mean = np.mean(results["combined"]["accuracy"])
diff = comb_mean - raw_mean
print(f"  Combined - Raw = {diff:+.4f}")
if diff < 0.01:
    print("  Interpretation: model capacity dominates (Raw ~= Combined)")
elif diff > 0.02:
    print("  Interpretation: DWT fusion provides additional discriminative power")
else:
    print("  Interpretation: results are similar; cannot strongly favor either claim")

# Save
output = {
    "raw_mlp": {
        "mean": float(np.mean(results["raw"]["accuracy"])),
        "std": float(np.std(results["raw"]["accuracy"])),
        "per_fold": [float(a) for a in results["raw"]["accuracy"]],
    },
    "combined_mlp": {
        "mean": float(np.mean(results["combined"]["accuracy"])),
        "std": float(np.std(results["combined"]["accuracy"])),
        "per_fold": [float(a) for a in results["combined"]["accuracy"]],
    },
    "diff_combined_minus_raw": float(diff),
    "interpretation": (
        "model capacity dominates (Raw ~= Combined)" if diff < 0.01
        else ("DWT fusion provides additional discriminative power" if diff > 0.02
              else "results are similar; cannot strongly favor either claim")
    ),
}
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {OUT_JSON}")