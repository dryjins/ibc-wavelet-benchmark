"""
Clean experiment: Closed-set Raw MLP vs Combined MLP
Issue 1 response: Is the 83% MLP result due to feature fusion or model capacity?

Protocol: StratifiedShuffleSplit (same as original notebooks producing 83%)
5 seeds: 42, 123, 456, 789, 999
Same MLP architecture as paper (2-layer, 256 hidden, ReLU, dropout=0.3)

Compares:
  (A) Raw spectrum (256-dim) + MLP
  (B) Combined (raw 256 + DWT stats 12-dim = 268-dim) + MLP

Outputs: mean ± std across 5 seeds, per-seed values, JSON result file
"""

import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(0, "/home/dryjins/works/ibc-wavelet-benchmark")
from features import get_feature

# =============================================================================
# Config
# =============================================================================
SEEDS = [42, 123, 456, 789, 999]
EPOCHS = 1000
BATCH_SIZE = 64
LR = 5e-4
WD = 1e-3
HIDDEN = 256
DROPOUT = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RAW_CSV = "data/processed/ibc_processed.csv"
LABS_CSV = "data/labels_filtered.csv"
OUT_JSON = "results/mlp_raw_vs_combined_clean.json"

print(f"Device: {DEVICE}")
print(f"Seeds: {SEEDS}")

# =============================================================================
# Load data
# =============================================================================
X_raw = pd.read_csv(RAW_CSV).values.astype(np.float32)
y = pd.read_csv(LABS_CSV)["subject_id"].values.astype(int)
print(f"Spectra: {X_raw.shape}, {len(np.unique(y))} subjects")

dwt_feat = get_feature("dwt_db4_l2")  # level-2: 3 sub-bands x 4 stats = 12 features
X_dwt_all = dwt_feat.fit(X_raw, y).transform(X_raw)
print(f"DWT stats: {X_dwt_all.shape}")

# =============================================================================
# MLP model
# =============================================================================
class MLP(nn.Module):
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_closed_mlp(X_train, y_train, X_val, y_val,
                     use_dwt: bool, label: str) -> float:
    """Train MLP on training data, evaluate on val. Return val accuracy."""

    # Encode train labels to 0..C-1
    classes = np.unique(y_train)
    to_index = {int(c): i for i, c in enumerate(classes)}
    to_label = {i: int(c) for i, c in enumerate(classes)}

    y_train_enc = np.vectorize(lambda t: to_index[int(t)])(y_train)

    # Encode val labels using same mapping; drop samples from unseen classes
    val_mask = np.isin(y_val, classes)
    X_v, y_v = X_val[val_mask], y_val[val_mask]
    y_v_enc = np.vectorize(lambda t: to_index[int(t)])(y_v)

    # Build feature matrix
    dim = X_train.shape[1]

    # Standardize using train stats only
    scaler = StandardScaler()
    X_trn = scaler.fit_transform(X_train)
    X_vs = scaler.transform(X_v)

    # Tensors
    Xt = torch.from_numpy(X_trn).float().to(DEVICE)
    yt = torch.from_numpy(y_train_enc).long().to(DEVICE)
    Xv = torch.from_numpy(X_vs).float().to(DEVICE)
    yv = torch.from_numpy(y_v_enc).long().to(DEVICE)

    n_classes = len(classes)
    model = MLP(dim, n_classes, hidden=HIDDEN, p=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE,
                     shuffle=True, drop_last=False)

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
            val_acc = (preds == y_v_enc).mean()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 200 == 0:
            print(f"    [{label}] Ep {epoch:04d}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_preds = model(Xv).argmax(dim=1).cpu().numpy()
        final_acc = (final_preds == y_v_enc).mean()

    return float(final_acc)


# =============================================================================
# Run 5-seed closed-set evaluation
# =============================================================================
results = {
    "raw": {"seeds": [], "accs": []},
    "combined": {"seeds": [], "accs": []},
}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(sss.split(X_raw, y))

    X_tr_raw, X_va_raw = X_raw[tr_idx], X_raw[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    X_dwt_tr = X_dwt_all[tr_idx]
    X_dwt_va = X_dwt_all[va_idx]

    print(f"  Train: {len(X_tr_raw)}, Val: {len(X_va_raw)}")
    print(f"  Train classes: {len(np.unique(y_tr))}, Val classes: {len(np.unique(y_va))}")

    # Raw MLP
    print("  Raw MLP:")
    acc_raw = train_closed_mlp(
        X_tr_raw.copy(), y_tr.copy(),
        X_va_raw.copy(), y_va.copy(),
        use_dwt=False, label="Raw")
    results["raw"]["seeds"].append(int(seed))
    results["raw"]["accs"].append(float(acc_raw))
    print(f"  -> Raw accuracy: {acc_raw:.4f}")

    # Combined MLP
    print("  Combined MLP:")
    X_tr_comb = np.hstack([X_tr_raw, X_dwt_tr])
    X_va_comb = np.hstack([X_va_raw, X_dwt_va])
    acc_comb = train_closed_mlp(
        X_tr_comb, y_tr,
        X_va_comb, y_va,
        use_dwt=True, label="Comb")
    results["combined"]["seeds"].append(int(seed))
    results["combined"]["accs"].append(float(acc_comb))
    print(f"  -> Combined accuracy: {acc_comb:.4f}")

    print(f"  >> Diff (Comb - Raw): {acc_comb - acc_raw:+.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("FINAL RESULTS (5-seed closed-set)")
print("="*60)

for name, res in results.items():
    accs = np.array(res["accs"])
    print(f"\n{name.upper()} MLP:")
    for s, a in zip(res["seeds"], res["accs"]):
        print(f"  seed {s}: {a:.4f}")
    print(f"  Mean: {accs.mean():.4f} ± {accs.std():.4f}")

raw_mean = np.mean(results["raw"]["accs"])
raw_std = np.std(results["raw"]["accs"])
comb_mean = np.mean(results["combined"]["accs"])
comb_std = np.std(results["combined"]["accs"])
diff = comb_mean - raw_mean

print(f"\nComparison:")
print(f"  Raw MLP:       {raw_mean:.4f} ± {raw_std:.4f}")
print(f"  Combined MLP:  {comb_mean:.4f} ± {comb_std:.4f}")
print(f"  Diff:          {diff:+.4f}")

if diff < 0.01:
    interp = "model_capacity_dominates"
    print("  Interpretation: model capacity dominates (Raw ~= Combined)")
elif diff > 0.02:
    interp = "dwt_fusion_beneficial"
    print("  Interpretation: DWT fusion provides additional discriminative power")
else:
    interp = "similar_results"
    print("  Interpretation: results are similar; cannot strongly favor either claim")

# Save JSON
output = {
    "raw_mlp": {
        "mean": float(raw_mean),
        "std": float(raw_std),
        "per_seed": {str(s): float(a) for s, a in zip(results["raw"]["seeds"], results["raw"]["accs"])},
    },
    "combined_mlp": {
        "mean": float(comb_mean),
        "std": float(comb_std),
        "per_seed": {str(s): float(a) for s, a in zip(results["combined"]["seeds"], results["combined"]["accs"])},
    },
    "diff_combined_minus_raw": float(diff),
    "interpretation": interp,
}
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUT_JSON}")
