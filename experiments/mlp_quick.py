"""
MLP Raw vs Combined - closed-set version (StratifiedShuffleSplit)
Reproduce original Untitled4/5 setup: same split type as the 83% result.
Then run subject-wise (StratifiedGroupKFold) as the proper protocol comparison.
"""
import random, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, "/home/dryjins/works/ibc-wavelet-benchmark")
from features import get_feature

SEED = 42
EPOCHS = 500
DEVICE = "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load
X_raw = pd.read_csv("data/processed/ibc_processed.csv").values.astype(np.float32)
y = pd.read_csv("data/labels_filtered.csv")["subject_id"].values.astype(int)
print(f"Spectra: {X_raw.shape}, {len(np.unique(y))} subjects")

dwt_feat = get_feature("dwt_db4_l2")
X_dwt_all = dwt_feat.fit(X_raw, y).transform(X_raw)
print(f"DWT: {X_dwt_all.shape}")

class MLP(nn.Module):
    def __init__(self, d, C, h=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(h, h), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(h, C),
        )
    def forward(self, x): return self.net(x)

def run_mlp(X_tr_raw, y_tr, X_te_raw, y_te, X_dwt_tr, X_dwt_te, use_dwt, label):
    classes = np.unique(y_tr)
    to_index = {int(c): i for i, c in enumerate(classes)}
    to_label = {i: int(c) for i, c in enumerate(classes)}
    y_tr_enc = np.vectorize(lambda t: to_index[int(t)])(y_tr)

    mask = np.isin(y_te, classes)
    X_q, y_q = X_te_raw[mask], y_te[mask]
    X_q_dwt = X_dwt_te[mask]
    y_q_enc = np.vectorize(lambda t: to_index[int(t)])(y_q)
    print(f"  [{label}] query: {len(X_q)}/{len(X_te_raw)}, classes: {len(np.unique(y_q))}")

    if use_dwt:
        X_tr_feat = np.hstack([X_tr_raw, X_dwt_tr])
        X_te_feat = np.hstack([X_q, X_q_dwt])
        dim = X_tr_feat.shape[1]
    else:
        X_tr_feat = X_tr_raw
        X_te_feat = X_q
        dim = X_tr_raw.shape[1]

    sc = StandardScaler()
    X_trn = sc.fit_transform(X_tr_feat)
    X_vs = sc.transform(X_te_feat)

    C = len(classes)
    model = MLP(dim, C, h=256, p=0.3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(
        torch.from_numpy(X_trn).float(),
        torch.from_numpy(y_tr_enc).long()),
        batch_size=64, shuffle=True, drop_last=False)
    Xv = torch.from_numpy(X_vs).float()
    yv = torch.from_numpy(y_q_enc).long()

    best_state, best_loss = None, float('inf')
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            l = crit(model(Xv), yv).item()
            preds = model(Xv).argmax(1).cpu().numpy()
            acc = (preds == y_q_enc).mean()
        if l < best_loss:
            best_loss = l
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 100 == 0:
            print(f"  [{label}] Ep {ep}: loss={l:.4f} acc={acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_preds = model(Xv).argmax(1).cpu().numpy()
        final_acc = (final_preds == y_q_enc).mean()
    print(f"  [{label}] Final acc: {final_acc:.4f}")
    return final_acc

# =============================================
# Test A: StratifiedShuffleSplit (closed-set, like original notebooks)
# =============================================
print("\n" + "="*60)
print("TEST A: StratifiedShuffleSplit (closed-set)")
print("="*60)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, va_idx = next(sss.split(X_raw, y))
print(f"Train: {len(tr_idx)}, Val: {len(va_idx)}")

X_tr_raw, X_va_raw = X_raw[tr_idx], X_raw[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]
X_dwt_tr = X_dwt_all[tr_idx]
X_dwt_va = X_dwt_all[va_idx]

print("Unique classes in train:", len(np.unique(y_tr)), "val:", len(np.unique(y_va)))

acc_raw_closed = run_mlp(X_tr_raw.copy(), y_tr.copy(),
                          X_va_raw.copy(), y_va.copy(),
                          X_dwt_tr.copy(), X_dwt_va.copy(),
                          use_dwt=False, label="Raw_closed")

acc_comb_closed = run_mlp(X_tr_raw.copy(), y_tr.copy(),
                           X_va_raw.copy(), y_va.copy(),
                           X_dwt_tr.copy(), X_dwt_va.copy(),
                           use_dwt=True, label="Comb_closed")

print(f"\n  Closed-set: Raw={acc_raw_closed:.4f}, Combined={acc_comb_closed:.4f}")

# =============================================
# Test B: StratifiedGroupKFold with enrollment (subject-wise)
# =============================================
print("\n" + "="*60)
print("TEST B: StratifiedGroupKFold + enrollment (subject-wise)")
print("="*60)

N_ENROLL = 5
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
tr_idx, te_idx = next(sgkf.split(X_raw, y, groups=y))
print(f"Group split: train={len(tr_idx)}, test={len(te_idx)}")

y_tr_all, y_te_all = y[tr_idx], y[te_idx]

enrolled_abs, remaining_abs = [], []
for sub in np.unique(y_te_all):
    sub_pos = np.where(y_te_all == sub)[0]
    abs_pos = te_idx[sub_pos]
    n = len(abs_pos)
    if n <= N_ENROLL:
        enrolled_abs.extend(abs_pos.tolist())
        continue
    enrolled_pos = random.sample(list(sub_pos), N_ENROLL)
    remaining_pos = [i for i in sub_pos if i not in enrolled_pos]
    enrolled_abs.extend([te_idx[i] for i in enrolled_pos])
    remaining_abs.extend([te_idx[i] for i in remaining_pos])

X_tr_raw = np.vstack([X_raw[tr_idx], X_raw[enrolled_abs]])
y_tr = np.concatenate([y_tr_all, y[enrolled_abs]])
X_te_raw = X_raw[remaining_abs]
y_te = y[remaining_abs]
X_dwt_tr = X_dwt_all[np.concatenate([tr_idx, enrolled_abs])]
X_dwt_te = X_dwt_all[remaining_abs]

print(f"After enrollment: train={len(X_tr_raw)}, query={len(X_te_raw)}")

acc_raw_open = run_mlp(X_tr_raw.copy(), y_tr.copy(),
                        X_te_raw.copy(), y_te.copy(),
                        X_dwt_tr.copy(), X_dwt_te.copy(),
                        use_dwt=False, label="Raw_open")

acc_comb_open = run_mlp(X_tr_raw.copy(), y_tr.copy(),
                          X_te_raw.copy(), y_te.copy(),
                          X_dwt_tr.copy(), X_dwt_te.copy(),
                          use_dwt=True, label="Comb_open")

print(f"\n  Subject-wise: Raw={acc_raw_open:.4f}, Combined={acc_comb_open:.4f}")

# =============================================
# Summary
# =============================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Closed-set (StratifiedShuffleSplit):")
print(f"  Raw MLP:      {acc_raw_closed:.4f}")
print(f"  Combined MLP: {acc_comb_closed:.4f}")
print(f"Subject-wise (StratifiedGroupKFold + enrollment):")
print(f"  Raw MLP:      {acc_raw_open:.4f}")
print(f"  Combined MLP: {acc_comb_open:.4f}")

diff_closed = acc_comb_closed - acc_raw_closed
diff_open = acc_comb_open - acc_raw_open
print(f"\nDWT fusion benefit:")
print(f"  Closed-set:   Combined - Raw = {diff_closed:+.4f}")
print(f"  Subject-wise:  Combined - Raw = {diff_open:+.4f}")

out = {
    "closed_set": {"raw": float(acc_raw_closed), "combined": float(acc_comb_closed), "diff": float(diff_closed)},
    "subject_wise": {"raw": float(acc_raw_open), "combined": float(acc_comb_open), "diff": float(diff_open)},
}
with open("results/mlp_raw_vs_combined.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved results/mlp_raw_vs_combined.json")