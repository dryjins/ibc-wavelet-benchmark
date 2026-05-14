# eval_utils.py
from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

def subject_folds(X, y, groups, n_splits=5, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(X, y, groups=groups):
        yield tr, te

def enroll_per_subject(X_te, y_te, k=5, rng=None):
    rng = rng or np.random.default_rng(42)
    keep = np.ones(len(y_te), dtype=bool)
    X_en, y_en = [], []
    for sid in np.unique(y_te):
        idx = np.where(y_te == sid)[0]
        if len(idx) <= k: continue
        chosen = rng.choice(idx, size=k, replace=False)
        keep[chosen] = False
        X_en.append(X_te[chosen]); y_en.append(y_te[chosen])
    X_en = np.vstack(X_en) if X_en else np.empty((0, X_te.shape[1]))
    y_en = np.concatenate(y_en) if y_en else np.empty((0,))
    return X_en, y_en, X_te[keep], y_te[keep]

def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))

def far_frr(y_true, y_pred) -> tuple[float, float]:
    cm = confusion_matrix(y_true, y_pred)
    fars, frrs = [], []
    for i in range(len(cm)):
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tn = cm.sum() - (fp + fn + cm[i, i])
        tp = cm[i, i]
        fars.append(fp / (fp + tn) if (fp + tn) else 0.0)
        frrs.append(fn / (fn + tp) if (fn + tp) else 0.0)
    return float(np.mean(fars)), float(np.mean(frrs))

def mean_eer_ovr(y_true, y_score, classes) -> float | float("nan"):
    try:
        y_bin = (y_true.reshape(-1,1) == classes.reshape(1,-1)).astype(int)
        eers = []
        for i in range(y_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            fnr = 1 - tpr
            eers.append(float(fpr[np.nanargmin(np.abs(fnr - fpr))]))
        return float(np.mean(eers))
    except Exception:
        return float("nan")
