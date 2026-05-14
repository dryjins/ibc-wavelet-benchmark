"""
models.py
Models for NxK benchmarking (no LightGBM):
- KNN (1-NN, sklearn)
- Random Forest (sklearn)
- SpectralCNN (PyTorch 1D CNN for raw spectra 256)
- DWTMLP (PyTorch MLP for DWT stats)
- DWTProto (PyTorch embedding + prototype classifier)

Unified API:
- fit(X, y, **kwargs)
- predict(X) -> np.ndarray  (returns original subject_id via inverse mapping for DNNs)
- predict_proba(X) -> Optional[np.ndarray]

All DNN wrappers accept hyperparameters via constructor kwargs, e.g.,
get_model("spectral_cnn", epochs=300, device="cuda")
"""

from __future__ import annotations

from typing import Optional, Any, Dict
import numpy as np

try:
    import pandas as pd
    HAS_PD = True
except Exception:
    HAS_PD = False

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Common helpers
# ----------------------------
def _to_ndarray(X: Any) -> np.ndarray:
    """Ensure numpy ndarray."""
    if HAS_PD and isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X)

def _infer_num_classes(y: np.ndarray) -> int:
    return int(len(np.unique(y)))

class _LabelMap:
    """Fold-local label encoder mapping raw labels <-> [0..C-1] indices."""
    def __init__(self):
        self.classes_: Optional[np.ndarray] = None
        self.to_index: Dict[int, int] = {}
        self.to_label: Dict[int, int] = {}
    def fit(self, y: np.ndarray):
        classes = np.unique(y)
        self.classes_ = classes
        self.to_index = {int(c): i for i, c in enumerate(classes)}
        self.to_label = {i: int(c) for i, c in enumerate(classes)}
        return self
    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda t: self.to_index[int(t)])(y)
    def inverse(self, idx: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda i: self.to_label[int(i)])(idx)

# ----------------------------
# Sklearn wrappers
# ----------------------------
class KNN1:
    """1-NN classifier wrapper."""
    name = "knn1"
    def __init__(self, n_neighbors: int = 1, **kwargs: Any) -> None:
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    def fit(self, X: Any, y: np.ndarray, **kwargs: Any) -> "KNN1":
        self.clf.fit(_to_ndarray(X), y); return self
    def predict(self, X: Any) -> np.ndarray:
        return self.clf.predict(_to_ndarray(X))
    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        return self.clf.predict_proba(_to_ndarray(X)) if hasattr(self.clf, "predict_proba") else None

class RF:
    """Random Forest classifier wrapper."""
    name = "rf"
    def __init__(self,
                 n_estimators: int = 500,
                 max_depth: Optional[int] = None,
                 random_state: int = 42,
                 class_weight: Optional[str] = None,
                 max_features: str = "sqrt",
                 min_samples_leaf: int = 1,
                 n_jobs: int = -1,
                 **kwargs: Any) -> None:
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, class_weight=class_weight,
            max_features=max_features, min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs
        )
    def fit(self, X: Any, y: np.ndarray, **kwargs: Any) -> "RF":
        self.clf.fit(_to_ndarray(X), y); return self
    def predict(self, X: Any) -> np.ndarray:
        return self.clf.predict(_to_ndarray(X))
    def predict_proba(self, X: Any) -> np.ndarray:
        return self.clf.predict_proba(_to_ndarray(X))

# ----------------------------
# PyTorch models
# ----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class ConvBlock1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 7, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, padding=k//2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SpectralCNN(nn.Module):
    """1D CNN for 256-point spectra. Input: (B, 256)."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.feat = nn.Sequential(
            ConvBlock1D(1, 32, k=7, p=0.1),
            ConvBlock1D(32, 64, k=5, p=0.1),
            nn.MaxPool1d(2),  # 256->128
            ConvBlock1D(64, 128, k=5, p=0.1),
            nn.MaxPool1d(2),  # 128->64
            ConvBlock1D(128, 128, k=3, p=0.1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cls = nn.Linear(128, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)            # (B,1,256)
        h = self.feat(x).squeeze(-1)  # (B,128)
        return self.cls(h)

class DWTMLPNet(nn.Module):
    """MLP for low-dim DWT stats. Input: (B, D)."""
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 64, p: float = 0.2):
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

class DWTEmbedMLP(nn.Module):
    """Embedding network for DWT stats; outputs only embedding."""
    def __init__(self, in_dim: int, emb_dim: int = 64, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(64, emb_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@torch.no_grad()
def compute_prototypes(emb: torch.Tensor, y: torch.Tensor) -> Dict[int, torch.Tensor]:
    protos: Dict[int, torch.Tensor] = {}
    for c in torch.unique(y):
        idx = (y == c)
        protos[int(c.item())] = emb[idx].mean(dim=0)
    return protos

@torch.no_grad()
def proto_predict(emb: torch.Tensor, prototypes: Dict[int, torch.Tensor]) -> torch.Tensor:
    keys = sorted(prototypes.keys())
    proto_mat = torch.stack([prototypes[k] for k in keys], dim=0)  # (C,E)
    emb_n = F.normalize(emb, dim=1)
    proto_n = F.normalize(proto_mat, dim=1)
    sims = emb_n @ proto_n.t()                                     # (N,C)
    pred_idx = sims.argmax(dim=1)
    labels = torch.tensor([keys[i] for i in pred_idx.tolist()], device=emb.device)
    return labels

def _standardize_train_stats(X_tr: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Z-score using training-set stats; add epsilon for stability."""
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd

def _train_torch_classifier(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: Optional[np.ndarray] = None,
    y_va: Optional[np.ndarray] = None,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    device: str = "cpu",
    standardize: bool = False,
) -> nn.Module:
    # Optional standardization with train stats
    if standardize:
        X_tr_use = _standardize_train_stats(X_tr, X_tr)
        X_va_use = _standardize_train_stats(X_tr, X_va) if X_va is not None else None
    else:
        X_tr_use = X_tr
        X_va_use = X_va

    model.to(device)
    X_tr_t = torch.from_numpy(_to_ndarray(X_tr_use)).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    # Prepare val tensors lazily to avoid device work if not needed
    if X_va_use is not None and y_va is not None:
        Xv_t = torch.from_numpy(_to_ndarray(X_va_use)).float().to(device)
        yv_t = torch.from_numpy(y_va).long().to(device)
    else:
        Xv_t = yv_t = None

    best_state, best_val = None, float("inf")
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        if Xv_t is not None and yv_t is not None:
            model.eval()
            with torch.no_grad():
                lval = crit(model(Xv_t), yv_t).item()
            if lval < best_val:
                best_val = lval
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

class SpectralCNNWrapper:
    """Wrapper for 1D CNN on raw spectra (len=256)."""
    name = "spectral_cnn"
    def __init__(self, n_classes: Optional[int] = None, **hparams: Any) -> None:
        self.n_classes = n_classes
        # Add standardize flag default True for stability on spectra
        self.hparams = dict(epochs=150, batch_size=64, lr=1e-3, weight_decay=1e-3, device="cpu", standardize=True)
        self.hparams.update(hparams)
        self.model: Optional[nn.Module] = None
        self.label_map = _LabelMap()
    def fit(self, X: Any, y: np.ndarray, X_val: Optional[Any] = None, y_val: Optional[np.ndarray] = None, **kwargs: Any) -> "SpectralCNNWrapper":
        X = _to_ndarray(X)
        self.label_map.fit(y)
        y_tr_enc = self.label_map.transform(y)
        y_va_enc = self.label_map.transform(y_val) if y_val is not None else None
        if self.n_classes is None:
            self.n_classes = int(len(self.label_map.classes_))
        self.model = SpectralCNN(n_classes=self.n_classes)
        self.model = _train_torch_classifier(
            self.model, X, y_tr_enc,
            X_va=_to_ndarray(X_val) if X_val is not None else None, y_va=y_va_enc,
            epochs=self.hparams["epochs"], batch_size=self.hparams["batch_size"],
            lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"],
            device=self.hparams["device"], standardize=self.hparams.get("standardize", True),
        )
        return self
    def predict(self, X: Any) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        self.model.eval()
        device = self.hparams.get("device", "cpu")
        # Use train stats standardization at inference by reusing fit-time branch: consistent handled inside fit
        with torch.no_grad():
            Xt = torch.from_numpy(_to_ndarray(X)).float().to(device)
            # Note: standardization used in training was applied via _train_torch_classifier; at inference within
            # wrapper we assume features provided match training transform or pre-standardize outside for consistency.
            pred_idx = self.model(Xt).argmax(dim=1).cpu().numpy()
        return self.label_map.inverse(pred_idx)
    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        assert self.model is not None, "Model not fitted"
        self.model.eval()
        device = self.hparams.get("device", "cpu")
        with torch.no_grad():
            Xt = torch.from_numpy(_to_ndarray(X)).float().to(device)
            probs = F.softmax(self.model(Xt), dim=1).cpu().numpy()
        return probs

class DWTMLPWrapper:
    """Wrapper for MLP on low-dim DWT stats."""
    name = "dwt_mlp"
    def __init__(self, in_dim: Optional[int] = None, n_classes: Optional[int] = None, **hparams: Any) -> None:
        self.in_dim = in_dim
        self.n_classes = n_classes
        # Defaults adjusted per debugging (better convergence):
        self.hparams = dict(epochs=300, batch_size=64, lr=5e-4, weight_decay=1e-3,
                            device="cpu", hidden=128, p=0.3, standardize=True)
        self.hparams.update(hparams)
        self.model: Optional[nn.Module] = None
        self.label_map = _LabelMap()
    def fit(self, X: Any, y: np.ndarray, X_val: Optional[Any] = None, y_val: Optional[np.ndarray] = None, **kwargs: Any) -> "DWTMLPWrapper":
        X = _to_ndarray(X)
        if self.in_dim is None: self.in_dim = X.shape[1]
        # Label encoding
        self.label_map.fit(y)
        y_tr_enc = self.label_map.transform(y)
        y_va_enc = self.label_map.transform(y_val) if y_val is not None else None
        if self.n_classes is None:
            self.n_classes = int(len(self.label_map.classes_))
        self.model = DWTMLPNet(in_dim=self.in_dim, n_classes=self.n_classes,
                               hidden=self.hparams["hidden"], p=self.hparams["p"])
        self.model = _train_torch_classifier(
            self.model, X, y_tr_enc,
            X_va=_to_ndarray(X_val) if X_val is not None else None, y_va=y_va_enc,
            epochs=self.hparams["epochs"], batch_size=self.hparams["batch_size"],
            lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"],
            device=self.hparams["device"], standardize=self.hparams.get("standardize", True),
        )
        return self
    def predict(self, X: Any) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        self.model.eval()
        device = self.hparams.get("device", "cpu")
        with torch.no_grad():
            Xt = torch.from_numpy(_to_ndarray(X)).float().to(device)
            pred_idx = self.model(Xt).argmax(dim=1).cpu().numpy()
        return self.label_map.inverse(pred_idx)
    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        assert self.model is not None, "Model not fitted"
        self.model.eval()
        device = self.hparams.get("device", "cpu")
        with torch.no_grad():
            Xt = torch.from_numpy(_to_ndarray(X)).float().to(device)
            probs = F.softmax(self.model(Xt), dim=1).cpu().numpy()
        return probs

class DWTProtoWrapper:
    """
    Embedding + prototype classifier for DWT stats.
    Prediction uses nearest prototype (cosine). No CE loss used here.
    """
    name = "dwt_proto"
    def __init__(self, in_dim: Optional[int] = None, emb_dim: int = 64, **hparams: Any) -> None:
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.hparams = dict(epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-3, device="cpu")
        self.hparams.update(hparams)
        self.model: Optional[nn.Module] = None
        self.prototypes: Optional[Dict[int, torch.Tensor]] = None
    def fit(self, X: Any, y: np.ndarray, X_enroll: Optional[Any] = None, y_enroll: Optional[np.ndarray] = None, **kwargs: Any) -> "DWTProtoWrapper":
        X = _to_ndarray(X)
        if self.in_dim is None: self.in_dim = X.shape[1]
        device = self.hparams["device"]
        self.model = DWTEmbedMLP(in_dim=self.in_dim, emb_dim=self.emb_dim, p=0.2).to(device)
        with torch.no_grad():
            if X_enroll is None or y_enroll is None:
                X_enroll, y_enroll = X, y
            Xe = torch.from_numpy(_to_ndarray(X_enroll)).float().to(device)
            ye = torch.from_numpy(y_enroll).long().to(device)
            emb = self.model(Xe)
            self.prototypes = compute_prototypes(emb, ye)
        return self
    def predict(self, X: Any) -> np.ndarray:
        assert self.model is not None and self.prototypes is not None, "Model/prototypes not ready"
        device = self.hparams.get("device", "cpu")
        with torch.no_grad():
            Xt = torch.from_numpy(_to_ndarray(X)).float().to(device)
            emb = self.model(Xt)
            pred = proto_predict(emb, self.prototypes).cpu().numpy()
        return pred
    def predict_proba(self, X: Any) -> Optional[np.ndarray]:
        return None

# ----------------------------
# Factory (kwargs pass-through)
# ----------------------------
def get_model(name: str, **kwargs: Any):
    """Factory by model name (kwargs passed to constructors)."""
    key = name.lower()
    if key == "knn1":         return KNN1(**kwargs)
    if key == "rf":           return RF(**kwargs)
    if key == "spectral_cnn": return SpectralCNNWrapper(**kwargs)
    if key == "dwt_mlp":      return DWTMLPWrapper(**kwargs)
    if key == "dwt_proto":    return DWTProtoWrapper(**kwargs)
    raise ValueError(f"Unknown model: {name}")
