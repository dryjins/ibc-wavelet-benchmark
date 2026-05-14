# features.py
from __future__ import annotations
import numpy as np
import pywt

class SimpleFeature:
    """First-principles low-dim features: specific points, band means/std."""
    name = "simple"
    def __init__(self, idx_points=None, band_slices=None):
        self.idx_points = idx_points or [200]           # e.g., ~12.5 MHz on 256 grid
        self.band_slices = band_slices or [slice(64,96)]
    def fit(self, X: np.ndarray, y=None): return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        out = []
        for row in X:
            f = []
            for i in self.idx_points:
                f.append(float(row[i]))
            for slc in self.band_slices:
                seg = row[slc]; f += [float(seg.mean()), float(seg.std())]
            out.append(f)
        return np.asarray(out, dtype=float)

class DWTdb4L2:
    """Daubechies-4 level-2; stats per band (A2,D2,D1)."""
    name = "dwt_db4_l2"
    def __init__(self, wavelet="db4", level=2):
        self.wavelet, self.level = wavelet, level

    @staticmethod
    def _stats(c: np.ndarray) -> list[float]:
        e = float(np.sum(c**2))
        p = (c**2)/e if e>0 else np.zeros_like(c)
        ent = float(-np.sum(p*np.log2(p+1e-12)))
        return [e, ent, float(c.mean()), float(c.std())]

    def fit(self, X: np.ndarray, y=None): return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for row in X:
            coeffs = pywt.wavedec(row, self.wavelet, level=self.level, mode="periodization")
            feats.append([s for c in coeffs for s in self._stats(c)])
        return np.asarray(feats, dtype=float)


class DWTdb4L6:
    """Daubechies-4 level-6; stats per band (A6,D6,D5,D4,D3,D2,D1). Matches paper: 7 sub-bands × 4 stats = 28 features."""
    name = "dwt_db4_l6"
    def __init__(self, wavelet="db4", level=6):
        self.wavelet, self.level = wavelet, level

    @staticmethod
    def _stats(c: np.ndarray) -> list[float]:
        e = float(np.sum(c**2))
        p = (c**2)/e if e>0 else np.zeros_like(c)
        ent = float(-np.sum(p*np.log2(p+1e-12)))
        return [e, ent, float(c.mean()), float(c.std())]

    def fit(self, X: np.ndarray, y=None): return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for row in X:
            coeffs = pywt.wavedec(row, self.wavelet, level=self.level, mode="periodization")
            feats.append([s for c in coeffs for s in self._stats(c)])
        return np.asarray(feats, dtype=float)

class CombinedSimpleDWT:
    """Early-fusion: SimpleFeature + DWT(db4, level=2) stats per band."""
    name = "combined_simple_dwt"
    def __init__(self, idx_points=None, band_slices=None, wavelet="db4", level=2):
        self.idx_points = idx_points or [200]        # ~12.5 MHz index on 256 grid (예시)
        self.band_slices = band_slices or [slice(64,96)]  # 예시 대역
        self.wavelet, self.level = wavelet, level

    @staticmethod
    def _dwt_stats(c: np.ndarray) -> list[float]:
        e = float(np.sum(c**2))
        p = (c**2)/e if e>0 else np.zeros_like(c)
        ent = float(-np.sum(p*np.log2(p+1e-12)))
        return [e, ent, float(c.mean()), float(c.std())]

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for row in X:
            # simple block
            f_simple = []
            for i in self.idx_points:
                f_simple.append(float(row[i]))
            for slc in self.band_slices:
                seg = row[slc]
                f_simple.extend([float(seg.mean()), float(seg.std())])

            # dwt block
            coeffs = pywt.wavedec(row, self.wavelet, level=self.level, mode="periodization")
            f_dwt = [s for c in coeffs for s in self._dwt_stats(c)]

            feats.append(f_simple + f_dwt)
        return np.asarray(feats, dtype=float)

def get_feature(name: str):
    if name == "simple": return SimpleFeature()
    if name == "dwt_db4_l2": return DWTdb4L2()
    if name == "dwt_db4_l6": return DWTdb4L6()
    if name == "combined_simple_dwt": return CombinedSimpleDWT()
    raise ValueError(f"Unknown feature: {name}")       
