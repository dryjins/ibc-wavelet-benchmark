# ibc_benchmark/features.py
"""Feature extraction utilities for the IBC benchmark.

This module provides functions to extract raw, DWT-based, and combined
feature sets from preprocessed spectral data, as described in the paper.
"""

from typing import Dict, Callable

import numpy as np
import pandas as pd
import pywt
from tqdm import tqdm

# --- Feature Extractor Configuration ---
WAVELET_FAMILY = 'db4'
WAVELET_LEVEL = 2


def get_raw_features(df: pd.DataFrame) -> np.ndarray:
    """Extracts the raw interpolated spectral data as a NumPy array.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame containing frequency
            columns named in the format 'freq_i'.

    Returns:
        np.ndarray: A 2D array of raw spectra (n_samples, n_frequency_points).
    """
    print("Extracting raw features...")
    feature_cols = [col for col in df.columns if col.startswith('freq_')]
    return df[feature_cols].values


def get_dwt_features(df: pd.DataFrame, wavelet: str = WAVELET_FAMILY, level: int = WAVELET_LEVEL) -> np.ndarray:
    """Calculates Discrete Wavelet Transform (DWT) statistical features.

    For each spectrum, this function performs a multilevel DWT and computes
    four statistics (energy, Shannon entropy, mean, std) for each of the
    resulting sub-bands (cA_n, cD_n, ..., cD_1).

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        wavelet (str): The name of the wavelet to use (e.g., 'db4').
        level (int): The decomposition level for the DWT.

    Returns:
        np.ndarray: A 2D array of DWT statistical features. The number of
            columns is (level + 1) * 4.
    """
    print(f"Extracting DWT features (wavelet: {wavelet}, level: {level})...")
    spectra = get_raw_features(df)
    
    all_features = []
    for spectrum in tqdm(spectra, desc="Calculating DWT stats", unit="spectra"):
        # Perform multilevel wavelet decomposition
        coeffs = pywt.wavedec(spectrum, wavelet, level=level, mode='periodization')
        
        band_features = []
        for band_coeffs in coeffs:
            # 1. Energy: Sum of squared coefficients
            energy = np.sum(np.square(band_coeffs))
            
            # 2. Shannon Entropy: -sum(p_i * log2(p_i))
            #    where p_i is the normalized energy of the coefficient
            #    Add a small epsilon to prevent log(0)
            p = np.square(band_coeffs) / (energy + 1e-12)
            entropy = -np.sum(p * np.log2(p + 1e-12))
            
            # 3. Mean of coefficients
            mean = np.mean(band_coeffs)
            
            # 4. Standard deviation of coefficients
            std = np.std(band_coeffs)
            
            band_features.extend([energy, entropy, mean, std])
            
        all_features.append(band_features)
        
    return np.array(all_features)


def get_combined_features(df: pd.DataFrame) -> np.ndarray:
    """Combines raw spectral features and DWT statistical features.

    This function horizontally stacks the raw feature matrix and the DWT
    feature matrix to create a single, richer feature representation.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.

    Returns:
        np.ndarray: A 2D array containing the concatenated features.
    """
    print("Extracting and combining raw and DWT features...")
    raw_feats = get_raw_features(df)
    dwt_feats = get_dwt_features(df)
    
    combined_feats = np.hstack((raw_feats, dwt_feats))
    return combined_feats


# --- Feature Factory ---
FEATURE_EXTRACTORS: Dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    'raw': get_raw_features,
    'dwt': get_dwt_features,
    'combined': get_combined_features,
}


def extract_features(df: pd.DataFrame, method: str) -> np.ndarray:
    """A factory function to dispatch to the correct feature extractor.

    Args:
        df (pd.DataFrame): The preprocessed data.
        method (str): The feature extraction method to use.
            Must be one of 'raw', 'dwt', or 'combined'.

    Returns:
        np.ndarray: The resulting feature matrix.
    
    Raises:
        ValueError: If an unknown feature extraction method is specified.
    """
    if method not in FEATURE_EXTRACTORS:
        raise ValueError(
            f"Unknown feature extraction method: '{method}'. "
            f"Available methods are: {list(FEATURE_EXTRACTORS.keys())}"
        )
    
    extractor = FEATURE_EXTRACTORS[method]
    return extractor(df)
