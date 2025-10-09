# ibc_benchmark/features.py

import numpy as np
import pandas as pd
import pywt
from tqdm import tqdm

def get_raw_features(processed_df: pd.DataFrame) -> np.ndarray:
    """
    Extracts the raw spectral data as a NumPy array.

    Args:
        processed_df (pd.DataFrame): The preprocessed data containing frequency columns.

    Returns:
        np.ndarray: A 2D array of raw spectra (n_samples, n_features).
    """
    print("Extracting raw features...")
    feature_cols = [col for col in processed_df.columns if col.startswith('freq_')]
    return processed_df[feature_cols].values

def get_dwt_features(processed_df: pd.DataFrame, wavelet: str = 'db4', level: int = 2) -> np.ndarray:
    """
    Calculates DWT statistical features for each spectrum.

    Args:
        processed_df (pd.DataFrame): The preprocessed data.
        wavelet (str): The name of the wavelet to use (e.g., 'db4').
        level (int): The decomposition level for DWT.

    Returns:
        np.ndarray: A 2D array of DWT statistical features (n_samples, n_dwt_features).
    """
    print(f"Extracting DWT features (wavelet: {wavelet}, level: {level})...")
    spectra = get_raw_features(processed_df)
    
    all_features = []
    for spectrum in tqdm(spectra, desc="Calculating DWT stats", unit="spectra"):
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(spectrum, wavelet, level=level, mode='periodization')
        
        # Calculate statistics for each sub-band (cA_n, cD_n, ..., cD_1)
        band_features = []
        for band_coeffs in coeffs:
            energy = np.sum(np.square(band_coeffs))
            # Use Shannon entropy definition
            p = np.square(band_coeffs) / (energy + 1e-12)
            entropy = -np.sum(p * np.log2(p + 1e-12))
            mean = np.mean(band_coeffs)
            std = np.std(band_coeffs)
            band_features.extend([energy, entropy, mean, std])
            
        all_features.append(band_features)
        
    return np.array(all_features)

def get_combined_features(processed_df: pd.DataFrame) -> np.ndarray:
    """
    Combines raw spectral features and DWT statistical features.

    Args:
        processed_df (pd.DataFrame): The preprocessed data.

    Returns:
        np.ndarray: A 2D array of the concatenated features.
    """
    print("Extracting and combining raw and DWT features...")
    raw_feats = get_raw_features(processed_df)
    dwt_feats = get_dwt_features(processed_df)
    
    # Horizontally stack the two feature sets
    combined_feats = np.hstack((raw_feats, dwt_feats))
    return combined_feats

