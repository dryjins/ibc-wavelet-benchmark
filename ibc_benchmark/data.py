# ibc_benchmark/data.py
"""Data loading, preprocessing, and splitting utilities for the IBC benchmark."""

import os
import zipfile
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from scipy import interpolate
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# --- Constants ---
DATASET_URL = "https://zenodo.org/record/8214497/files/all_measurements.zip"
RAW_FILENAME = "all_measurements.csv"
CACHE_DIR = "data_cache"
PROCESSED_FILENAME = os.path.join(CACHE_DIR, "processed_spectra.pkl")

# Preprocessing parameters from the paper
N_POINTS = 256
FREQ_MIN = 50e3
FREQ_MAX = 20e6
EXPERIMENT_ID = 1


def download_data(url: str = DATASET_URL, dest_folder: str = CACHE_DIR) -> str:
    """Downloads and extracts the dataset from Zenodo if not already cached.

    Args:
        url (str): The URL of the dataset zip file.
        dest_folder (str): The directory to cache the downloaded data.

    Returns:
        str: The path to the raw CSV file.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    raw_file_path = os.path.join(dest_folder, RAW_FILENAME)
    if os.path.exists(raw_file_path):
        print(f"Dataset found in cache: '{raw_file_path}'. Skipping download.")
        return raw_file_path

    zip_path = raw_file_path + ".zip"
    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(RAW_FILENAME, path=dest_folder)
        os.remove(zip_path)
        print(f"Dataset extracted to '{raw_file_path}'.")
        return raw_file_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return None
    except zipfile.BadZipFile:
        print(f"Error: Downloaded file '{zip_path}' is not a valid zip file.")
        return None


def preprocess_spectra(
    file_path: str,
    n_points: int = N_POINTS,
    freq_min: float = FREQ_MIN,
    freq_max: float = FREQ_MAX,
    experiment_id: int = EXPERIMENT_ID
) -> pd.DataFrame:
    """Loads, filters, cleans, and interpolates IBC spectra.

    This function implements the full preprocessing pipeline described in the paper,
    including filtering, outlier removal, and linear interpolation.

    Args:
        file_path (str): Path to the raw CSV file from the dataset.
        n_points (int): The number of points to interpolate each spectrum to.
        freq_min (float): The minimum frequency for interpolation.
        freq_max (float): The maximum frequency for interpolation.
        experiment_id (int): The experiment ID to filter by.

    Returns:
        pd.DataFrame: A DataFrame containing the processed spectra and metadata.
    """
    print(f"Loading raw data from '{file_path}'...")
    df = pd.read_csv(file_path)

    # 1. Filter for the specified experiment
    df = df[df['experiment_id'] == experiment_id].copy()
    print(f"Filtered for experiment_id={experiment_id}, {len(df):,} records remaining.")

    # 2. Extract frequency points and spectra
    freq_cols = [col for col in df.columns if col.startswith('ch1_magnitude_')]
    freq_values = np.array([float(col.split('_')[-1]) for col in freq_cols])
    spectra = df[freq_cols].values

    # 3. Remove rows with any NaN values
    nan_mask = ~np.isnan(spectra).any(axis=1)
    df_clean = df[nan_mask].copy()
    spectra = spectra[nan_mask]
    print(f"Removed {np.sum(~nan_mask)} rows with NaNs, {len(df_clean):,} records remaining.")

    # 4. Remove outliers based on 3-sigma distance from the mean spectrum
    mean_spectrum_val = np.mean(spectra, axis=0)
    dist_from_mean = np.linalg.norm(spectra - mean_spectrum_val, axis=1)
    outlier_mask = dist_from_mean < (np.mean(dist_from_mean) + 3 * np.std(dist_from_mean))
    
    df_clean = df_clean[outlier_mask].reset_index(drop=True)
    spectra = spectra[outlier_mask]
    print(f"Removed {np.sum(~outlier_mask)} outliers, {len(df_clean):,} records remaining.")
    
    # 5. Linearly interpolate each spectrum to a fixed number of points
    print(f"Interpolating {len(spectra):,} spectra to {n_points} points...")
    new_freqs = np.linspace(freq_min, freq_max, n_points)
    interpolated_spectra = np.zeros((len(spectra), n_points))
    
    for i, spectrum in enumerate(tqdm(spectra, desc="Interpolating", unit="spectra")):
        interp_func = interpolate.interp1d(freq_values, spectrum, kind='linear', fill_value="extrapolate")
        interpolated_spectra[i] = interp_func(new_freqs)
        
    # 6. Create the final processed DataFrame
    processed_df = df_clean[['subject_id', 'session_id']].copy()
    spectra_df = pd.DataFrame(
        interpolated_spectra,
        columns=[f'freq_{i}' for i in range(n_points)]
    )
    
    return pd.concat([processed_df, spectra_df], axis=1)


def get_subject_wise_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs a subject-wise (group-based) split of the data.

    This ensures that all samples from a given subject belong to either the
    training set or the test set, preventing data leakage.

    Args:
        df (pd.DataFrame): The dataframe containing a 'subject_id' column.
        test_size (float): The proportion of subjects to allocate to the test set.
        random_state (int): Seed for reproducibility of the split.

    Returns:
        A tuple containing the training indices and test indices.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df['subject_id']
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    
    print(f"Data split into {len(train_idx)} training samples and {len(test_idx)} test samples.")
    print(f"Unique subjects in train: {df.loc[train_idx, 'subject_id'].nunique()}")
    print(f"Unique subjects in test: {df.loc[test_idx, 'subject_id'].nunique()}")
    
    return train_idx, test_idx


def get_labels(df: pd.DataFrame) -> np.ndarray:
    """Encodes subject_id strings into integer labels.

    Args:
        df (pd.DataFrame): DataFrame with a 'subject_id' column.

    Returns:
        np.ndarray: An array of integer-encoded labels.
    """
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(df['subject_id'])

