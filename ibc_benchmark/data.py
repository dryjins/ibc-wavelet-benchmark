# ibc_benchmark/data.py

import pandas as pd
import numpy as np
from scipy import interpolate
import requests
import os
from tqdm import tqdm
import zipfile

# --- Constants ---
DATASET_URL = "https://zenodo.org/record/8214497/files/all_measurements.zip"
RAW_FILENAME = "all_measurements.csv"
CACHE_DIR = "data_cache"
PROCESSED_FILENAME = os.path.join(CACHE_DIR, "processed_spectra.pkl")

# Preprocessing parameters
N_POINTS = 256
FREQ_MIN = 50e3
FREQ_MAX = 20e6
EXPERIMENT_ID = 1

def download_data(url: str = DATASET_URL, dest_folder: str = CACHE_DIR):
    """
    Downloads and extracts the dataset from Zenodo if it's not already present.

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
        print(f"Dataset '{RAW_FILENAME}' already exists in '{dest_folder}'. Skipping download.")
        return raw_file_path

    zip_path = raw_file_path + ".zip"
    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, stream=True)
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

def load_and_preprocess_data(file_path: str, n_points: int = N_POINTS) -> pd.DataFrame:
    """
    Loads, filters, cleans, and interpolates the IBC spectra according to the paper's methodology.

    Args:
        file_path (str): Path to the raw CSV file.
        n_points (int): The number of points to interpolate each spectrum to.

    Returns:
        pd.DataFrame: A DataFrame containing the processed spectra and metadata.
    """
    print(f"Loading raw data from '{file_path}'...")
    df = pd.read_csv(file_path)

    # Step 1: Filter for environmental consistency
    df = df[df['experiment_id'] == EXPERIMENT_ID].copy()
    print(f"Filtered for experiment_id = {EXPERIMENT_ID}, {len(df)} records remaining.")

    # Step 2: Extract frequency points and spectra (using ch1 for consistency)
    freq_cols = [col for col in df.columns if col.startswith('ch1_magnitude_')]
    freq_values = np.array([float(col.split('_')[-1]) for col in freq_cols])
    spectra = df[freq_cols].values

    # Step 3: Remove missing values and outliers
    nan_mask = ~np.isnan(spectra).any(axis=1)
    df_clean = df[nan_mask].copy()
    spectra = spectra[nan_mask]
    print(f"Removed {np.sum(~nan_mask)} rows with NaNs. {len(df_clean)} records remaining.")

    mean_spectrum_val = np.mean(spectra, axis=0)
    dist_from_mean = np.linalg.norm(spectra - mean_spectrum_val, axis=1)
    outlier_mask = dist_from_mean < (np.mean(dist_from_mean) + 3 * np.std(dist_from_mean))
    
    df_clean = df_clean[outlier_mask].reset_index(drop=True)
    spectra = spectra[outlier_mask]
    print(f"Removed {np.sum(~outlier_mask)} outliers. {len(df_clean)} records remaining.")
    
    # Step 4: Linearly interpolate each spectrum
    print(f"Interpolating {len(spectra)} spectra to {n_points} points...")
    new_freqs = np.linspace(FREQ_MIN, FREQ_MAX, n_points)
    interpolated_spectra = np.zeros((len(spectra), n_points))
    
    for i, spectrum in enumerate(tqdm(spectra, desc="Interpolating", unit="spectra")):
        interp_func = interpolate.interp1d(freq_values, spectrum, kind='linear', fill_value="extrapolate")
        interpolated_spectra[i] = interp_func(new_freqs)
        
    # Step 5: Create the final DataFrame
    # Note: Z-score normalization is intentionally omitted here.
    processed_df = df_clean[['subject_id', 'session_id']].copy()
    spectra_df = pd.DataFrame(interpolated_spectra, columns=[f'freq_{i}' for i in range(n_points)])
    
    return pd.concat([processed_df, spectra_df], axis=1)

if __name__ == '__main__':
    # This block allows the script to be run directly to set up the data.
    # e.g., `python ibc_benchmark/data.py`
    raw_path = download_data()
    if raw_path:
        processed_df = load_and_preprocess_data(raw_path)
        print(f"\nPreprocessing complete. Saving processed data to '{PROCESSED_FILENAME}'...")
        processed_df.to_pickle(PROCESSED_FILENAME)
        print("Done.")
        print("\n--- Processed Data Summary ---")
        print(processed_df.head())
        print(f"\nShape: {processed_df.shape}")
        print(f"Unique subjects: {processed_df['subject_id'].nunique()}")
