"""
Download the public IBC dataset (Zenodo DOI: 10.5281/zenodo.8214497)
and unzip it into data/raw/.

Usage
-----
python data/download_dataset.py
"""
import pathlib, requests, zipfile, io

RECORD_ID = 8214497
ZIP_URL   = f"https://zenodo.org/api/records/{RECORD_ID}/files-archive"
DEST_DIR  = pathlib.Path("data/raw")
DEST_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading dataset ZIP…")
resp = requests.get(ZIP_URL, timeout=60)
resp.raise_for_status()

print("Extracting…")
with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
    zf.extractall(DEST_DIR)

print("✔ Done. CSV is in data/raw/all_measurements.csv")