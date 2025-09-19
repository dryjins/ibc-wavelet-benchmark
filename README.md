# Intrabody Communication Biometric Identification Benchmark

This repository contains the source code, data processing scripts, model training, and evaluation notebooks for the reproducible benchmark study described in the paper "Reproducible Benchmark of Wavelet-Enhanced Intrabody Communication Biometric Identification".

## Repository Structure
- **data/**: Scripts to download and preprocess the public IBC dataset from Zenodo.
- **features/**: Implementations of feature extraction methods (handcrafted, discrete wavelet transform, lifting-scheme biorthogonal wavelets, wavelet scattering).
- **models/**: Training scripts for classifiers (KNN, SVM, Random Forest, LightGBM).
- **evaluation/**: Scripts and notebooks for computing performance metrics (accuracy, FAR, FRR, EER) and generating figures (ROC curves, scalograms).
- **embedded/**: Simulation code for MCU energy profiling on Cortex-M4 (using Keil MDK and CMSIS-DSP).
- **notebooks/**: Jupyter notebooks for end-to-end reproduction of experiments and results.
- **tests/**: Unit tests for key functions to ensure reproducibility.

## Prerequisites
- Python 3.8+
- Libraries: scikit-learn, PyWavelets, Kymatio, LightGBM, NumPy, Pandas, Matplotlib (see `requirements.txt`)

## Setup
1. Clone the repository:
   ```
   git clone https://github.com/dryjins/ibc-wavelet-benchmark.git
   cd ibc-wavelet-benchmark
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset:
   ```
   python data/download_dataset.py
   ```
   (Downloads from Zenodo DOI: 10.5281/zenodo.8214497 and preprocesses spectra.)

## Running the Benchmark
1. Preprocess data:
   ```
   python data/preprocess.py --input data/raw_spectra.csv --output data/processed_spectra.csv
   ```

2. Extract features (e.g., lifting wavelets):
   ```
   python features/extract_features.py --method lift-bior --input data/processed_spectra.csv --output features/lift_bior_features.csv
   ```

3. Train and evaluate a model (e.g., Random Forest):
   ```
   python models/train_evaluate.py --features features/lift_bior_features.csv --model rf --output results/rf_results.json
   ```

4. Generate figures (e.g., ROC curve):
   ```
   jupyter notebook notebooks/generate_figures.ipynb
   ```

5. Run MCU energy profiling simulation:
   ```
   cd embedded
   # Run simulation script (requires Keil MDK installed)
   python simulate_energy.py --model lift-bior
   ```

## Reproducibility Notes
- Fixed random seeds (e.g., 42) are used for splits and training to ensure identical results.
- Run the full pipeline with:
  ```
  python run_benchmark.py --feature all --model all
  ```
- Results match the paper's Table 3 and Figures 2-3.
- For questions, open an issue or contact sedzhin@hse.ru.

## License
MIT License

## Citation
If you use this code, please cite:
```
Jin, S. & Komarov, M. M. Reproducible Benchmark of Wavelet-Enhanced Intrabody Communication Biometric Identification. (2025).
```
