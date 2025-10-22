# Reproducible Benchmark of Wavelet-Enhanced Intrabody Communication Biometric Identification

This repository contains the source code, data processing scripts, model training, and evaluation notebooks for the reproducible benchmark study described in our paper:

> Jin, S. & Komarov, M. M. (2025). *Reproducible Benchmark of Wavelet-Enhanced Intrabody Communication Biometric Identification*. 

Our work establishes the first public, leakage-free benchmark for IBC biometrics to address methodological flaws in prior studies and provide a reliable baseline for future research.

## Key Contributions
- **Leakage-Free Protocol**: We implement strict subject-wise data splits to prevent data leakage and provide a realistic evaluation of model performance.
- **Comprehensive Evaluation**: We systematically compare multiple feature sets (raw spectra, DWT statistics, and their combination) across a range of classifiers, from classical methods to lightweight deep learning models.
- **Reproducibility**: All code, data, and experimental pipelines are made publicly available to ensure full reproducibility.
- **Embedded Feasibility**: We profile the feature extraction process on a Cortex-M4 class MCU to demonstrate the practical viability of our approach for wearable devices.

## Summary of Results

Our key finding is that a lightweight Multilayer Perceptron (MLP) trained on a combination of raw spectra and Discrete Wavelet Transform (DWT) features achieves the highest accuracy. This demonstrates the power of feature fusion for this task.

| Model        | Feature Set | Validation Accuracy (%) |
|--------------|-------------|-------------------------|
| **MLP**      | **Combined**| **~83%**                |
| SpectralCNN  | Raw         | ~74%                    |
| Random Forest| Combined    | ~49%                    |
| SVD (Ridge)  | Raw         | ~42%                    |
| KNN (1-NN)   | Raw         | ~41%                    |

## Repository Structure

- `data/`: Scripts to download and preprocess the public IBC dataset from Zenodo.
- `features/`: Implementations of feature extraction methods (raw, DWT, combined).
- `models/`: Implementations and training scripts for classifiers (KNN, RF, MLP, SpectralCNN, SVD).
- `evaluation/`: Scripts and notebooks for performance evaluation and figure generation.
- `embedded/`: Simulation code for MCU energy profiling.
- `notebooks/`: Jupyter notebooks for end-to-end analysis and visualization.
- `tests/`: Unit tests for ensuring reproducibility and code correctness.

## Setup and Usage

### Prerequisites
- Python 3.8+
- PyTorch
- scikit-learn, PyWavelets, NumPy, Pandas, Matplotlib
- (For embedded simulation) Keil MDK & CMSIS-DSP

### Installation
1.  **Clone the repository:**
    ```
    git clone https://github.com/dryjins/ibc-wavelet-benchmark.git
    cd ibc-wavelet-benchmark
    ```
2.  **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

### Running the Benchmark
The entire benchmark can be reproduced by running the main execution script. You can specify the desired feature set and model.

1.  **Run a single experiment (e.g., MLP on combined features):**
    ```
    python run_benchmark.py --feature combined --model mlp
    ```
2.  **Run all experiments:**
    ```
    python run_benchmark.py --feature all --model all
    ```
    *(This will iterate through all feature-model combinations described in the paper and save the results.)*

## Data and Code Availability

- **Dataset**: The public IBC dataset is downloaded automatically from Zenodo (DOI: [10.5281/zenodo.8214497](https://zenodo.org/records/8214497)).
- **Source Code**: All code for this study is available in this repository.

## Citation

If you use this code or benchmark in your research, please cite our paper:
```
@article{jin2025ibc,
  title={Reproducible Benchmark of Wavelet-Enhanced Intrabody Communication Biometric Identification},
  author={Jin, Seungmin and Komarov, Mikhail M.},
  journal={},
  year={2025}
}
```
## License
This project is licensed under the MIT License.
