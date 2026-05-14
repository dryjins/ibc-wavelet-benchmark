# Reproducible Benchmark of Wavelet-Enhanced Intrabody Communication Biometric Identification

This repository contains code, processed data, notebooks, and experiment artifacts for the study:

> Jin, S. and Komarov, M. M. (2025). *Reproducible Benchmark of Wavelet-Enhanced Intrabody Communication Biometric Identification*.

The repository focuses on the reproducible benchmark pipeline. The manuscript source is maintained separately in Overleaf and is intentionally excluded from this GitHub repository.

## Key Contributions

- **Leakage-free protocol:** strict subject-wise splits prevent subject leakage between training and evaluation.
- **Feature comparison:** raw spectra, DWT statistics, lifting wavelets, scattering features, and fused representations are compared.
- **Model comparison:** classical models, template/linear baselines, and closed-set neural upper-bound analyses are separated.
- **Embedded feasibility:** wavelet feature extraction and classifier inference are profiled for low-power wearable constraints.

## Evaluation Positioning

The primary benchmark is the subject-wise leakage-free evaluation with held-out subjects and enrollment samples. Neural analyses such as MLP and SpectralCNN are closed-set exploratory upper-bound experiments, not the primary subject-wise benchmark.

The latest 5-seed Raw-vs-Combined MLP check reports:

| Setting | Raw MLP | Combined MLP | Combined - Raw |
|---|---:|---:|---:|
| Closed-set, 5 seeds | 83.73% ± 3.95% | 81.24% ± 4.58% | -2.49 percentage points |

This supports the revised interpretation that the high closed-set neural result is driven primarily by model capacity and evaluation setting, not by DWT feature fusion.

## Repository Structure

```text
.
├── ibc_benchmark/        # Package-style implementation from the main benchmark runner
├── run_benchmark.py      # Main command-line benchmark runner
├── analysis_and_visualization.ipynb
├── data/                 # Raw/processed benchmark data and data-preparation utilities
├── experiments/          # Additional neural comparison scripts for reviewer analyses
├── features.py           # Lightweight feature extractors used by notebook/experiment scripts
├── models.py             # Lightweight model wrappers and neural definitions
├── eval_utils.py         # Subject-wise split, enrollment, and metric helpers
├── features/             # Generated feature CSV artifacts
├── models/               # Training/evaluation CLI and saved model artifacts
├── notebooks/            # Named notebooks moved out of the repository root
├── results/              # Reported metrics, logs, and generated figures
└── requirements.txt      # Python dependencies
```

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The public IBC dataset is available from Zenodo: `10.5281/zenodo.8214497`.

The repository includes processed matrices used by the benchmark:

- `data/processed/ibc_processed.csv`: 256-point spectra
- `data/labels_filtered.csv`: subject labels aligned with the processed spectra
- `data/processed/features_simple.csv`: Simple-3 features
- `data/processed/features_dwt_db4_l2.csv`: level-2 db4-DWT statistics

The original archive can be downloaded into `data/raw/` with:

```bash
python data/download_dataset.py
```

## Running Analyses

Run the package-style benchmark runner:

```bash
python run_benchmark.py --feature combined --model mlp
```

Run all feature/model combinations supported by the package runner:

```bash
python run_benchmark.py --feature all --model all
```

Run the closed-set Raw MLP vs Combined MLP reviewer analysis:

```bash
python experiments/mlp_raw_vs_combined_clean.py
```

The script writes:

```text
results/mlp_raw_vs_combined_clean.json
```

Run a classical model using the legacy training CLI:

```bash
python models/train_evaluate.py \
  --features data/processed/features_dwt_db4_l2.csv \
  --labels data/labels_filtered.csv \
  --model rf \
  --output_model models/rf_model.pkl \
  --output_results results/rf_results.json
```

## Notebooks

Exploratory notebooks are kept under `notebooks/` with descriptive names:

- `01_preprocess_and_knn_baseline.ipynb`
- `02_subjectwise_sklearn_benchmarks.ipynb`
- `03_processed_dataset_builder.ipynb`
- `04_closed_set_neural_baselines.ipynb`
- `05_embedded_profiling_and_reports.ipynb`
- `06_synthetic_roc_from_experiments.ipynb`

Legacy failed or superseded notebook work is kept under `notebooks/archive/`. Notebook outputs are stripped to keep diffs readable.

## Reproducibility Notes

- Subject-wise evaluation uses fixed seeds: `42`, `123`, `456`, `789`, and `999` where applicable.
- Closed-set neural analyses use stratified splits and are reported separately from the subject-wise benchmark.
- Jupyter checkpoints, local manuscript exports, raw download archives, and the Overleaf project directory are ignored by Git.

## Data and Code Availability

- Dataset: Zenodo DOI `10.5281/zenodo.8214497`
- Code: this repository
- Manuscript source: maintained separately in Overleaf

## Citation

```bibtex
@article{jin2025ibc,
  title={Reproducible Benchmark of Wavelet-Enhanced Intrabody Communication Biometric Identification},
  author={Jin, Seungmin and Komarov, Mikhail M.},
  year={2025}
}
```

## License

MIT License. See `LICENSE`.
