# run_benchmark.py
"""Main entrypoint for running the IBC benchmark experiments.

This script orchestrates the entire pipeline from data download to model
evaluation and result saving. It uses command-line arguments to allow
flexible selection of feature sets and models to run.

Usage:
    # Run a single experiment (e.g., MLP on combined features)
    python run_benchmark.py --feature combined --model mlp

    # Run all defined experiments
    python run_benchmark.py --feature all --model all
"""

import argparse
import os
import time
from typing import List

import numpy as np
import pandas as pd
import torch

# Import all necessary modules from our benchmark package
from ibc_benchmark import data, features, models, trainer, evaluate


def run_experiment(
    feature_method: str,
    model_name: str,
    df: pd.DataFrame,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    num_classes: int,
    config: Dict
):
    """Runs a single experiment for a given feature-model combination.

    Args:
        feature_method (str): The feature extraction method to use.
        model_name (str): The model to train.
        df (pd.DataFrame): The full preprocessed dataframe.
        labels (np.ndarray): The full array of encoded labels.
        train_idx (np.ndarray): Indices for the training set.
        test_idx (np.ndarray): Indices for the test set.
        num_classes (int): Total number of unique classes.
        config (Dict): A dictionary holding experiment configurations (e.g., epochs).
    """
    print(f"\n===== Running Experiment: Model='{model_name}' on Features='{feature_method}' =====")
    
    # 1. Extract features for train and test sets
    X_train = features.extract_features(df.iloc[train_idx], feature_method)
    X_test = features.extract_features(df.iloc[test_idx], feature_method)
    y_train, y_test = labels[train_idx], labels[test_idx]

    # 2. Train and evaluate the model
    if model_name in ['knn', 'rf', 'svd']:
        # scikit-learn model training
        model = models.get_sklearn_model(model_name)
        trained_model, metrics = trainer.train_sklearn_model(model, X_train, y_train, X_test, y_test)
        
        # Get probabilities for ROC AUC if possible
        if hasattr(trained_model, "predict_proba"):
            y_prob = trained_model.predict_proba(X_test)
            metrics.update(evaluate.compute_metrics(y_test, trained_model.predict(X_test), y_prob, num_classes))
        
    elif model_name in ['mlp', 'cnn']:
        # PyTorch model training
        input_dim = X_train.shape[1] if model_name == 'mlp' else None
        model = models.get_pytorch_model(model_name, input_dim, num_classes)
        
        trained_model, metrics = trainer.train_pytorch_model(
            model, X_train, y_train, X_test, y_test,
            epochs=config['epochs'], batch_size=config['batch_size'],
            lr=config['lr'], weight_decay=config['weight_decay']
        )
        
        # Get predictions and probabilities for evaluation
        trained_model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(X_test).to(next(trained_model.parameters()).device)
            outputs = trained_model(inputs)
            y_prob = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            y_pred = np.argmax(y_prob, axis=1)
        
        metrics = evaluate.compute_metrics(y_test, y_pred, y_prob, num_classes)

    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    # 3. Save results
    result_package = {
        'config': {
            'model': model_name,
            'feature': feature_method,
            **config
        },
        'metrics': metrics
    }
    evaluate.save_results(result_package)


def main():
    """Main function to parse arguments and orchestrate the benchmark runs."""
    parser = argparse.ArgumentParser(description="Run IBC Benchmark Experiments")
    parser.add_argument('--feature', type=str, default='all', choices=['raw', 'dwt', 'combined', 'all'],
                        help='Feature set to use for the experiment.')
    parser.add_argument('--model', type=str, default='all', choices=['knn', 'rf', 'svd', 'mlp', 'cnn', 'all'],
                        help='Model to train and evaluate.')
    # Add PyTorch training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for PyTorch models.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for PyTorch models.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for PyTorch models.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for PyTorch models.')
    
    args = parser.parse_args()

    start_time = time.time()

    # --- Data Preparation ---
    raw_path = data.download_data()
    df_processed = data.preprocess_spectra(raw_path)
    labels = data.get_labels(df_processed)
    num_classes = len(np.unique(labels))
    train_idx, test_idx = data.get_subject_wise_split(df_processed)
    
    # --- Experiment Execution ---
    feature_methods = ['raw', 'dwt', 'combined'] if args.feature == 'all' else [args.feature]
    model_names = ['knn', 'rf', 'svd', 'mlp', 'cnn'] if args.model == 'all' else [args.model]
    
    train_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }

    for feature_method in feature_methods:
        for model_name in model_names:
            # Skip invalid combinations as per the paper
            if model_name == 'cnn' and feature_method != 'raw':
                print(f"\n===== Skipping: CNN model only supports 'raw' features, not '{feature_method}'. =====")
                continue
            
            run_experiment(
                feature_method=feature_method,
                model_name=model_name,
                df=df_processed,
                labels=labels,
                train_idx=train_idx,
                test_idx=test_idx,
                num_classes=num_classes,
                config=train_config
            )

    end_time = time.time()
    print(f"\nBenchmark finished. Total execution time: {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    # To make `ibc_benchmark` importable, we need an __init__.py file.
    # We will create it if it doesn't exist.
    init_path = os.path.join('ibc_benchmark', '__init__.py')
    if not os.path.exists('ibc_benchmark'):
        os.makedirs('ibc_benchmark')
    if not os.path.exists(init_path):
        with open(init_path, 'w') as f:
            pass  # Create an empty __init__.py
    
    main()
