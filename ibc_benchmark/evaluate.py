# ibc_benchmark/evaluate.py
"""Evaluation utilities for computing and saving model performance metrics.

This module provides functions to calculate standard classification metrics
and save the results of a benchmark run to a structured file.
"""

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """Computes a dictionary of classification performance metrics.

    Args:
        y_true (np.ndarray): Ground truth labels (integer-encoded).
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray): Predicted probabilities for each class.
        num_classes (int): The total number of unique classes.

    Returns:
        A dictionary containing key performance metrics like 'accuracy'
        and 'roc_auc_ovr' (One-vs-Rest ROC AUC).
    """
    print("Calculating performance metrics...")
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Binarize labels for One-vs-Rest ROC AUC calculation
    y_true_binarized = label_binarize(y_true, classes=range(num_classes))
    
    # Ensure y_prob has the correct shape for multi-class AUC
    if y_prob.shape[1] != num_classes:
        # Handle cases where a class might be missing from a batch
        # This is a safeguard, less likely with a full test set
        roc_auc = float('nan') 
    else:
        roc_auc = roc_auc_score(y_true_binarized, y_prob, multi_class='ovr')

    metrics = {
        'accuracy': accuracy,
        'roc_auc_ovr': roc_auc,
    }
    
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
    
    return metrics


def save_results(
    results: Dict[str, Any],
    output_dir: str = "results"
):
    """Saves the experiment results to a JSON file.

    The results are saved in a structured format, including the experiment
    configuration (model and features) and the resulting metrics. The
    filename is constructed from the model and feature names.

    Args:
        results (Dict[str, Any]): A dictionary containing both the experiment
            configuration and the performance metrics.
        output_dir (str): The directory where results will be saved.
    """
    model_name = results['config']['model']
    feature_name = results['config']['feature']
    
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Construct a descriptive filename
    filename = f"{model_name}_on_{feature_name}_features.json"
    output_path = Path(output_dir) / filename
    
    print(f"Saving results to '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print("Results saved successfully.")

