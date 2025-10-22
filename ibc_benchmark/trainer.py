# ibc_benchmark/trainer.py
"""Model training and evaluation runners for scikit-learn and PyTorch models.

This module provides high-level functions to run the training and evaluation
process for both classical scikit-learn pipelines and PyTorch neural networks,
returning the final trained model and key performance metrics.
"""

from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_sklearn_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Trains a scikit-learn model and evaluates it on the test set.

    Args:
        model (BaseEstimator): The scikit-learn model or pipeline to train.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature data.
        y_test (np.ndarray): Test labels.

    Returns:
        A tuple containing:
        - The trained scikit-learn model.
        - A dictionary with evaluation metrics (e.g., 'accuracy').
    """
    print(f"Training scikit-learn model: {model.__class__.__name__}...")
    model.fit(X_train, y_train)

    print("Evaluating model on the test set...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    metrics = {'accuracy': accuracy}
    return model, metrics


def train_pytorch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Trains a PyTorch model and evaluates it on the test set.

    This function handles the standard PyTorch training loop, including
    data loading, optimization, and final evaluation.

    Args:
        model (nn.Module): The PyTorch model to train.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature data.
        y_test (np.ndarray): Test labels.
        epochs (int): The number of training epochs.
        batch_size (int): The size of batches for training.
        lr (float): The learning rate for the AdamW optimizer.
        weight_decay (float): The weight decay for the AdamW optimizer.

    Returns:
        A tuple containing:
        - The trained PyTorch model.
        - A dictionary with evaluation metrics (e.g., 'accuracy').
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PyTorch model on device: {device}...")
    model.to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Final evaluation on the test set
    print("Evaluating model on the test set...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    metrics = {'accuracy': accuracy}
    return model, metrics
