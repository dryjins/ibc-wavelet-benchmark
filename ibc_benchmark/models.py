# ibc_benchmark/models.py
"""Model definitions for scikit-learn classifiers and PyTorch neural networks.

This module provides factory functions to instantiate all models used in the
benchmark, including classical machine learning models and the custom
PyTorch architectures (MLP and SpectralCNN) described in the paper.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

# --- PyTorch Model Definitions ---

class SpectralCNN(nn.Module):
    """A 1D CNN for learning from raw spectral data.

    This architecture consists of three convolutional blocks with decreasing
    kernel sizes, each followed by batch normalization, ReLU activation, and
    max pooling. A final global average pooling layer and a fully connected

    layer produce the classification output.

    Args:
        in_channels (int): Number of input channels (typically 1 for spectra).
        num_classes (int): The number of output classes (subjects).
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 30):
        super(SpectralCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the SpectralCNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: The output logits of shape (batch_size, num_classes).
        """
        # Add a channel dimension for 1D convolution
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.pool(self.conv_block1(x))
        x = self.pool(self.conv_block2(x))
        x = self.conv_block3(x)
        
        x = self.gap(x).squeeze(-1)  # (batch_size, 128)
        x = self.fc(x)
        return x


class SimpleMLP(nn.Module):
    """A two-hidden-layer MLP for combined (raw + DWT) features.

    This network uses ReLU activations and dropout for regularization.

    Args:
        input_dim (int): The dimensionality of the input features.
        num_classes (int): The number of output classes (subjects).
    """
    def __init__(self, input_dim: int, num_classes: int = 30):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the MLP."""
        return self.layers(x)


# --- Model Factory Functions ---

def get_sklearn_model(model_name: str, random_state: int = 42) -> Any:
    """Factory function for scikit-learn based models.

    Args:
        model_name (str): The name of the model. Supported: 'knn', 'rf', 'svd'.
        random_state (int): Seed for reproducibility.

    Returns:
        An unfitted scikit-learn classifier or pipeline.
    """
    if model_name == 'knn':
        # Pipeline includes scaling for consistency
        return Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=1))
        ])
    
    elif model_name == 'rf':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1))
        ])
        
    elif model_name == 'svd':
        # SVD is implemented via PCA projection followed by a Ridge classifier
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=64, random_state=random_state)),
            ('ridge', RidgeClassifier(alpha=1e-2))
        ])
        
    else:
        raise ValueError(f"Unknown scikit-learn model: {model_name}")


def get_pytorch_model(model_name: str, input_dim: int, num_classes: int) -> nn.Module:
    """Factory function for PyTorch based models.

    Args:
        model_name (str): The name of the model. Supported: 'mlp', 'cnn'.
        input_dim (int): Input feature dimension for the model.
        num_classes (int): Number of target classes.

    Returns:
        An instantiated PyTorch model.
    """
    if model_name == 'mlp':
        return SimpleMLP(input_dim=input_dim, num_classes=num_classes)
        
    elif model_name == 'cnn':
        # The CNN input dimension is fixed by its internal architecture for spectra
        return SpectralCNN(num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown PyTorch model: {model_name}")
