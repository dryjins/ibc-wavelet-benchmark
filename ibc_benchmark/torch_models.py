# ibc_benchmark/torch_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralCNN(nn.Module):
    """
    A 1D CNN for learning from raw spectral data, as described in the paper.
    Architecture: 3 conv blocks with decreasing kernel sizes (7, 5, 3).
    """
    def __init__(self, in_channels=1, num_classes=30):
        super(SpectralCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape for CNN: (batch_size, channels, features)
        # Raw spectra are (batch_size, num_features), so we add a channel dimension.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.pool(self.conv_block1(x))
        x = self.pool(self.conv_block2(x))
        x = self.conv_block3(x)
        
        x = self.gap(x).squeeze(-1) # (batch_size, 128)
        x = self.fc(x)
        return x

class SimpleMLP(nn.Module):
    """
    A two-hidden-layer MLP for learning from combined (raw + DWT) features.
    """
    def __init__(self, input_dim, num_classes=30):
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

    def forward(self, x):
        return self.layers(x)

