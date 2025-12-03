import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    """
    Simple baseline model for ECG lead reconstruction
    Uses a simple CNN with residual connections
    """
    def __init__(self, in_channels=3, out_channels=5, features=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, features, kernel_size=7, padding=3),
            nn.BatchNorm1d(features),
            nn.ReLU(),
            nn.Conv1d(features, features*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(features*2),
            nn.ReLU(),
            nn.Conv1d(features*2, features*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(features*4),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(features*4, features*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(features*2),
            nn.ReLU(),
            nn.Conv1d(features*2, features, kernel_size=5, padding=2),
            nn.BatchNorm1d(features),
            nn.ReLU(),
            nn.Conv1d(features, out_channels, kernel_size=7, padding=3)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x