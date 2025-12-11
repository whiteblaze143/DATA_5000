import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, padding=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Classifier1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=30, features=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, features, kernel=7),
            ConvBlock(features, features * 2, kernel=5, padding=2),
            nn.MaxPool1d(2),
            ConvBlock(features * 2, features * 4, kernel=5, padding=2),
            nn.MaxPool1d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(features * 4, num_classes)

    def forward(self, x):
        # x: [B, C, L]
        h = self.encoder(x)
        h = self.global_pool(h).squeeze(-1)
        logits = self.fc(h)
        return logits
