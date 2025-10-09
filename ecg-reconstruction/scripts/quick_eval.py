#!/usr/bin/env python3
"""
Quick evaluation of synthetic data baseline
"""

import sys
import os
sys.path.append('.')

import numpy as np
import torch
from data.generate_test_data import generate_synthetic_ecg
from src.evaluation import calculate_metrics

# Generate synthetic data
signals = generate_synthetic_ecg(n_samples=100, seq_len=5000, noise_level=0.05)
inputs = signals[:, [0, 1, 9], :]  # I, II, V4
targets = signals

# For baseline, use perfect reconstruction (no model yet)
predictions = targets.copy()

# Calculate metrics for chest leads (V1-V6)
chest_lead_indices = [6, 7, 8, 9, 10, 11]  # V1-V6 indices
metrics = calculate_metrics(targets[:, chest_lead_indices, :], predictions[:, chest_lead_indices, :])

lead_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
print('MAE values:', [f'{metrics["mae"][i]:.4f}' for i in range(6)])
print('Correlation values:', [f'{metrics["correlation"][i]:.4f}' for i in range(6)])
print('SNR values:', [f'{metrics["snr"][i]:.1f}' for i in range(6)])