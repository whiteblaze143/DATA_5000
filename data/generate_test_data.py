#!/usr/bin/env python3
# filepath: data/generate_test_data.py

import os
import numpy as np
import argparse

def generate_synthetic_ecg(n_samples=100, seq_len=5000, noise_level=0.1):
    """
    Generate synthetic ECG signals for testing

    Args:
        n_samples: Number of ECG recordings to generate
        seq_len: Length of each recording (samples)
        noise_level: Amount of noise to add

    Returns:
        signals: Array of shape (n_samples, 12, seq_len)
    """
    rng = np.random.default_rng(42)

    # Create time axis
    t = np.linspace(0, 10, seq_len)  # 10 seconds at 500 Hz

    signals = []

    for _ in range(n_samples):
        # Generate base ECG signal components
        # P wave
        p_wave = 0.1 * np.exp(-((t - 0.1) / 0.05)**2)

        # QRS complex
        qrs = -0.5 * np.exp(-((t - 0.2) / 0.02)**2) + \
              1.0 * np.exp(-((t - 0.22) / 0.015)**2) + \
              -0.3 * np.exp(-((t - 0.24) / 0.02)**2)

        # T wave
        t_wave = 0.2 * np.exp(-((t - 0.4) / 0.08)**2)

        # Combine to make one lead
        base_signal = p_wave + qrs + t_wave

        # Create 12 leads with variations
        leads = []
        for lead_idx in range(12):
            # Add lead-specific variations
            variation = np.sin(2 * np.pi * (lead_idx + 1) * t / 10) * 0.1
            noise = rng.normal(0, noise_level, seq_len)
            lead_signal = base_signal + variation + noise
            leads.append(lead_signal)

        signals.append(np.array(leads))

    return np.array(signals)

def create_test_data(output_path, n_train=50, n_val=20, n_test=30):
    """
    Create test data splits for the ECG reconstruction task
    """
    print("Generating synthetic ECG test data...")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Generate data for each split
    splits = {
        'train': n_train,
        'val': n_val,
        'test': n_test
    }

    min_val = float('inf')
    max_val = float('-inf')

    for split_name, n_samples in splits.items():
        print(f"Generating {split_name} data: {n_samples} samples")

        # Generate signals
        signals = generate_synthetic_ecg(n_samples=n_samples)

        # Update global min/max for normalization
        min_val = min(min_val, np.min(signals))
        max_val = max(max_val, np.max(signals))

        # Normalize to [0, 1] range
        signals_norm = (signals - min_val) / (max_val - min_val)

        # Create input (I, II, V4) and target (all 12 leads)
        input_indices = [0, 1, 9]  # I, II, V4
        target_indices = list(range(12))

        input_data = signals_norm[:, input_indices, :]
        target_data = signals_norm[:, target_indices, :]

        # Save data
        np.save(os.path.join(output_path, f'{split_name}_input.npy'), input_data)
        np.save(os.path.join(output_path, f'{split_name}_target.npy'), target_data)

        print(f"  Input shape: {input_data.shape}")
        print(f"  Target shape: {target_data.shape}")

    # Save normalization parameters
    with open(os.path.join(output_path, 'norm_params.pkl'), 'wb') as f:
        import pickle
        pickle.dump({'min': min_val, 'max': max_val}, f)

    # Create dummy metadata
    import pandas as pd
    rng = np.random.default_rng(42)
    metadata = pd.DataFrame({
        'ecg_id': range(sum(splits.values())),
        'patient_id': rng.integers(1000, 9999, sum(splits.values())),
        'strat_fold': [0] * n_train + [8] * n_val + [9] * n_test
    })
    metadata.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)

    print(f"Test data saved to: {output_path}")
    print("Data generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic test data')
    parser.add_argument('--output', type=str, default='data/test_data',
                        help='Output directory for test data')
    parser.add_argument('--n_train', type=int, default=50,
                        help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=20,
                        help='Number of validation samples')
    parser.add_argument('--n_test', type=int, default=30,
                        help='Number of test samples')

    args = parser.parse_args()
    create_test_data(args.output, args.n_train, args.n_val, args.n_test)