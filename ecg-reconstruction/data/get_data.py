#!/usr/bin/env python3
# filepath: data/get_data.py

import os
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
import argparse
import pickle
import h5py

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare PTB-XL data or load test data')
    parser.add_argument('--ptb_xl_path', type=str, help='Path to PTB-XL dataset')
    parser.add_argument('--output_path', type=str, help='Path to save processed data')
    parser.add_argument('--test_mode', action='store_true', help='Use synthetic test data instead of PTB-XL')
    parser.add_argument('--sampling_rate', type=int, default=500, help='Target sampling rate')
    return parser.parse_args()

def load_ptb_xl(ptb_xl_path, sampling_rate=500):
    """
    Load PTB-XL dataset and metadata
    """
    print(f"Loading PTB-XL metadata from: {ptb_xl_path}")
    # Load metadata
    metadata = pd.read_csv(os.path.join(ptb_xl_path, 'ptbxl_database.csv'), index_col='ecg_id')
    metadata.scp_codes = metadata.scp_codes.apply(lambda x: eval(x))
    
    # Load SCP code descriptions
    scp_codes = pd.read_csv(os.path.join(ptb_xl_path, 'scp_statements.csv'), index_col=0)
    scp_codes = scp_codes[scp_codes.diagnostic == 1]
    
    # Load raw data - only load records that exist
    records = []
    signals = []

    print("Loading ECG signals...")
    # First, get list of available records
    records100_path = os.path.join(ptb_xl_path, 'records100')
    available_records = []
    if os.path.exists(records100_path):
        for file in os.listdir(records100_path):
            if file.endswith('.hea'):
                record_id = int(file.split('_')[0])
                available_records.append(record_id)

    print(f"Found {len(available_records)} available records: {available_records[:10]}...")

    # Filter metadata to only include available records
    metadata_subset = metadata[metadata.index.isin(available_records)]
    print(f"Metadata contains {len(metadata_subset)} matching records")

    for idx, row in tqdm(metadata_subset.iterrows(), total=len(metadata_subset)):
        # For our subset, construct filename directly from record ID
        # Our files are named like "00001_hr.hea" in records100/
        filename = f"{idx:05d}_hr"
        file_path = os.path.join(ptb_xl_path, 'records100', filename)

        # Load record
        try:
            record = wfdb.rdrecord(file_path)
            signals.append(record.p_signal)
            records.append(idx)
        except Exception as e:
            print(f"Error loading record {idx}: {e}")
            continue
    
    signals = np.array(signals)
    
    # Load patient splits
    fold_to_idx = {
        'train': [0, 1, 2, 3, 4, 5, 6, 7],
        'val': [8],
        'test': [9]
    }
    
    splits = {}
    for fold_name, fold_idx in fold_to_idx.items():
        patient_ids = metadata[metadata.strat_fold.isin(fold_idx)].patient_id.unique()
        record_ids = metadata[metadata.patient_id.isin(patient_ids)].index
        splits[fold_name] = [records.index(rid) for rid in record_ids if rid in records]
    
    return signals, metadata.loc[records], splits

def prepare_reconstruction_data(signals, splits):
    """
    Prepare input (I, II, V4) and target (all 12 leads) data for reconstruction task
    
    Standard lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    """
    print("Preparing reconstruction data...")
    
    # Normalize to [0, 1] range
    min_val = np.min(signals)
    max_val = np.max(signals)
    signals_norm = (signals - min_val) / (max_val - min_val)
    
    # Map for standard 12 leads to PTB-XL indices
    # PTB-XL order: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
    # This is actually already the standard order, but we'll be explicit
    lead_map = {
        'I': 0,    # I
        'II': 1,   # II
        'III': 2,  # III
        'aVR': 3,  # aVR
        'aVL': 4,  # aVL
        'aVF': 5,  # aVF
        'V1': 6,   # V1
        'V2': 7,   # V2
        'V3': 8,   # V3
        'V4': 9,   # V4
        'V5': 10,  # V5
        'V6': 11   # V6
    }
    
    # Input leads: I, II, V4
    input_indices = [lead_map['I'], lead_map['II'], lead_map['V4']]
    
    # Target: all 12 leads
    target_indices = list(range(12))
    
    # Prepare data for each split
    data = {}
    for split_name, indices in splits.items():
        split_signals = signals_norm[indices]
        
        # Input: 3 leads (I, II, V4)
        input_data = split_signals[:, input_indices, :]
        
        # Target: all 12 leads
        target_data = split_signals[:, target_indices, :]
        
        data[split_name] = {
            'input': input_data,
            'target': target_data
        }
    
    return data, min_val, max_val

def load_test_data(test_data_path):
    """
    Load synthetic test data for pipeline testing
    """
    print(f"Loading synthetic test data from: {test_data_path}")

    # Load data splits
    splits = {}
    for split_name in ['train', 'val', 'test']:
        input_path = os.path.join(test_data_path, f'{split_name}_input.npy')
        target_path = os.path.join(test_data_path, f'{split_name}_target.npy')

        if os.path.exists(input_path) and os.path.exists(target_path):
            input_data = np.load(input_path)
            target_data = np.load(target_path)
            splits[split_name] = {
                'input': input_data,
                'target': target_data
            }
            print(f"  {split_name}: input {input_data.shape}, target {target_data.shape}")
        else:
            print(f"  Warning: {split_name} data not found")

    # Load metadata if available
    metadata_path = os.path.join(test_data_path, 'metadata.csv')
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
    else:
        # Create dummy metadata
        total_samples = sum(len(split_data['input']) for split_data in splits.values())
        rng = np.random.default_rng(42)
        metadata = pd.DataFrame({
            'ecg_id': range(total_samples),
            'patient_id': rng.integers(1000, 9999, total_samples),
            'strat_fold': [0] * len(splits['train']['input']) +
                         [8] * len(splits['val']['input']) +
                         [9] * len(splits['test']['input'])
        })

    return splits, metadata

def main():
    args = parse_args()

    if args.test_mode:
        # Load synthetic test data
        test_data_path = args.output_path or 'data/test_data'
        splits, metadata = load_test_data(test_data_path)

        # Save data if output path is different
        if args.output_path and args.output_path != test_data_path:
            os.makedirs(args.output_path, exist_ok=True)
            for split_name, split_data in splits.items():
                np.save(os.path.join(args.output_path, f'{split_name}_input.npy'), split_data['input'])
                np.save(os.path.join(args.output_path, f'{split_name}_target.npy'), split_data['target'])
            metadata.to_csv(os.path.join(args.output_path, 'metadata.csv'), index=False)

        # Print summary
        for split_name, split_data in splits.items():
            print(f"{split_name} data:")
            print(f"  - Input shape: {split_data['input'].shape}")
            print(f"  - Target shape: {split_data['target'].shape}")

        print("Test data loading complete!")
        return

    # Original PTB-XL processing
    if not args.ptb_xl_path:
        raise ValueError("ptb_xl_path is required when not in test mode")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Load PTB-XL data
    signals, metadata, splits = load_ptb_xl(args.ptb_xl_path, args.sampling_rate)

    # Prepare reconstruction data
    data, min_val, max_val = prepare_reconstruction_data(signals, splits)

    # Save data
    print(f"Saving processed data to: {args.output_path}")
    for split_name, split_data in data.items():
        np.save(os.path.join(args.output_path, f'{split_name}_input.npy'), split_data['input'])
        np.save(os.path.join(args.output_path, f'{split_name}_target.npy'), split_data['target'])

    # Save normalization parameters
    with open(os.path.join(args.output_path, 'norm_params.pkl'), 'wb') as f:
        pickle.dump({'min': min_val, 'max': max_val}, f)

    # Save metadata
    metadata.to_csv(os.path.join(args.output_path, 'metadata.csv'))

    # Print summary
    for split_name, split_data in data.items():
        print(f"{split_name} data:")
        print(f"  - Input shape: {split_data['input'].shape}")
        print(f"  - Target shape: {split_data['target'].shape}")

    print("Data preparation complete!")