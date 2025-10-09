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
    parser = argparse.ArgumentParser(description='Prepare PTB-XL data')
    parser.add_argument('ptb_xl_path', type=str, help='Path to PTB-XL dataset')
    parser.add_argument('output_path', type=str, help='Path to save processed data')
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
    
    # Load raw data
    records = []
    signals = []
    
    print("Loading ECG signals...")
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        # Get file path
        if row.strat_fold <= 8:  # Training
            file_path = os.path.join(ptb_xl_path, 'records100', row.filename_hr)
        else:  # Test
            file_path = os.path.join(ptb_xl_path, 'records100', row.filename_hr)
            
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

def main():
    args = parse_args()
    
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

if __name__ == "__main__":
    main()