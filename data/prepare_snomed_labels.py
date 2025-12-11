#!/usr/bin/env python3
"""Prepare SNOMED-style multi-label arrays from PTB-XL metadata.

Creates train/val/test numpy arrays of shape [N_samples, N_labels] and a labels.json mapping.

Usage:
  python data/prepare_snomed_labels.py --processed_dir data/processed_full --top_k 30
"""
from pathlib import Path
import argparse
import csv
import json
import ast
from collections import Counter
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--processed_dir', type=str, default='data/processed_full')
    p.add_argument('--ptb_xl_path', type=str, default='data/ptb_xl')
    p.add_argument('--top_k', type=int, default=30)
    p.add_argument('--out_dir', type=str, default='data/processed_full/labels')
    return p.parse_args()


def load_metadata(processed_dir=None, ptb_xl_path=None):
    # Prefer processed_dir/metadata.csv if present; otherwise fall back to ptb_xl_path/ptbxl_database.csv
    if processed_dir is not None:
        md_path = Path(processed_dir) / 'metadata.csv'
        if md_path.exists():
            rows = []
            with md_path.open('r') as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    rows.append(r)
            return rows
    if ptb_xl_path is None:
        raise FileNotFoundError('No metadata.csv found and no ptb_xl_path provided to construct rows')
    # Read ptbxl_database.csv and filter to available records pathing
    p = Path(ptb_xl_path)
    db_path = p / 'ptbxl_database.csv'
    if not db_path.exists():
        raise FileNotFoundError('ptbxl_database.csv not found at ptb_xl_path')
    # Read ptbxl database into rows (csv)
    with db_path.open('r') as fh:
        reader = csv.DictReader(fh)
        metadata = {int(r['ecg_id']): r for r in reader}
    # Find available record files in records100
    records100_path = p / 'records100'
    records500_path = p / 'records500'
    available_files = set()
    if records100_path.exists():
        for root, dirs, files in __import__('os').walk(records100_path):
            for file in files:
                if file.endswith('.hea'):
                    record_id = int(file.split('_')[0])
                    available_files.add(record_id)
    if records500_path.exists():
        for root, dirs, files in __import__('os').walk(records500_path):
            for file in files:
                if file.endswith('.hea'):
                    record_id = int(file.split('_')[0])
                    available_files.add(record_id)
    # Filter metadata to these records
    rows = [metadata[rid] for rid in sorted(metadata.keys()) if rid in available_files]
    return rows


def extract_scp_codes(rows):
    # scp_codes may be a stringified dict like "{'NORM':1, 'AF':1}"
    all_codes = []
    sample_codes = []
    for r in rows:
        codes_raw = r.get('scp_codes') or r.get('scp_codes_raw') or ''
        if codes_raw == '':
            sample_codes.append([])
            continue
        try:
            val = ast.literal_eval(codes_raw)
            if isinstance(val, dict):
                codes = [k for k, v in val.items() if v]
            elif isinstance(val, list):
                codes = val
            else:
                codes = []
        except Exception:
            # fallback - empty
            codes = []
        sample_codes.append(codes)
        all_codes.extend(codes)
    return sample_codes, all_codes


def split_indices(rows):
    # strat_fold likely present (0..9), map to train/val/test as in get_data
    train_idx = []
    val_idx = []
    test_idx = []
    for i, r in enumerate(rows):
        sf = r.get('strat_fold')
        try:
            sf = int(sf)
        except Exception:
            sf = None
        if sf in (0,1,2,3,4,5,6,7):
            train_idx.append(i)
        elif sf == 8:
            val_idx.append(i)
        elif sf == 9:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx, test_idx


def build_label_matrix(sample_codes, label_list):
    L = len(label_list)
    out = np.zeros((len(sample_codes), L), dtype=np.uint8)
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    for i, codes in enumerate(sample_codes):
        for c in codes:
            if c in label_to_idx:
                out[i, label_to_idx[c]] = 1
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metadata(args.processed_dir, args.ptb_xl_path)
    sample_codes, all_codes = extract_scp_codes(rows)
    cnt = Counter(all_codes)
    most_common = [c for c, _ in cnt.most_common(args.top_k)]

    labels = most_common
    label_json = out_dir / 'labels.json'
    label_json.write_text(json.dumps(labels, indent=2))

    label_matrix = build_label_matrix(sample_codes, labels)
    train_idx, val_idx, test_idx = split_indices(rows)

    np.save(out_dir / 'train_labels.npy', label_matrix[train_idx])
    np.save(out_dir / 'val_labels.npy', label_matrix[val_idx])
    np.save(out_dir / 'test_labels.npy', label_matrix[test_idx])
    print(f"Saved labels for {len(labels)} classes. train/val/test sizes:" \
          f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    # Attempt to align labels to processed inputs if available (heuristic: trim to input lengths per split)
    proc_dir = Path(args.processed_dir)
    try:
        train_inp = np.load(proc_dir / 'train_input.npy')
        val_inp = np.load(proc_dir / 'val_input.npy')
        test_inp = np.load(proc_dir / 'test_input.npy')
        lens = (train_inp.shape[0], val_inp.shape[0], test_inp.shape[0])
        label_lens = (len(train_idx), len(val_idx), len(test_idx))
        if lens != label_lens:
            print(f"Detected mismatch between processed inputs {lens} and labels {label_lens}. Aligning labels by trimming to input sizes.")
            # Trim labels to the first N samples per split
            np.save(out_dir / 'train_labels.npy', label_matrix[train_idx][:lens[0]])
            np.save(out_dir / 'val_labels.npy', label_matrix[val_idx][:lens[1]])
            np.save(out_dir / 'test_labels.npy', label_matrix[test_idx][:lens[2]])
            print('Saved aligned label arrays to', out_dir)
    except FileNotFoundError:
        # processed inputs not available - nothing to align
        pass


if __name__ == '__main__':
    main()
