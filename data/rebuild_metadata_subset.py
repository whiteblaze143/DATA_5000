#!/usr/bin/env python3
"""Rebuild `metadata.csv` for a processed dataset without loading ECG signals.

This replicates the selection logic from `data/get_data.py` but avoids loading WFDB records,
so it can be used to create `data/processed_full/metadata.csv` aligned with available files
under `data/ptb_xl_full`.

Usage:
  python data/rebuild_metadata_subset.py --ptb_xl_path data/ptb_xl_full --out_dir data/processed_full
"""
from pathlib import Path
import argparse
import os
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ptb_xl_path', type=str, default='data/ptb_xl_full')
    p.add_argument('--out_dir', type=str, default='data/processed_full')
    return p.parse_args()


def main():
    args = parse_args()
    p = Path(args.ptb_xl_path)
    db_path = p / 'ptbxl_database.csv'
    if not db_path.exists():
        raise FileNotFoundError(f'{db_path} not found')
    metadata = pd.read_csv(db_path, index_col='ecg_id')

    # find available files in records100 and records500
    available_files = set()
    for rec_dir in ['records100', 'records500']:
        rec_path = p / rec_dir
        if not rec_path.exists():
            continue
        for root, dirs, files in os.walk(rec_path):
            for f in files:
                if f.endswith('.hea'):
                    try:
                        rid = int(f.split('_')[0])
                    except Exception:
                        continue
                    available_files.add(rid)

    print(f'Found {len(available_files)} available records')
    metadata_subset = metadata[metadata.index.isin(available_files)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'metadata.csv'
    metadata_subset.to_csv(out_file)
    print(f'Wrote metadata subset to {out_file} with {len(metadata_subset)} rows')


if __name__ == '__main__':
    main()
