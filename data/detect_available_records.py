#!/usr/bin/env python3
"""Detect which PTB-XL records can be successfully read and produce a records list.

Writes `data/processed_full/records_list.txt` with one ecg_id per line for records that
wfdb can read and that have 12 channels.
"""
from pathlib import Path
import wfdb
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ptb_xl_path', type=str, default='data/ptb_xl_full')
    p.add_argument('--out_dir', type=str, default='data/processed_full')
    return p.parse_args()


def main():
    args = parse_args()
    p = Path(args.ptb_xl_path)
    available_files = {}
    for rec_dir in ['records100', 'records500']:
        rec_path = p / rec_dir
        if not rec_path.exists():
            continue
        for root, dirs, files in __import__('os').walk(rec_path):
            for f in files:
                if f.endswith('.hea'):
                    try:
                        rid = int(f.split('_')[0])
                    except Exception:
                        continue
                    file_path = str(Path(root) / f[:-4])
                    available_files[rid] = file_path

    # iterate through metadata order
    db_path = p / 'ptbxl_database.csv'
    import pandas as pd
    md = pd.read_csv(db_path, index_col='ecg_id')

    ok = []
    for rid in md.index:
        fp = available_files.get(rid)
        if fp is None:
            continue
        try:
            rec = wfdb.rdrecord(fp)
            if rec.p_signal.shape[1] == 12:
                ok.append(rid)
        except Exception:
            continue

    outp = Path(args.out_dir) / 'records_list.txt'
    outp.write_text('\n'.join(str(x) for x in ok))
    print(f'Wrote {len(ok)} records to {outp}')


if __name__ == '__main__':
    main()
