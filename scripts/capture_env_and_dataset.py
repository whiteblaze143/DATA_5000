--#!/usr/bin/env python3
"""Capture environment information and dataset checksums.

Writes to `out_dir/env_metadata.json` and `out_dir/dataset_checksums.json`.
"""
import json
import subprocess
import sys
import os
from pathlib import Path
import hashlib

def sha256_of_file(p: Path):
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='results/env')
    parser.add_argument('--data_dir', type=str, default='data/processed_full')
    args = parser.parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = {
        'python_executable': sys.executable,
    }
    # pip freeze
    try:
        env['pip_freeze'] = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8').splitlines()
    except Exception:
        env['pip_freeze'] = []

    # system info
    try:
        env['uname'] = subprocess.check_output(['uname', '-a']).decode('utf-8').strip()
    except Exception:
        env['uname'] = ''
    try:
        env['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']).decode('utf-8').strip()
    except Exception:
        env['nvidia_smi'] = ''

    with open(outdir / 'env_metadata.json', 'w') as fh:
        json.dump(env, fh, indent=2)

    # dataset checksums
    data_dir = Path(args.data_dir)
    checksums = {}
    if data_dir.exists():
        for f in data_dir.rglob('*.npy'):
            checksums[str(f.relative_to(data_dir))] = sha256_of_file(f)
    with open(outdir / 'dataset_checksums.json', 'w') as fh:
        json.dump(checksums, fh, indent=2)

    print('Saved environment metadata to', outdir)

if __name__ == '__main__':
    main()
