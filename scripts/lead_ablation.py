#!/usr/bin/env python3
"""
Lead input ablation skeleton.
- Builds alternative input arrays from existing targets for chosen lead sets
- Launches training runs per configuration (or just validates)
"""
import os
import argparse
import subprocess
import numpy as np

LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

DEFAULT_COMBOS = [
    ('I+II+V4', [0,1,9]),
    ('I+II+V3', [0,1,8]),
    ('I+II+V2', [0,1,7]),
    ('I+II+V1', [0,1,6]),
    ('I+V3+V4', [0,8,9]),
    ('II+V3+V4', [1,8,9]),
]

def make_inputs_from_targets(data_dir: str, out_dir: str, indices: list):
    os.makedirs(out_dir, exist_ok=True)
    for split in ['train','val','test']:
        tgt_path = os.path.join(data_dir, f'{split}_target.npy')
        tgt = np.load(tgt_path, mmap_mode='r')
        inputs = tgt[:, indices, :]  # [N,3,T]
        np.save(os.path.join(out_dir, f'{split}_input.npy'), inputs.astype(np.float32))
        # copy targets
        np.save(os.path.join(out_dir, f'{split}_target.npy'), np.array(tgt, dtype=np.float32))


def run_training(run_name: str, ablated_dir: str, output_root: str, epochs: int, batch_size: int, lr: float, amp: bool):
    out_dir = os.path.join(output_root, run_name)
    cmd = [
        'python','run_training.py',
        '--data_dir', ablated_dir,
        '--output_dir', out_dir,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr)
    ]
    if amp:
        cmd.append('--amp')
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description='Lead input ablation runner')
    ap.add_argument('--data_dir', required=True, help='Processed_full data dir with *_target.npy')
    ap.add_argument('--output_root', default='models/ablation', help='Root to save runs')
    ap.add_argument('--combos', nargs='*', default=None, help='Custom combos as comma-separated lead names, e.g., I,II,V3')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--dry_run', action='store_true', help='Only prepare data, skip training')
    args = ap.parse_args()

    combos = []
    if args.combos:
        for c in args.combos:
            leads = [x.strip() for x in c.split(',')]
            idx = [LEAD_NAMES.index(l) for l in leads]
            combos.append(('+'.join(leads), idx))
    else:
        combos = DEFAULT_COMBOS

    os.makedirs(args.output_root, exist_ok=True)

    for name, idx in combos:
        ablated_dir = os.path.join(args.output_root, f'data_{name.replace("+","_")}')
        print(f'Preparing inputs for {name} -> indices {idx}')
        make_inputs_from_targets(args.data_dir, ablated_dir, idx)
        if args.dry_run:
            continue
        print(f'Running training: {name}')
        run_training(name, ablated_dir, args.output_root, args.epochs, args.batch_size, args.lr, args.amp)

if __name__ == '__main__':
    main()
