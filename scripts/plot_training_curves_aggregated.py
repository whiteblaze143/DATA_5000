#!/usr/bin/env python3
"""
Aggregate and plot training curves (validation correlation per epoch) across multiple seeds for each variant.

Usage:
  python3 scripts/plot_training_curves_aggregated.py --base_prefix models/exp --out figures/training_curves_variants.png

This script expects directories like `models/exp_<variant>_seed_<n>/training_history.json`.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def find_variant_seed_dirs(base_prefix: str):
    root = Path('.')
    variant_map = {}
    for d in root.glob(f"{base_prefix}_*_seed_*/"):
        name = d.name
        parts = name.split('_')
        if len(parts) >= 4 and parts[-2] == 'seed':
            variant = parts[-3]
            variant_map.setdefault(variant, []).append(d)
    return variant_map


def load_val_corrs(dirs):
    lists = []
    for d in dirs:
        p = d / 'training_history.json'
        if not p.exists():
            continue
        with open(p, 'r') as f:
            h = json.load(f)
        corrs = [m.get('correlation_overall', None) for m in h.get('val_metrics', [])]
        # filter None
        corrs = [c for c in corrs if c is not None]
        lists.append(corrs)
    return lists


def pad_and_stack(lists):
    if not lists:
        return None
    maxlen = max(len(l) for l in lists)
    arr = np.full((len(lists), maxlen), np.nan)
    for i, l in enumerate(lists):
        arr[i, :len(l)] = l
    return arr


def plot_variants(variant_map, out_path):
    plt.figure(figsize=(10, 4))
    for variant, dirs in variant_map.items():
        lists = load_val_corrs(dirs)
        arr = pad_and_stack(lists)
        if arr is None:
            continue
        median = np.nanmedian(arr, axis=0)
        lower = np.nanpercentile(arr, 2.5, axis=0)
        upper = np.nanpercentile(arr, 97.5, axis=0)
        epochs = np.arange(1, len(median) + 1)
        plt.plot(epochs, median, label=f"{variant} (n={arr.shape[0]})")
        plt.fill_between(epochs, lower, upper, alpha=0.2)

    plt.xlabel('Epoch')
    plt.ylabel('Validation correlation (r)')
    plt.ylim(0, 1)
    plt.title('Validation correlation across epochs (median ± 95% CI)')
    plt.legend()
    plt.grid(alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_prefix', type=str, default='models/exp', help='Prefix for model run dirs')
    parser.add_argument('--out', type=str, default='figures/training_curves_variants.png')
    args = parser.parse_args()

    variant_map = find_variant_seed_dirs(args.base_prefix)
    if not variant_map:
        raise SystemExit('No variant seed directories found; run experiments first')
    plot_variants(variant_map, Path(args.out))
    print(f"Saved training curves to {args.out}")


if __name__ == '__main__':
    main()
