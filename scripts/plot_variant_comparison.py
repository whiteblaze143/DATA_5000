#!/usr/bin/env python3
"""
Plot per-lead correlation comparison across variants using aggregated results produced by `aggregate_multiseed_results.py`.

Produces a grouped bar plot with 95% CI error bars for each variant and lead.

Usage:
    python scripts/plot_variant_comparison.py --aggregated_dir results/experiments --out figures/variant_comparison_corr.png
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def load_aggregated(variant_dir: Path):
    ajson = variant_dir / 'aggregated.json'
    if not ajson.exists():
        return None
    with open(ajson, 'r') as f:
        return json.load(f)


def plot(variants: list, out_path: Path, dl_only: bool = True):
    lead_indices = [6, 7, 8, 10, 11] if dl_only else list(range(12))
    lead_labels = [LEAD_NAMES[i] for i in lead_indices]

    n_leads = len(lead_labels)
    n_variants = len(variants)
    width = 0.7 / n_variants

    plt.figure(figsize=(12, 4))
    x = np.arange(n_leads)

    for vi, variant in enumerate(variants):
        vals = [variant['per_lead'][l]['mean_corr'] for l in lead_labels]
        cis = [variant['per_lead'][l]['ci_corr'] for l in lead_labels]
        lowers = [v - c[0] for v, c in zip(vals, cis)]
        uppers = [c[1] - v for v, c in zip(vals, cis)]
        errs = [lowers, uppers]
        xoff = x - 0.35 + width/2 + vi * width
        plt.bar(xoff, vals, width, yerr=errs, capsize=4, label=variant.get('name', f'var{vi}'))

    plt.xticks(x, lead_labels)
    plt.ylim(0, 1)
    plt.ylabel('Pearson r')
    plt.title('Per-Lead Correlation Across Variants (mean ± 95% CI)')
    plt.legend()
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregated_dir', type=str, default='results/experiments', help='Directory with per-variant aggregated results')
    parser.add_argument('--out', type=str, default='figures/variant_comparison_corr.png')
    parser.add_argument('--variants', type=str, nargs='*', default=None)
    parser.add_argument('--dl_only', action='store_true', help='Plot DL leads only (V1,V2,V3,V5,V6)')
    args = parser.parse_args()

    base = Path(args.aggregated_dir)
    if args.variants is None:
        variants = [p for p in base.iterdir() if p.is_dir()]
    else:
        variants = [base / v for v in args.variants]

    loaded = []
    for v in variants:
        data = load_aggregated(v)
        if data is None:
            print(f"Ignoring {v} (no aggregated.json)")
            continue
        data['name'] = v.name
        loaded.append(data)

    if len(loaded) == 0:
        raise SystemExit('No aggregated variant directories found')

    plot(loaded, Path(args.out), dl_only=args.dl_only)
    print(f"Saved variant comparison plot to {args.out}")


if __name__ == '__main__':
    main()
