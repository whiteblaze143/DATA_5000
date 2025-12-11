#!/usr/bin/env python3
"""
Aggregate multi-seed experiment results and compute bootstrap confidence intervals.

Produces:
- results/experiments/<variant>/aggregated.csv
- results/experiments/<variant>/aggregated.json
- figures/<variant>_corr_per_lead_ci.png

Usage:
    python scripts/aggregate_multiseed_results.py --variants models/exp_baseline models/exp_hybrid --out_dir results/experiments --bootstrap_iters 2000
"""
import argparse
import json
from pathlib import Path
import numpy as np
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import math

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
DL_LEAD_INDICES = [6, 7, 8, 10, 11]


def load_test_results(path: Path) -> Dict:
    tr = path / 'test_results.json'
    if not tr.exists():
        raise FileNotFoundError(f"Missing test_results.json in {path}")
    with open(tr, 'r') as f:
        return json.load(f)


def bootstrap_ci(arr: np.ndarray, n_bootstrap: int = 2000, confidence: float = 0.95) -> Tuple[float, float]:
    np.random.seed(42)
    n = len(arr)
    if n == 0:
        return float('nan'), float('nan')
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        boot_means[i] = float(np.mean(arr[idx]))
    alpha = 1 - confidence
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def aggregate_variant(variant_dirs: List[Path], out_dir: Path, bootstrap_iters: int = 2000) -> Dict:
    # variant_dirs: list of directories for each seed
    # Collect per-seed metrics
    per_seed_corrs = []  # list of np.array len 12
    per_seed_mae = []
    per_seed_snr = []
    seeds = []
    for d in variant_dirs:
        try:
            tr = load_test_results(d)
        except Exception:
            print(f"Skipping {d} (missing test_results.json)")
            continue
        per_seed_corrs.append(np.array(tr['test_correlation_per_lead']))
        per_seed_mae.append(np.array(tr['test_mae_per_lead']))
        per_seed_snr.append(np.array(tr.get('test_snr_per_lead', [math.nan]*12)))
        seeds.append(d.name)

    if len(per_seed_corrs) == 0:
        print(f"No valid seeds found for variant in {variant_dirs}; skipping.")
        return None

    per_seed_corrs = np.stack(per_seed_corrs)  # shape (n_seeds, 12)
    per_seed_mae = np.stack(per_seed_mae)
    per_seed_snr = np.stack(per_seed_snr)

    results = {
        'n_seeds': per_seed_corrs.shape[0],
        'seeds': seeds,
        'per_lead': {},
    }

    for i, lead in enumerate(LEAD_NAMES):
        arr_corr = per_seed_corrs[:, i]
        arr_mae = per_seed_mae[:, i]
        arr_snr = per_seed_snr[:, i]
        mean_corr = float(np.nanmean(arr_corr))
        std_corr = float(np.nanstd(arr_corr, ddof=1)) if arr_corr.size > 1 else 0.0
        ci_lower_corr, ci_upper_corr = bootstrap_ci(arr_corr, n_bootstrap=bootstrap_iters)

        mean_mae = float(np.nanmean(arr_mae))
        std_mae = float(np.nanstd(arr_mae, ddof=1)) if arr_mae.size > 1 else 0.0
        ci_lower_mae, ci_upper_mae = bootstrap_ci(arr_mae, n_bootstrap=bootstrap_iters)

        mean_snr = float(np.nanmean(arr_snr)) if not np.all(np.isnan(arr_snr)) else float('nan')
        std_snr = float(np.nanstd(arr_snr, ddof=1)) if arr_snr.size > 1 else float('nan')
        ci_lower_snr, ci_upper_snr = bootstrap_ci(arr_snr, n_bootstrap=bootstrap_iters) if not np.all(np.isnan(arr_snr)) else (float('nan'), float('nan'))

        results['per_lead'][lead] = {
            'mean_corr': mean_corr,
            'std_corr': std_corr,
            'ci_corr': [ci_lower_corr, ci_upper_corr],
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'ci_mae': [ci_lower_mae, ci_upper_mae],
            'mean_snr': mean_snr,
            'std_snr': std_snr,
            'ci_snr': [ci_lower_snr, ci_upper_snr],
        }

    # Save CSV and JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    out_json = out_dir / 'aggregated.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # CSV
    import csv
    out_csv = out_dir / 'aggregated.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lead', 'mean_corr', 'std_corr', 'ci_corr_lower', 'ci_corr_upper',
                         'mean_mae', 'std_mae', 'ci_mae_lower', 'ci_mae_upper',
                         'mean_snr', 'std_snr', 'ci_snr_lower', 'ci_snr_upper'])
        for lead in LEAD_NAMES:
            p = results['per_lead'][lead]
            writer.writerow([
                lead,
                p['mean_corr'], p['std_corr'], p['ci_corr'][0], p['ci_corr'][1],
                p['mean_mae'], p['std_mae'], p['ci_mae'][0], p['ci_mae'][1],
                p['mean_snr'], p['std_snr'], p['ci_snr'][0], p['ci_snr'][1]
            ])

    return results


def plot_per_lead_corr(results: Dict, out_fig: Path, dl_only: bool = True):
    lead_names = [l for l in LEAD_NAMES if (not dl_only) or l in [LEAD_NAMES[i] for i in DL_LEAD_INDICES]]
    means = [results['per_lead'][l]['mean_corr'] for l in lead_names]
    cis = [results['per_lead'][l]['ci_corr'] for l in lead_names]
    lower = [m - c[0] for m, c in zip(means, cis)]
    upper = [c[1] - m for m, c in zip(means, cis)]
    errs = [lower, upper]

    plt.figure(figsize=(10, 4))
    x = np.arange(len(lead_names))
    plt.bar(x, means, yerr=errs, capsize=6, color='#1f77b4', alpha=0.85)
    plt.xticks(x, lead_names)
    plt.ylabel('Pearson r')
    plt.ylim([0.0, 1.0])
    plt.title('Per-Lead Correlation (mean ± 95% CI)')
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=180)
    plt.close()


def find_variant_seed_dirs(base_prefix: str) -> Dict[str, List[Path]]:
    # base_prefix expected like 'models/exp' or 'models/exp_test'
    root = Path('.')
    variant_map = {}
    for d in root.glob(f"{base_prefix}_*_seed_*/"):
        name = d.name  # e.g., exp_baseline_seed_42
        # extract variant
        parts = name.split('_')
        # pattern: <prefix>_<variant>_seed_<seed>
        if len(parts) >= 4 and parts[-2] == 'seed':
            variant = parts[-3]
            variant_dir = d
            variant_map.setdefault(variant, []).append(variant_dir)
    return variant_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_prefix', type=str, default='models/exp',
                        help='Base prefix for multi-seed outputs (models/exp)')
    parser.add_argument('--out_dir', type=str, default='results/experiments',
                        help='Output directory for aggregated results (one per variant)')
    parser.add_argument('--bootstrap_iters', type=int, default=2000,
                        help='Bootstrap iterations (default 2000)')
    parser.add_argument('--dl_only', action='store_true', help='Plot DL leads only')
    parser.add_argument('--no_plot', action='store_true', help='Skip plot generation')
    parser.add_argument('--dirs', type=str, nargs='*', default=None,
                        help='Explicit list of directories to aggregate as separate variants')
    args = parser.parse_args()

    prefix = args.base_prefix
    out_root = Path(args.out_dir)

    variant_map = find_variant_seed_dirs(prefix)
    # If explicit dirs provided, add them as single-seed variants
    if args.dirs:
        for d in args.dirs:
            p = Path(d)
            if not p.exists():
                print(f"Warning: explicit dir {d} does not exist; skipping")
                continue
            # use p.name as variant key and single-element list as dirs
            variant_map.setdefault(p.name, []).append(p)
    if not variant_map:
        raise SystemExit(f"No variant seed directories found with prefix {prefix}_*_seed_*")

    all_summaries = {}
    for variant, dirs in variant_map.items():
        print(f"Processing variant {variant} with {len(dirs)} seeds")
        # sort directories alphabetically to make seed order deterministic
        dirs_sorted = sorted(dirs)
        out_dir = out_root / variant
        try:
            summary = aggregate_variant(dirs_sorted, out_dir, bootstrap_iters=args.bootstrap_iters)
            if summary is None:
                print(f"No valid seeds for variant {variant}. Skipping plot and save.")
                continue
            all_summaries[variant] = summary
            if not args.no_plot:
                fig_out = Path('figures') / f"{variant}_corr_per_lead_ci.png"
                plot_per_lead_corr(summary, fig_out, dl_only=args.dl_only)
                print(f"Wrote plot: {fig_out}")
        except Exception as e:
            print(f"Error processing {variant}: {e}")

    # Save overall summary
    with open('results/experiments/aggregated_all.json', 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print("Aggregation complete. Saved results under results/experiments/")


if __name__ == '__main__':
    main()
