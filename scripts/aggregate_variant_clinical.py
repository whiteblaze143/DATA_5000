#!/usr/bin/env python3
"""Aggregate per-variant clinical feature CSVs and perform pairwise comparisons.

Usage: python3 scripts/aggregate_variant_clinical.py --variants results/eval/baseline results/eval/hybrid results/eval/physics --output results/eval/variant_clinical_summary.csv
"""
import argparse
import os
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def load_threshold_csv(path):
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            lid = int(r['lead_idx'])
            data[lid] = r
    return data


def load_per_sample_df(path):
    import pandas as pd
    return pd.read_csv(path)


def aggregate_variants(variant_dirs, out_csv):
    rows = []
    for vdir in variant_dirs:
        vname = Path(vdir).name
        thresh_csv = Path(vdir) / 'clinical_features_threshold_exceedance.csv'
        per_sample_csv = Path(vdir) / 'clinical_features_per_sample.csv'
        if not thresh_csv.exists():
            print('Warning: threshold CSV missing for', vdir)
            continue
        tdata = load_threshold_csv(thresh_csv)
        # read per-lead entries
        for lid, rec in tdata.items():
            rows.append({
                'variant': vname,
                'lead_idx': int(lid),
                'lead_name': rec.get('lead_name', ''),
                'n_valid': int(float(rec.get('n_valid', 0))),
                'qrs_exceed_pct': float(rec.get('qrs_exceed_pct', np.nan)),
                'pr_exceed_pct': float(rec.get('pr_exceed_pct', np.nan)),
                'qt_exceed_pct': float(rec.get('qt_exceed_pct', np.nan)),
                'hr_exceed_pct': float(rec.get('hr_exceed_pct', np.nan)),
                'qrs_duration_error_mean': float(rec.get('qrs_duration_error_mean', np.nan)),
                'pr_interval_error_mean': float(rec.get('pr_interval_error_mean', np.nan)),
                'qt_interval_error_mean': float(rec.get('qt_interval_error_mean', np.nan)),
                'hr_error_mean': float(rec.get('hr_error_mean', np.nan))
            })

    # Output CSV
    fieldnames = list(rows[0].keys()) if rows else []
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    with open(out_csv, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print('Saved variant clinical summary to', out_csv)
    return rows


def permutation_test_paired(x, y, n_permutations=2000, seed=None):
    """Paired permutation test using sign flips on differences.
    Returns a two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    d = np.array(x) - np.array(y)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return np.nan
    obs = np.abs(d.mean())
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=d.shape)
        perm = np.abs((d * signs).mean())
        if perm >= obs:
            count += 1
    p = (count + 1) / (n_permutations + 1)
    return float(p)


def compare_pairwise(variant_dirs, metric='qrs_duration_error', out_dir='results/eval', min_n_valid=30, permutation_n=5000, force_permutation=False):
    # baseline is variant_dirs[0]
    import pandas as pd
    base_dir = Path(variant_dirs[0])
    base_df = load_per_sample_df(base_dir / 'clinical_features_per_sample.csv')
    # pairwise t-tests for each lead
    comparisons = []
    skipped = []
    # helper for FDR correction
    def benjamini_hochberg(pvals):
        """Return BH-adjusted p-values preserving original order. pvals can include NaN; NaN stay NaN."""
        pvals = np.array(pvals, dtype=float)
        n = np.sum(~np.isnan(pvals))
        if n == 0:
            return pvals
        # indices of finite p-values
        idx = np.where(~np.isnan(pvals))[0]
        finite = pvals[idx]
        sorted_idx = np.argsort(finite)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, len(finite)+1)
        adj = finite * n / ranks
        # enforce monotonicity
        adj_sorted = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.full_like(pvals, np.nan)
        out[idx] = adj_sorted
        # clip at 1.0
        out = np.minimum(out, 1.0)
        return out
    for other in variant_dirs[1:]:
        other_dir = Path(other)
        other_df = load_per_sample_df(other_dir / 'clinical_features_per_sample.csv')
        # align on sample and lead
        merged = base_df.merge(other_df, on=['sample','lead_idx'], suffixes=('_base','_other'))
        for lid in sorted(merged['lead_idx'].unique()):
            subset = merged[merged['lead_idx']==lid]
            vals = subset[[f'{metric}_base', f'{metric}_other']].astype(float).dropna()
            n = len(vals)
            if n >= min_n_valid:
                base_vals = vals[f'{metric}_base']
                other_vals = vals[f'{metric}_other']
                # perform paired t-test
                try:
                    t_stat, p_val = ttest_rel(base_vals, other_vals)
                except Exception:
                    t_stat, p_val = np.nan, np.nan
                # non-parametric
                try:
                    w_stat, w_p = wilcoxon(base_vals, other_vals)
                except Exception:
                    w_stat, w_p = np.nan, np.nan
                perm_p = np.nan
                # compute permutation test either in fallback or when forced
                if force_permutation or (np.isnan(p_val) or np.isnan(w_p)):
                    try:
                        perm_p = permutation_test_paired(base_vals.values, other_vals.values, n_permutations=permutation_n)
                    except Exception:
                        perm_p = np.nan
                diff = (base_vals - other_vals)
                comparisons.append({
                    'lead_idx': int(lid),
                    'base_variant': base_dir.name,
                    'other_variant': other_dir.name,
                    'n': len(base_vals),
                    'mean_diff': float(np.mean(diff)),
                    'median_diff': float(np.median(diff)),
                    'std_diff': float(np.std(diff)),
                    'ttest_p': float(p_val) if not np.isnan(p_val) else np.nan,
                    'wilcoxon_p': float(w_p) if not np.isnan(w_p) else np.nan,
                    'permutation_p': float(perm_p) if not np.isnan(perm_p) else np.nan
                })
            else:
                skipped.append({'lead_idx': int(lid), 'base_variant': base_dir.name, 'other_variant': other_dir.name, 'n': int(n)})
            # Add multiple-testing correction for each base/other group and lead across all metrics
            # We'll compute corrected p-values per (base, other) pair after we've collected comparisons
    # Apply BH correction per pair (base_variant, other_variant)
    if comparisons:
        comps = comparisons
        out_list = []
        for base, other in {(c['base_variant'], c['other_variant']) for c in comps}:
            group = [c for c in comps if c['base_variant'] == base and c['other_variant'] == other]
            # extract p-values arrays
            t_p = [c['ttest_p'] if (c['ttest_p'] is not None) else np.nan for c in group]
            w_p = [c['wilcoxon_p'] if (c['wilcoxon_p'] is not None) else np.nan for c in group]
            perm_p = [c['permutation_p'] if (c['permutation_p'] is not None) else np.nan for c in group]
            t_adj = benjamini_hochberg(t_p)
            w_adj = benjamini_hochberg(w_p)
            perm_adj = benjamini_hochberg(perm_p)
            for i, c in enumerate(group):
                c['ttest_p_fdr'] = float(t_adj[i]) if not np.isnan(t_adj[i]) else np.nan
                c['wilcoxon_p_fdr'] = float(w_adj[i]) if not np.isnan(w_adj[i]) else np.nan
                c['permutation_p_fdr'] = float(perm_adj[i]) if not np.isnan(perm_adj[i]) else np.nan
                out_list.append(c)
        comparisons = out_list

    # Save comparisons
    path = Path(out_dir) / 'variant_pairwise_comparisons.csv'
    # collect all possible keys in case some groups didn't have pvals
    keys = sorted({k for c in comparisons for k in c.keys()}) if comparisons else []
    with open(path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=keys)
        writer.writeheader()
        for r in comparisons:
            writer.writerow(r)
    print('Saved pairwise comparisons to', path)
    # Save skipped list if any
    if skipped:
        skip_path = Path(out_dir) / 'variant_pairwise_skipped.csv'
        keys = list(skipped[0].keys())
        with open(skip_path, 'w', newline='') as sf:
            writer = csv.DictWriter(sf, fieldnames=keys)
            writer.writeheader()
            for r in skipped:
                writer.writerow(r)
        print('Saved skipped pairwise leads to', skip_path)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description='Aggregate per-variant clinical results and compare variants')
    parser.add_argument('--variants', nargs='+', help='Model variant results directories (e.g., results/eval/baseline)', required=True)
    parser.add_argument('--output', type=str, default='results/eval/variant_clinical_summary.csv')
    parser.add_argument('--compare_metric', type=str, default='qrs_duration_error')
    parser.add_argument('--min_n_valid', type=int, default=30, help='Minimum paired samples required to run statistical tests (per lead)')
    parser.add_argument('--permutation_n', type=int, default=5000, help='Number of permutations for paired permutation test (default 5000)')
    parser.add_argument('--force_permutation', action='store_true', help='Always run permutation test for all pairwise comparisons, not just as a fallback')
    parser.add_argument('--out_dir', type=str, default='results/eval')
    args = parser.parse_args()
    rows = aggregate_variants(args.variants, args.output)
    comparisons = compare_pairwise(args.variants, metric=args.compare_metric, out_dir=args.out_dir, min_n_valid=args.min_n_valid, permutation_n=args.permutation_n, force_permutation=args.force_permutation)

if __name__ == '__main__':
    main()
