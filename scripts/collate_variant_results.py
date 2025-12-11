#!/usr/bin/env python3
"""Collate test_results.json across seeds for each variant and compute mean + 95% CI."""
import json
import argparse
from pathlib import Path
import numpy as np
import math

def mean_ci(x, alpha=0.05):
    arr = np.array(x)
    m = float(np.nanmean(arr))
    se = float(np.nanstd(arr, ddof=1) / math.sqrt(np.sum(~np.isnan(arr)))) if np.sum(~np.isnan(arr))>1 else float('nan')
    from scipy import stats
    if np.sum(~np.isnan(arr))>1:
        t = stats.t.ppf(1-alpha/2, df=np.sum(~np.isnan(arr))-1)
        return m, se * t
    else:
        return m, float('nan')

def collate(variant_dirs):
    results = {}
    for d in variant_dirs:
        p = Path(d)
        if not p.exists():
            continue
        tr = p / 'test_results.json'
        if tr.exists():
            data = json.loads(tr.read_text())
            results[d] = data
    # collect metrics
    metrics = {}
    # overall correlation
    corrs = [v['test_correlation_overall'] for v in results.values()]
    maes = [v['test_mae_overall'] for v in results.values()]
    snrs = [v['test_snr_overall'] for v in results.values()]
    metrics['corr_mean'], metrics['corr_ci95'] = mean_ci(corrs)
    metrics['mae_mean'], metrics['mae_ci95'] = mean_ci(maes)
    metrics['snr_mean'], metrics['snr_ci95'] = mean_ci(snrs)
    # per-lead correlations: assume same length
    per_lead = np.array([v['test_correlation_per_lead'] for v in results.values() if 'test_correlation_per_lead' in v])
    if per_lead.size>0:
        per_lead_mean = np.nanmean(per_lead, axis=0).tolist()
        per_lead_ci = []
        for i in range(per_lead.shape[1]):
            m, ci = mean_ci(per_lead[:,i].tolist())
            per_lead_ci.append(ci)
        metrics['per_lead_mean'] = per_lead_mean
        metrics['per_lead_ci95'] = per_lead_ci
    metrics['n_runs'] = len(results)
    return metrics, results

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pattern', type=str, required=True, help='Glob or prefix to locate variant runs (e.g., models/exp_baseline_* )')
    p.add_argument('--out', type=str, default='results/experiments/collated')
    args = p.parse_args()
    import glob
    dirs = sorted(glob.glob(args.pattern))
    metrics, results = collate(dirs)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp = outp.with_suffix('.json')
    outp.write_text(json.dumps({'metrics':metrics, 'runs': list(results.keys())}, indent=2))
    print('Wrote', outp)

if __name__ == '__main__':
    main()
