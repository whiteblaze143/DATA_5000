#!/usr/bin/env python3
"""
Comprehensive comparison of ECG reconstruction model variants.

Performs rigorous statistical analysis including:
1. Per-lead correlation comparison across variants
2. Paired statistical tests (t-test, Wilcoxon signed-rank)
3. Effect size computation (Cohen's d, rank-biserial)
4. Bootstrap confidence intervals
5. Multiple comparison correction (Bonferroni, FDR)

Usage:
    python scripts/compare_variants.py --baseline models/exp_baseline --variants models/exp_hybrid models/exp_physics
    
Output:
    - Console summary with statistical tests
    - JSON file with all results
    - Visualization of per-lead comparisons
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
DL_LEAD_INDICES = [6, 7, 8, 10, 11]  # V1, V2, V3, V5, V6
DL_LEAD_NAMES = ['V1', 'V2', 'V3', 'V5', 'V6']
PHYSICS_LEAD_INDICES = [2, 3, 4, 5]  # III, aVR, aVL, aVF
INPUT_LEAD_INDICES = [0, 1, 9]  # I, II, V4


def load_results(model_dir: str) -> Tuple[Dict, Dict]:
    """Load test results and training history from a model directory."""
    model_dir = Path(model_dir)
    
    test_path = model_dir / 'test_results.json'
    history_path = model_dir / 'training_history.json'
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test results not found: {test_path}")
    
    with open(test_path, 'r') as f:
        test_results = json.load(f)
    
    history = {}
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
    
    return test_results, history


def get_dl_correlations(results: Dict) -> np.ndarray:
    """Extract DL lead correlations from test results."""
    all_corrs = results['test_correlation_per_lead']
    return np.array([all_corrs[i] for i in DL_LEAD_INDICES])


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for paired samples.
    
    d = mean(x - y) / std(x - y)
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large
    """
    diff = x - y
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)


def rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute rank-biserial correlation (effect size for Wilcoxon test).
    
    r = 1 - (2*W) / (n*(n+1)/2)
    where W is the smaller of the two rank sums.
    """
    diff = x - y
    diff = diff[diff != 0]  # Remove ties at zero
    n = len(diff)
    if n == 0:
        return 0.0
    
    # Get ranks of absolute differences
    abs_diff = np.abs(diff)
    ranks = stats.rankdata(abs_diff)
    
    # Sum ranks of positive and negative differences
    pos_ranks = np.sum(ranks[diff > 0])
    neg_ranks = np.sum(ranks[diff < 0])
    
    # Rank-biserial = (sum_positive - sum_negative) / total_ranks
    total = n * (n + 1) / 2
    r = (pos_ranks - neg_ranks) / total
    return r


def bootstrap_ci(x: np.ndarray, y: np.ndarray, 
                 n_bootstrap: int = 10000, 
                 confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for mean difference.
    
    Uses bias-corrected and accelerated (BCa) method approximation.
    """
    np.random.seed(42)  # Reproducibility
    diff = x - y
    n = len(diff)
    
    # Bootstrap resampling
    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        boot_means[i] = np.mean(diff[indices])
    
    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    
    return lower, upper


def paired_tests(x: np.ndarray, y: np.ndarray, 
                 alternative: str = 'two-sided') -> Dict:
    """
    Perform paired statistical tests comparing two sets of measurements.
    
    Args:
        x: First sample (e.g., baseline correlations)
        y: Second sample (e.g., hybrid correlations)
        alternative: 'two-sided', 'greater', 'less'
        
    Returns:
        Dictionary with test statistics and p-values
    """
    results = {}
    diff = x - y
    n = len(diff)
    
    # Descriptive statistics
    results['mean_x'] = float(np.mean(x))
    results['mean_y'] = float(np.mean(y))
    results['mean_diff'] = float(np.mean(diff))
    results['std_diff'] = float(np.std(diff, ddof=1))
    results['n'] = n
    
    # Paired t-test
    if n >= 2:
        t_stat, t_pval = stats.ttest_rel(x, y, alternative=alternative)
        results['ttest_statistic'] = float(t_stat)
        results['ttest_pvalue'] = float(t_pval)
    
    # Wilcoxon signed-rank test (non-parametric)
    if n >= 5:  # Minimum for meaningful Wilcoxon
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w_stat, w_pval = stats.wilcoxon(x, y, alternative=alternative)
            results['wilcoxon_statistic'] = float(w_stat)
            results['wilcoxon_pvalue'] = float(w_pval)
        except ValueError:
            # All differences are zero
            results['wilcoxon_statistic'] = None
            results['wilcoxon_pvalue'] = 1.0
    
    # Effect sizes
    results['cohens_d'] = float(cohens_d(x, y))
    results['rank_biserial'] = float(rank_biserial_correlation(x, y))
    
    # Bootstrap CI for mean difference
    ci_lower, ci_upper = bootstrap_ci(x, y)
    results['bootstrap_ci_95'] = [float(ci_lower), float(ci_upper)]
    results['ci_excludes_zero'] = not (ci_lower <= 0 <= ci_upper)
    
    return results


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_pvalue(p: float, alpha: float = 0.05) -> str:
    """Interpret p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < alpha:
        return "*"
    else:
        return "n.s."


def bonferroni_correction(pvalues: List[float], alpha: float = 0.05) -> Dict:
    """Apply Bonferroni correction for multiple comparisons."""
    m = len(pvalues)
    corrected_alpha = alpha / m
    
    return {
        'original_alpha': alpha,
        'corrected_alpha': corrected_alpha,
        'n_comparisons': m,
        'significant_uncorrected': sum(p < alpha for p in pvalues),
        'significant_corrected': sum(p < corrected_alpha for p in pvalues),
        'corrected_pvalues': [min(p * m, 1.0) for p in pvalues]
    }


def compare_two_variants(baseline_dir: str, variant_dir: str, 
                         variant_name: str) -> Dict:
    """
    Comprehensive comparison between baseline and a variant.
    """
    # Load results
    base_results, base_history = load_results(baseline_dir)
    var_results, var_history = load_results(variant_dir)
    
    # Extract DL lead correlations (the ones we care about)
    base_dl_corrs = get_dl_correlations(base_results)
    var_dl_corrs = get_dl_correlations(var_results)
    
    comparison = {
        'baseline': os.path.basename(baseline_dir),
        'variant': variant_name,
        'variant_dir': os.path.basename(variant_dir),
    }
    
    # Per-lead comparison
    per_lead = {}
    for i, lead_name in enumerate(DL_LEAD_NAMES):
        per_lead[lead_name] = {
            'baseline_corr': float(base_dl_corrs[i]),
            'variant_corr': float(var_dl_corrs[i]),
            'delta': float(var_dl_corrs[i] - base_dl_corrs[i]),
            'winner': 'variant' if var_dl_corrs[i] > base_dl_corrs[i] else 'baseline'
        }
    comparison['per_lead'] = per_lead
    
    # Overall DL leads summary
    comparison['dl_leads_summary'] = {
        'baseline_mean': float(np.mean(base_dl_corrs)),
        'baseline_std': float(np.std(base_dl_corrs)),
        'variant_mean': float(np.mean(var_dl_corrs)),
        'variant_std': float(np.std(var_dl_corrs)),
        'mean_improvement': float(np.mean(var_dl_corrs) - np.mean(base_dl_corrs)),
        'leads_improved': int(np.sum(var_dl_corrs > base_dl_corrs)),
        'leads_total': len(DL_LEAD_NAMES),
    }
    
    # Statistical tests (variant - baseline, so positive = variant better)
    # We test if variant > baseline (one-sided)
    comparison['statistical_tests'] = paired_tests(
        var_dl_corrs, base_dl_corrs, alternative='greater'
    )
    
    # Training metadata
    comparison['training_info'] = {
        'baseline_best_epoch': base_history.get('best_epoch'),
        'baseline_time_min': base_history.get('total_time_minutes'),
        'variant_best_epoch': var_history.get('best_epoch'),
        'variant_time_min': var_history.get('total_time_minutes'),
    }
    
    return comparison


def print_comparison(comparison: Dict) -> None:
    """Pretty print a comparison result."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {comparison['baseline']} vs {comparison['variant']}")
    print(f"{'='*70}")
    
    # Per-lead table
    print("\n📊 Per-Lead Correlation Comparison (DL Leads Only):")
    print("-" * 55)
    print(f"{'Lead':<8} {'Baseline':>10} {'Variant':>10} {'Delta':>10} {'Winner':>10}")
    print("-" * 55)
    
    for lead_name in DL_LEAD_NAMES:
        data = comparison['per_lead'][lead_name]
        delta = data['delta']
        delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
        winner = "→" if data['winner'] == 'variant' else "←"
        print(f"{lead_name:<8} {data['baseline_corr']:>10.4f} {data['variant_corr']:>10.4f} "
              f"{delta_str:>10} {winner:>10}")
    
    print("-" * 55)
    summary = comparison['dl_leads_summary']
    print(f"{'MEAN':<8} {summary['baseline_mean']:>10.4f} {summary['variant_mean']:>10.4f} "
          f"{summary['mean_improvement']:>+10.4f} "
          f"({summary['leads_improved']}/{summary['leads_total']} improved)")
    
    # Statistical tests
    tests = comparison['statistical_tests']
    print(f"\n📈 Statistical Analysis:")
    print("-" * 55)
    print(f"  Sample size (n leads): {tests['n']}")
    print(f"  Mean difference:       {tests['mean_diff']:+.4f} ± {tests['std_diff']:.4f}")
    
    if 'ttest_pvalue' in tests:
        sig = interpret_pvalue(tests['ttest_pvalue'])
        print(f"  Paired t-test:         t={tests['ttest_statistic']:.3f}, p={tests['ttest_pvalue']:.4f} {sig}")
    
    if 'wilcoxon_pvalue' in tests:
        sig = interpret_pvalue(tests['wilcoxon_pvalue'])
        print(f"  Wilcoxon signed-rank:  W={tests['wilcoxon_statistic']:.1f}, p={tests['wilcoxon_pvalue']:.4f} {sig}")
    
    d = tests['cohens_d']
    interp = interpret_effect_size(d)
    print(f"  Cohen's d:             {d:+.3f} ({interp})")
    print(f"  Rank-biserial r:       {tests['rank_biserial']:+.3f}")
    
    ci = tests['bootstrap_ci_95']
    excludes = "YES ✓" if tests['ci_excludes_zero'] else "NO"
    print(f"  Bootstrap 95% CI:      [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"  CI excludes zero:      {excludes}")
    
    # Interpretation
    print(f"\n🎯 Interpretation:")
    if tests['mean_diff'] > 0:
        direction = f"{comparison['variant']} outperforms {comparison['baseline']}"
    else:
        direction = f"{comparison['baseline']} outperforms {comparison['variant']}"
    
    print(f"  Direction: {direction}")
    
    # Determine statistical significance
    if tests.get('ttest_pvalue', 1.0) < 0.05 or tests['ci_excludes_zero']:
        if abs(d) >= 0.8:
            conclusion = "STRONG evidence for difference (large effect, significant)"
        elif abs(d) >= 0.5:
            conclusion = "MODERATE evidence for difference (medium effect, significant)"
        else:
            conclusion = "WEAK evidence for difference (small effect, significant)"
    else:
        if abs(d) >= 0.5:
            conclusion = "Suggestive but not significant (medium effect, p>0.05)"
        else:
            conclusion = "No meaningful difference detected"
    
    print(f"  Conclusion: {conclusion}")


def main():
    parser = argparse.ArgumentParser(description='Compare ECG model variants')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline model directory')
    parser.add_argument('--variants', type=str, nargs='+', required=True,
                        help='Paths to variant model directories')
    parser.add_argument('--names', type=str, nargs='+', default=None,
                        help='Names for variant models (optional)')
    parser.add_argument('--output', type=str, default='models/variant_comparison.json',
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Check baseline exists
    if not os.path.exists(args.baseline):
        print(f"ERROR: Baseline directory not found: {args.baseline}")
        sys.exit(1)
    
    # Default names from directory names
    if args.names is None:
        args.names = [os.path.basename(v) for v in args.variants]
    
    if len(args.names) != len(args.variants):
        print("ERROR: Number of names must match number of variants")
        sys.exit(1)
    
    print("=" * 70)
    print("ECG LEAD RECONSTRUCTION - VARIANT COMPARISON")
    print("=" * 70)
    print(f"Baseline: {args.baseline}")
    print(f"Variants: {', '.join(args.variants)}")
    
    all_comparisons = []
    all_pvalues = []
    
    for variant_path, variant_name in zip(args.variants, args.names):
        if not os.path.exists(variant_path):
            print(f"\n⚠️  Skipping {variant_name}: directory not found")
            continue
        
        try:
            comparison = compare_two_variants(args.baseline, variant_path, variant_name)
            all_comparisons.append(comparison)
            all_pvalues.append(comparison['statistical_tests'].get('ttest_pvalue', 1.0))
            print_comparison(comparison)
        except Exception as e:
            print(f"\n⚠️  Error comparing {variant_name}: {e}")
    
    if len(all_comparisons) == 0:
        print("\nNo valid comparisons completed.")
        sys.exit(1)
    
    # Multiple comparison correction
    if len(all_pvalues) > 1:
        print(f"\n{'='*70}")
        print("MULTIPLE COMPARISON CORRECTION")
        print(f"{'='*70}")
        
        bonf = bonferroni_correction(all_pvalues)
        print(f"Number of comparisons: {bonf['n_comparisons']}")
        print(f"Original α: {bonf['original_alpha']}")
        print(f"Bonferroni-corrected α: {bonf['corrected_alpha']:.4f}")
        print(f"Significant (uncorrected): {bonf['significant_uncorrected']}")
        print(f"Significant (corrected): {bonf['significant_corrected']}")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Variant':<20} {'Mean r':>10} {'Δ vs Base':>12} {'Cohen d':>10} {'p-value':>10}")
    print("-" * 70)
    
    # Baseline
    if all_comparisons:
        base_mean = all_comparisons[0]['dl_leads_summary']['baseline_mean']
        print(f"{'baseline':<20} {base_mean:>10.4f} {'---':>12} {'---':>10} {'---':>10}")
    
    # Variants
    for comp in all_comparisons:
        var_mean = comp['dl_leads_summary']['variant_mean']
        delta = comp['dl_leads_summary']['mean_improvement']
        d = comp['statistical_tests']['cohens_d']
        p = comp['statistical_tests'].get('ttest_pvalue', float('nan'))
        sig = interpret_pvalue(p) if not np.isnan(p) else ''
        print(f"{comp['variant']:<20} {var_mean:>10.4f} {delta:>+12.4f} {d:>+10.3f} {p:>9.4f} {sig}")
    
    # Determine overall winner
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    
    best_variant = None
    best_improvement = 0
    
    for comp in all_comparisons:
        improvement = comp['dl_leads_summary']['mean_improvement']
        d = abs(comp['statistical_tests']['cohens_d'])
        ci_excludes = comp['statistical_tests']['ci_excludes_zero']
        
        if improvement > best_improvement and (d >= 0.5 or ci_excludes):
            best_improvement = improvement
            best_variant = comp['variant']
    
    if best_variant and best_improvement > 0:
        print(f"✓ Best variant: {best_variant} (+{best_improvement:.4f} correlation)")
    elif all_comparisons:
        # Check if baseline is actually best
        any_worse = any(c['dl_leads_summary']['mean_improvement'] < 0 for c in all_comparisons)
        if any_worse:
            print("✓ Baseline remains the best choice (variants did not improve)")
        else:
            print("⚠ No variant shows statistically significant improvement")
            print("  Consider: more training, hyperparameter tuning, or architectural changes")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'baseline': args.baseline,
        'comparisons': all_comparisons,
        'multiple_comparison': bonferroni_correction(all_pvalues) if len(all_pvalues) > 1 else None,
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
