# Aggregation and Variant Plotting

Use these scripts to aggregate multi-seed experiment results and produce per-lead CI plots.

Typical workflow:

1. Run multi-seed experiments (see `scripts/run_multiseed_variants.sh`). Example:
```
./scripts/run_multiseed_variants.sh baseline models/exp
```

2. Aggregate and compute bootstrap CIs (default 2k bootstraps):
```
python3 scripts/aggregate_multiseed_results.py --base_prefix models/exp --out_dir results/experiments --bootstrap_iters 2000
```

Output:
- `results/experiments/<variant>/aggregated.json` and `aggregated.csv`
- `figures/<variant>_corr_per_lead_ci.png`

3. Plot cross-variant comparison from aggregated results (grouped bar plot):
```
python3 scripts/plot_variant_comparison.py --aggregated_dir results/experiments --out figures/variant_comparison_corr.png --dl_only
```

Notes:
- If some seeds are incomplete or missing test results, the aggregator will skip those seeds and produce summaries for variants that have at least one valid seed.
- For reproducibility, set `--bootstrap_iters` to 10k for final figures; smaller values are fine for exploratory runs.
