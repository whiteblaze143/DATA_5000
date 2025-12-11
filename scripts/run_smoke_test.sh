#!/usr/bin/env bash
set -euo pipefail

# Quick end-to-end smoke test for CI or local verification.
# Steps:
# 1. Generate synthetic test data into data/test_data_smoke
# 2. Train a single-epoch baseline with a small batch size
# 3. Aggregate results and produce plots

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
fi

OUT_PREFIX=models/exp_smoke

echo "Generating synthetic data..."
python3 data/generate_test_data.py --output data/test_data_smoke --n_train 10 --n_val 2 --n_test 4

echo "Training a quick baseline for smoke test..."
OUTDIR="${OUT_PREFIX}_baseline_seed_42"
mkdir -p "$OUTDIR"
python3 scripts/train_variants.py --variant baseline --output_dir "$OUTDIR" --data_dir data/test_data_smoke --epochs 1 --seed 42

echo "Aggregating smoke results..."
python3 scripts/aggregate_multiseed_results.py --base_prefix models/exp_smoke --out_dir results/experiments --bootstrap_iters 200 --no_plot

echo "Plotting variant comparison for smoke test..."
python3 scripts/plot_variant_comparison.py --aggregated_dir results/experiments --out figures/smoke_variant_comparison.png --dl_only

echo "Smoke run complete. Outputs placed in results/experiments and figures/"
