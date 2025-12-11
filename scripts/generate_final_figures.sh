#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
fi

echo "Aggregating results and generating final figures..."
python3 scripts/aggregate_multiseed_results.py --base_prefix models/exp --out_dir results/experiments --bootstrap_iters 10000
python3 scripts/plot_variant_comparison.py --aggregated_dir results/experiments --out figures/variant_comparison_corr.png --dl_only
python3 scripts/plot_training_curves_aggregated.py --base_prefix models/exp --out figures/training_curves_variants.png

echo "Done. Output in figures/ and results/experiments/"
