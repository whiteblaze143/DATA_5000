#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

usage(){
  echo "Usage: $0 --base_prefix models/exp --variants baseline,hybrid,physics --seeds 42,123,7,99,2024 --interval 60 --timeout 86400 --bootstrap_iters 10000 --once"
}

BASE_PREFIX=models/exp
VARIANTS=all
SEEDS="42,123,7,99,2024"
INTERVAL=60
TIMEOUT=86400
BOOTSTRAP_ITERS=10000
ONCE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --base_prefix) BASE_PREFIX=$2; shift 2;;
    --variants) VARIANTS=$2; shift 2;;
    --seeds) SEEDS=$2; shift 2;;
    --interval) INTERVAL=$2; shift 2;;
    --timeout) TIMEOUT=$2; shift 2;;
    --bootstrap_iters) BOOTSTRAP_ITERS=$2; shift 2;;
    --once) ONCE=true; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
if [ "$VARIANTS" = "all" ]; then
  VARS=(baseline hybrid physics)
else
  IFS=',' read -ra VARS <<< "$VARIANTS"
fi

echo "Waiting for completed seeds for variants: ${VARS[*]}"
echo "Base prefix: $BASE_PREFIX will be used for seed dir names like ${BASE_PREFIX}_<variant>_seed_<n>"

start=$(date +%s)
while true; do
  all_ok=true
  for variant in "${VARS[@]}"; do
    pending=()
    for s in "${SEED_ARR[@]}"; do
      d=${BASE_PREFIX}_${variant}_seed_${s}
      if [ ! -f "$d/test_results.json" ]; then
        pending+=("$d")
      fi
    done
    if [ ${#pending[@]} -ne 0 ]; then
      all_ok=false
      echo "Variant $variant - pending seeds: ${#pending[@]}" >&2
    else
      echo "Variant $variant - all seeds complete" >&2
    fi
  done

  if $all_ok ; then
    echo "All seeds complete. Running aggregator..."
    python3 scripts/aggregate_multiseed_results.py --base_prefix $BASE_PREFIX --out_dir results/experiments --bootstrap_iters $BOOTSTRAP_ITERS
    python3 scripts/plot_variant_comparison.py --aggregated_dir results/experiments --out figures/variant_comparison_corr.png --dl_only
    echo "Aggregation and plotting completed at $(date)"
    if $ONCE ; then
      exit 0
    fi
  fi

  now=$(date +%s)
  elapsed=$((now - start))
  if [ $elapsed -gt $TIMEOUT ]; then
    echo "Timeout reached ($TIMEOUT s). Exiting." >&2
    exit 1
  fi
  sleep $INTERVAL
done
