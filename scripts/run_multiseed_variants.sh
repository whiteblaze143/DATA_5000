#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_multiseed_variants.sh [baseline|hybrid|physics] [outdir_prefix]
VARIANT=${1:-all}
OUT_PREFIX=${2:-models/exp}
PARALLEL_SEEDS=${3:-0}
SEEDS=(42 123 7 99 2024)

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
# Source virtualenv if present
if [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

mkdir -p results/experiments

run_variant_seed() {
  local variant=$1
  local seed=$2
  local outdir=${OUT_PREFIX}_${variant}_seed_${seed}
  mkdir -p "$outdir"
  echo "Starting $variant seed $seed -> $outdir"
  python3 scripts/train_variants.py --variant "$variant" --output_dir "$outdir" --seed "$seed"
}

if [ "$VARIANT" = "all" ]; then
  VARS=(baseline hybrid physics)
else
  VARS=($VARIANT)
fi

for v in "${VARS[@]}"; do
  if [ "$PARALLEL_SEEDS" -eq 1 ]; then
    for s in "${SEEDS[@]}"; do
      run_variant_seed "$v" "$s" &
      sleep 1
    done
    # Wait for seeds for this variant to finish
    wait
  else
    for s in "${SEEDS[@]}"; do
      run_variant_seed "$v" "$s"
    done
  fi
done

echo "All runs completed"
