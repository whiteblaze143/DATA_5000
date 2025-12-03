#!/bin/bash
# Run complete experiment pipeline: train all variants then compare
# Usage: ./scripts/run_experiment_pipeline.sh

set -e

cd /home/mithunmanivannan/DATA_5000/DATA_5000
source /home/mithunmanivannan/ecg_recon_env/bin/activate

echo "=============================================="
echo "ECG Lead Reconstruction - Full Experiment"
echo "=============================================="
echo "Start: $(date)"
echo ""

# Check if training already completed
BASELINE_DONE=false
HYBRID_DONE=false
PHYSICS_DONE=false

[ -f "models/exp_baseline/test_results.json" ] && BASELINE_DONE=true
[ -f "models/exp_hybrid/test_results.json" ] && HYBRID_DONE=true
[ -f "models/exp_physics/test_results.json" ] && PHYSICS_DONE=true

# Train baseline if needed
if [ "$BASELINE_DONE" = false ]; then
    echo ">>> Training BASELINE..."
    python scripts/train_variants.py --variant baseline --output_dir models/exp_baseline
    echo "Baseline done: $(date)"
else
    echo ">>> BASELINE already trained, skipping"
fi

# Train hybrid if needed
if [ "$HYBRID_DONE" = false ]; then
    echo ""
    echo ">>> Training HYBRID..."
    python scripts/train_variants.py --variant hybrid --output_dir models/exp_hybrid
    echo "Hybrid done: $(date)"
else
    echo ">>> HYBRID already trained, skipping"
fi

# Train physics if needed
if [ "$PHYSICS_DONE" = false ]; then
    echo ""
    echo ">>> Training PHYSICS..."
    python scripts/train_variants.py --variant physics --lambda_physics 0.1 --output_dir models/exp_physics
    echo "Physics done: $(date)"
else
    echo ">>> PHYSICS already trained, skipping"
fi

echo ""
echo "=============================================="
echo "All training complete! Running comparison..."
echo "=============================================="
echo ""

# Run comparison
python scripts/compare_variants.py \
    --baseline models/exp_baseline \
    --variants models/exp_hybrid models/exp_physics \
    --names hybrid physics \
    --output models/variant_comparison.json

echo ""
echo "=============================================="
echo "Experiment complete: $(date)"
echo "=============================================="
