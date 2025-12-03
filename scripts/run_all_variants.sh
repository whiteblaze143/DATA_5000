#!/bin/bash
# Run all three model variants sequentially and compare results
# Estimated time: ~4.5 hours total (1.5h per variant)

set -e  # Exit on error

cd /home/mithunmanivannan/DATA_5000/DATA_5000
source /home/mithunmanivannan/ecg_recon_env/bin/activate

echo "=============================================="
echo "ECG Lead Reconstruction - Variant Comparison"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Run baseline
echo ">>> Training BASELINE variant..."
python scripts/train_variants.py --variant baseline --output_dir models/exp_baseline
echo "BASELINE completed at $(date)"
echo ""

# Run hybrid
echo ">>> Training HYBRID variant..."
python scripts/train_variants.py --variant hybrid --output_dir models/exp_hybrid
echo "HYBRID completed at $(date)"
echo ""

# Run physics
echo ">>> Training PHYSICS variant..."
python scripts/train_variants.py --variant physics --lambda_physics 0.1 --output_dir models/exp_physics
echo "PHYSICS completed at $(date)"
echo ""

echo "=============================================="
echo "All training complete!"
echo "End time: $(date)"
echo "=============================================="

# Compare results
echo ""
echo ">>> Test Results Summary:"
echo ""
for variant in baseline hybrid physics; do
    echo "=== $variant ==="
    if [ -f "models/exp_${variant}/test_results.json" ]; then
        cat "models/exp_${variant}/test_results.json"
    else
        echo "No test results found"
    fi
    echo ""
done
