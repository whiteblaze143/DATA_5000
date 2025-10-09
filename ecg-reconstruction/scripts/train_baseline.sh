#!/bin/bash
# filepath: scripts/train_baseline.sh

# Navigate to project root directory
cd "$(dirname "$0")/.." || exit

# Define paths
DATA_DIR="data/processed"
OUTPUT_DIR="results/baseline"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training script with baseline parameters
python src/train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --model unet_1d \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --patience 10 \
    --seed 42

echo "Training complete! Results saved to $OUTPUT_DIR"