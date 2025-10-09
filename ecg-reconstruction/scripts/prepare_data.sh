#!/bin/bash
# filepath: scripts/prepare_data.sh

# Check if arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <path-to-ptb-xl> <output-path>"
    exit 1
fi

PTB_XL_PATH=$1
OUTPUT_PATH=$2

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_PATH

# Run data preparation script
echo "Processing PTB-XL data from $PTB_XL_PATH"
echo "Output will be saved to $OUTPUT_PATH"

# Navigate to project root directory
cd "$(dirname "$0")/.." || exit

# Run data preparation script
python data/get_data.py "$PTB_XL_PATH" "$OUTPUT_PATH"

echo "Data preparation complete!"
echo "Output files saved to $OUTPUT_PATH"