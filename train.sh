#!/bin/bash
# =============================================================================
# ECG Lead Reconstruction - VM Training Script
# =============================================================================
# 
# Usage:
#   ./train.sh                    # Quick test with synthetic data
#   ./train.sh --full             # Full training with PTB-XL data
#   ./train.sh --resume           # Resume from checkpoint
#
# Prerequisites:
#   1. SSH into VM: ssh user@vm-address
#   2. Clone repo: git clone <repo-url>
#   3. cd ecg-reconstruction
#   4. chmod +x train.sh
#   5. ./train.sh
#
# =============================================================================

set -e  # Exit on error

# Get script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "ECG Lead Reconstruction Training"
echo "=============================================="
echo "Project root: $SCRIPT_DIR"
echo ""

# =============================================================================
# Environment Setup
# =============================================================================

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Install dependencies if needed
if [ ! -f "venv/.deps_installed" ]; then
    echo ""
    echo "Installing dependencies..."
    pip install --upgrade pip
    
    # Install PyTorch with CUDA support
    # Uncomment the appropriate line for your CUDA version:
    # CUDA 11.8:
    # pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    # CUDA 12.1:
    # pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    # CPU only:
    # pip install torch torchvision
    
    # Default: auto-detect (works for most setups)
    pip install torch torchvision
    
    # Install other requirements
    pip install -r requirements.txt
    
    touch venv/.deps_installed
    echo "Dependencies installed!"
fi

# =============================================================================
# Check GPU
# =============================================================================

echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || echo "GPU check failed"

# =============================================================================
# Parse Arguments
# =============================================================================

MODE="test"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            MODE="full"
            shift
            ;;
        --test)
            MODE="test"
            shift
            ;;
        --resume)
            EXTRA_ARGS="$EXTRA_ARGS --resume"
            shift
            ;;
        --epochs)
            EXTRA_ARGS="$EXTRA_ARGS --epochs $2"
            shift 2
            ;;
        --batch_size)
            EXTRA_ARGS="$EXTRA_ARGS --batch_size $2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# =============================================================================
# Run Training
# =============================================================================

echo ""
echo "=============================================="
echo "Starting Training (mode: $MODE)"
echo "=============================================="
echo ""

if [ "$MODE" == "test" ]; then
    # Quick test with synthetic data
    python run_training.py \
        --test_mode \
        --epochs 5 \
        --batch_size 16 \
        $EXTRA_ARGS
        
elif [ "$MODE" == "full" ]; then
    # Full training with PTB-XL data
    python run_training.py \
        --vm_mode \
        --data_dir data/processed \
        --epochs 100 \
        --batch_size 64 \
        --amp \
        $EXTRA_ARGS
fi

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Check results in: models/"
