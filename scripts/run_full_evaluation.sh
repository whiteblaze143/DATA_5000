#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run full evaluation: inference (save predictions) + clinical features evaluation
# Usage: ./scripts/run_full_evaluation.sh [--venv] [--batch_size N]

BATCH_SIZE=32
USE_VENV=true
VENV_DIR=".venv"
PYTHON_CMD=python3

while [[ $# -gt 0 ]]; do
  case $1 in
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --no-venv)
      USE_VENV=false
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [ "$USE_VENV" = true ]; then
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv in $VENV_DIR..."
    $PYTHON_CMD -m venv $VENV_DIR
  fi
  # shellcheck disable=SC1091
  source $VENV_DIR/bin/activate
  echo "Activated venv: $VENV_DIR"
  pip install -r requirements.txt
fi

mkdir -p results/eval

# Step 1: Run inference and save predictions
$PYTHON_CMD scripts/evaluate_model.py --model_path models/final_exp_baseline/best_model.pt \
  --test_input data/processed_full/test_input.npy \
  --test_target data/processed_full/test_target.npy \
  --save_dir results/eval \
  --batch_size ${BATCH_SIZE} \
  --save_predictions

# Step 2: Run clinical feature evaluation on saved predictions
$PYTHON_CMD scripts/clinical_features_eval.py \
  --y_true results/eval/test_true.npy \
  --y_pred results/eval/test_pred.npy \
  --output results/eval/clinical_features_eval.png

if [ "$USE_VENV" = true ]; then
  deactivate || true
fi

echo "Full evaluation complete. Outputs in results/eval"
