#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run inference and clinical eval for multiple model variants
# Usage: ./scripts/run_eval_variants.sh [--models "models/final_exp_baseline models/final_exp_hybrid models/final_exp_physics"] [--batch_size 32]

MODELS=("models/final_exp_baseline" "models/final_exp_hybrid" "models/final_exp_physics")
BATCH_SIZE=32
PYTHON_CMD=python3
VENV_DIR=".venv"
USE_VENV=true

AGGREGATE=false
MIN_N_VALID=30
PERM_N=5000
FORCE_PERM=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --models)
      IFS=',' read -ra MODELS <<< "$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --aggregate)
      AGGREGATE=true
      shift
      ;;
    --min_n_valid)
      MIN_N_VALID="$2"
      shift 2
      ;;
    --permutation_n)
      PERM_N="$2"
      shift 2
      ;;
    --force_permutation)
      FORCE_PERM=true
      shift
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
    $PYTHON_CMD -m venv $VENV_DIR
  fi
  source $VENV_DIR/bin/activate
  pip install -r requirements.txt
fi

# Ensure results/eval folder
mkdir -p results/eval

# Function to map model directory to variant label and model type
variant_label_from_path(){
  local modelpath="$1"
  case "$modelpath" in
    *baseline*) echo "baseline" ;;
    *hybrid*) echo "hybrid" ;;
    *physics*) echo "physics" ;;
    *) echo "custom" ;;
  esac
}

model_type_from_path(){
  local modelpath="$1"
  case "$modelpath" in
    *hybrid*) echo "unet_hybrid" ;;
    *leadspecific*) echo "unet_leadspecific" ;;
    *) echo "unet" ;;
  esac
}

for mp in "${MODELS[@]}"; do
  VARIANT=$(variant_label_from_path "$mp")
  MODEL_PATH="$mp/best_model.pt"
  if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model path not found: $MODEL_PATH. Skipping variant $VARIANT"
    continue
  fi
  MODEL_TYPE=$(model_type_from_path "$mp")
  OUT_DIR="results/eval/${VARIANT}"
  mkdir -p "$OUT_DIR"

  echo "Running inference for variant: $VARIANT (type=$MODEL_TYPE)"
  $PYTHON_CMD scripts/evaluate_model.py \
    --model_path "$MODEL_PATH" \
    --model_type $MODEL_TYPE \
    --test_input data/processed_full/test_input.npy \
    --test_target data/processed_full/test_target.npy \
    --save_dir "$OUT_DIR" \
    --batch_size ${BATCH_SIZE} \
    --save_predictions

  echo "Running clinical feature eval for variant: $VARIANT"
  $PYTHON_CMD scripts/clinical_features_eval.py \
    --y_true "$OUT_DIR/test_true.npy" \
    --y_pred "$OUT_DIR/test_pred.npy" \
    --lead_idx all \
    --save_csv \
    --csv_path "$OUT_DIR/clinical_features_per_sample.csv" \
    --out_dir "$OUT_DIR" \
    --output "$OUT_DIR/clinical_features_eval.png" \
    --qrs-threshold 10 --pr-threshold 20 --qt-threshold 30 --hr-threshold 5

  echo "Finished variant: $VARIANT"
  echo "Outputs: $OUT_DIR"
done

if [ "$USE_VENV" = true ]; then
  deactivate || true
fi

echo "All variants processed. Results in results/eval/<variant>/"

if [ "$AGGREGATE" = true ]; then
  echo "Running aggregation and pairwise comparisons..."
  VARS="${MODELS[@]/#/results/eval/}"
  $PYTHON_CMD scripts/aggregate_variant_clinical.py --variants "results/eval/baseline" "results/eval/hybrid" "results/eval/physics" --out_dir results/eval --output results/eval/variant_clinical_summary.csv --min_n_valid $MIN_N_VALID --permutation_n $PERM_N $( [ $FORCE_PERM = true ] && echo --force_permutation || true )
  echo "Aggregation finished. Outputs in results/eval/"
fi
