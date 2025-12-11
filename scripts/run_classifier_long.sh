#!/usr/bin/env bash
# Lightweight wrapper to run classifier training with safety checks
set -euo pipefail
OUT_DIR=${1:-models/classifier_full_long}
EPOCHS=${2:-50}
BS=${3:-64}
LOGFILE=${4:-logs/classifier_train.log}
DEVICE=${5:-cuda}
shift 5 || true
EXTRA_ARGS=("$@")


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.venv/bin/activate"
fi

mkdir -p "$(dirname $LOGFILE)"
mkdir -p "$OUT_DIR"

if [ -f "$OUT_DIR/best_model.pt" ]; then
  echo "Found existing $OUT_DIR/best_model.pt; moving to $OUT_DIR/best_model.pt.bak" | tee -a "$LOGFILE"
  mv "$OUT_DIR/best_model.pt" "$OUT_DIR/best_model.pt.bak_$(date +%s)" || true
fi

CMD=(python3 scripts/train_classifier.py --data_dir data/processed_full --labels_dir data/processed_full/labels --output "$OUT_DIR" --epochs "$EPOCHS" --batch_size "$BS" --device "$DEVICE" "${EXTRA_ARGS[@]}")
./scripts/run_with_retry.sh "$LOGFILE" "${CMD[@]}"
