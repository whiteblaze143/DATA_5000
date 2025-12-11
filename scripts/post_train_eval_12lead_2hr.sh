#!/usr/bin/env bash
# Wait for 12-lead training to complete and run evaluation + ROC generation
MODEL_DIR=models/classifier_12lead_2hr
BEST_MODEL=${MODEL_DIR}/best_model.pt
EPOCH_TARGET=295
SLEEP=60
# Poll
while true; do
  export PATH=$PWD/.venv/bin:$PATH
  PYTHON="$PWD/.venv/bin/python"
  if [ -f "${MODEL_DIR}/last_checkpoint.pth" ]; then
    python - <<'PY'
import torch, sys
ckpt = torch.load('models/classifier_12lead_2hr/last_checkpoint.pth', map_location='cpu')
print('epoch', ckpt.get('epoch', -1))
if ckpt.get('epoch', 0) >= 295:
    sys.exit(0)
sys.exit(1)
PY
    if [ $? -eq 0 ]; then
      echo "Training reached target epoch; running evaluation..."
      break
    fi
  fi
  echo "Waiting for ${MODEL_DIR} epoch ${EPOCH_TARGET}..."
  sleep ${SLEEP}
done
# Run evaluations
VEVAL=results/eval
$PWD/.venv/bin/python scripts/evaluate_classifier.py --model_path ${BEST_MODEL} --input data/processed_full/test_target.npy --labels data/processed_full_12lead/labels/test_labels.npy --save_dir ${VEVAL}/classifier_true_12_2hr --device cuda
$PWD/.venv/bin/python scripts/evaluate_classifier.py --model_path ${BEST_MODEL} --input results/eval/baseline/test_pred.npy --labels data/processed_full_12lead/labels/test_labels.npy --save_dir ${VEVAL}/baseline_12_2hr --device cuda
$PWD/.venv/bin/python scripts/evaluate_classifier.py --model_path ${BEST_MODEL} --input results/eval/hybrid/test_pred.npy --labels data/processed_full_12lead/labels/test_labels.npy --save_dir ${VEVAL}/hybrid_12_2hr --device cuda
$PWD/.venv/bin/python scripts/evaluate_classifier.py --model_path ${BEST_MODEL} --input results/eval/physics/test_pred.npy --labels data/processed_full_12lead/labels/test_labels.npy --save_dir ${VEVAL}/physics_12_2hr --device cuda
# Generate ROC figures with new variants
$PWD/.venv/bin/python scripts/generate_roc_diagnostic_figures.py --variants classifier_true_12_2hr baseline_12_2hr hybrid_12_2hr physics_12_2hr --eval_dir ${VEVAL} --out_dir docs/figures/diag_stats_12_2hr --bootstrap_iters 1000 --p_adjust bh --n_jobs 4 --bootstrap_mmap memmap --mmap_dir /tmp
