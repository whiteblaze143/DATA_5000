#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG=logs/diag_boot_supervised.log
mkdir -p logs
while true; do
  if tmux has-session -t diag_boot 2>/dev/null; then
    echo "diag_boot session active at $(date)" >> "$LOG"
  else
    echo "diag_boot missing; starting supervised job at $(date)" | tee -a "$LOG"
    tmux new-session -d -s diag_boot "bash -lc 'cd $PROJECT_ROOT && source $PROJECT_ROOT/.venv/bin/activate && ./scripts/run_with_retry.sh logs/diag_boot_supervised.log python3 scripts/generate_roc_diagnostic_figures.py --variants classifier_true baseline hybrid physics --eval_dir results/eval --out_dir docs/figures/diag_stats_boot1000 --bootstrap_iters 1000 --n_jobs $(nproc) --bootstrap_mmap shm --mmap_dir /tmp --bootstrap_idx_generation seeded --seed 42 --p_adjust bh --use_delong --top_k 10'"
  fi
  sleep 60
done
