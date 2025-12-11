#!/usr/bin/env bash
# Usage: ./scripts/run_with_retry.sh <logfile> <command...>
set -euo pipefail
logfile="$1"
shift
cmd=("$@")
retry_sleep=30
max_retries=20
retry_count=0
lockfile="/tmp/run_with_retry_$(basename $logfile).lock"

echo "Starting command (retry wrapper) at $(date)" | tee -a "$logfile"

# Determine project root (repo root is scripts/..)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# If a project virtualenv exists at the project root, source it to ensure consistent python
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  echo "Sourcing project virtualenv $PROJECT_ROOT/.venv" | tee -a "$logfile"
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Acquire lock to ensure single instance. Remove stale lock if older than 1h
if [ -f "$lockfile" ]; then
  age=$(( $(date +%s) - $(stat -c %Y "$lockfile") ))
  if [ "$age" -gt 3600 ]; then
    echo "Stale lock ($age s old). Removing $lockfile" | tee -a "$logfile"
    rm -f "$lockfile"
  else
    echo "Lock file $lockfile exists; another run may be active. Exiting." | tee -a "$logfile"
    exit 1
  fi
fi
touch "$lockfile"

# Use the active python binary (can be overridden via PYTHON_BIN env)
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python)}

check_deps() {
  echo "Checking Torch availability with $PYTHON_BIN" | tee -a "$logfile"
  if "$PYTHON_BIN" -c "import torch" 2>>"$logfile"; then
    echo "torch OK" | tee -a "$logfile"
  else
    echo "torch missing; attempting to install requirements.txt with $PYTHON_BIN" | tee -a "$logfile"
    "$PYTHON_BIN" -m pip install --upgrade -r requirements.txt 2>&1 | tee -a "$logfile"
    if "$PYTHON_BIN" -c "import torch" 2>>"$logfile"; then
      echo "torch installed OK" | tee -a "$logfile"
    else
      echo "Failed to import torch after install. Will proceed and rely on retries." | tee -a "$logfile"
    fi
  fi
}

trap 'rm -f "$lockfile"' EXIT

check_deps

while true; do
  echo "Running: ${cmd[*]}" | tee -a "$logfile"
  if "${cmd[@]}" 2>&1 | tee -a "$logfile"; then
    echo "Command completed successfully at $(date)" | tee -a "$logfile"
    break
  else
    echo "Command failed at $(date). Will retry after ${retry_sleep}s..." | tee -a "$logfile"
    retry_count=$((retry_count+1))
    if [ "$retry_count" -ge "$max_retries" ]; then
      echo "Reached max retries ($max_retries). Exiting with failure." | tee -a "$logfile"
      exit 1
    fi
    # re-check dependencies before next retry
    check_deps
    sleep "$retry_sleep"
  fi
done

