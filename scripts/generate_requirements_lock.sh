#!/usr/bin/env bash
set -euo pipefail
OUT=requirements.lock
echo "Generating $OUT from current environment (pip freeze)"
python -m pip freeze > $OUT
echo "Wrote $OUT"
