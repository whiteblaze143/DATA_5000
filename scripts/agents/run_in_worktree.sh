#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd)"
WORKTREE_DIR="$1"
shift
CMD="$@"

mkdir -p "$WORKTREE_DIR"
git worktree add "$WORKTREE_DIR" || true
pushd "$WORKTREE_DIR" >/dev/null
echo "Running in worktree $WORKTREE_DIR: $CMD"
eval "$CMD"
popd >/dev/null
