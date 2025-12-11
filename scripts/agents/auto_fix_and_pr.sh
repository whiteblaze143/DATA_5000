#!/usr/bin/env bash
set -euo pipefail

# Auto-run pre-commit, commit any fixes, and create a PR from a new branch (draft)
# Usage: auto_fix_and_pr.sh [--dry-run] [--title "..."] [--body "..."] [--repo owner/repo]

DRY_RUN=0
NO_PR=0
BRANCH_NAME=""
TITLE="chore: apply autoformat with pre-commit"
BODY="Run pre-commit hooks and apply automatic fixes; prepare PR for review."
REPO_ARG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-pr)
      NO_PR=1
      shift
      ;;
    --branch)
      BRANCH_NAME="$2"
      shift 2
      ;;
    --title)
      TITLE="$2"
      shift 2
      ;;
    --body)
      BODY="$2"
      shift 2
      ;;
    --repo)
      REPO_ARG="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

echo "Running pre-commit to auto-fix files..."
if ! command -v pre-commit >/dev/null 2>&1; then
  echo "pre-commit not installed; install it first via 'pip install pre-commit'"
  exit 0
fi

pre-commit run --all-files || true

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run: pre-commit completed; no PR created." 
  exit 0
fi

# Detect if any changes were made
if ! git diff --quiet; then
  BRANCH="auto/fix-pretty-$(date +%Y%m%d%H%M%S)"
  if [[ -n "$BRANCH_NAME" ]]; then
    BRANCH="$BRANCH_NAME"
  fi
  git checkout -b "$BRANCH"
  git add -A
  git commit -m "Apply autoformat via pre-commit"
  git push -u origin "$BRANCH"

  # Create a draft PR using gh if available, otherwise fall back to REST API
  if [[ $NO_PR -eq 0 ]]; then
    if command -v gh >/dev/null 2>&1; then
      gh pr create --title "$TITLE" --body "$BODY" --base main --head "$BRANCH" --draft
      exit 0
    fi

    # Use token fallback
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    REPO=${REPO_ARG:-$(git remote get-url origin | sed -E 's%.*github.com[:/](.+).git%\1%')}
    API_URL="https://api.github.com/repos/$REPO/pulls"
    BODY_JSON=$(jq -n --arg t "$TITLE" --arg b "$BODY" --arg head "$BRANCH" --arg base "main" '{title: $t, body: $b, head: $head, base: $base, draft: true}')
    curl -s -H "Authorization: token $GITHUB_TOKEN" -X POST -d "$BODY_JSON" "$API_URL" | jq .html_url
    exit 0
    fi
  else
    echo "Skipped PR creation as requested (--no-pr)."
    exit 0
  fi

  echo "No 'gh' CLI and no GITHUB_TOKEN set; PR was not created. Branch pushed: $BRANCH"
else
  echo "No changes detected after auto-formatting; nothing to commit." 
fi
