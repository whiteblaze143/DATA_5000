#!/usr/bin/env bash
set -euo pipefail

PR_URL=""
REPO=""
PR_NUM=""
DRY_RUN=0
LABELS=()
REVIEWERS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --pr-url)
      PR_URL="$2"; shift 2;;
    --repo)
      REPO="$2"; shift 2;;
    --pr)
      PR_NUM="$2"; shift 2;;
    --label)
      LABELS+=("$2"); shift 2;;
    --reviewer)
      REVIEWERS+=("$2"); shift 2;;
    --dry-run)
      DRY_RUN=1; shift;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$PR_URL" && ( -z "$REPO" || -z "$PR_NUM" ) ]]; then
  echo "Provide --pr-url or (--repo and --pr)";
  exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run: labels=${LABELS[*]} reviewers=${REVIEWERS[*]}"; exit 0
fi

if [[ -n "$(command -v gh || true)" ]]; then
  for lab in "${LABELS[@]}"; do gh pr edit --add-label "$lab" "$PR_URL"; done
  if [[ ${#REVIEWERS[@]} -gt 0 ]]; then
    gh pr review --request "${REVIEWERS[*]}" "$PR_URL" || true
  fi
  exit 0
fi

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  # parse pr url to get owner/repo/pr
  if [[ -z "$REPO" ]]; then
    REPO=$(echo "$PR_URL" | sed -E 's%https?://github.com/([^/]+/[^/]+)/pull/([0-9]+).*%\1%')
    PR_NUM=$(echo "$PR_URL" | sed -E 's%https?://github.com/([^/]+/[^/]+)/pull/([0-9]+).*%\2%')
  fi
  if [[ ${#LABELS[@]} -gt 0 ]]; then
    labels_json=$(printf '%s,' "${LABELS[@]}" | sed 's/,$//')
    labels_json="[${labels_json}]"
    curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" -H "Content-Type: application/json" -d "{\"labels\": ${labels_json}}" "https://api.github.com/repos/$REPO/issues/$PR_NUM/labels"
  fi
  if [[ ${#REVIEWERS[@]} -gt 0 ]]; then
    reviewers_json=$(printf '"%s",' "${REVIEWERS[@]}" | sed 's/,$//')
    reviewers_json="{\"reviewers\": [${reviewers_json}] }"
    curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" -H "Content-Type: application/json" -d "${reviewers_json}" "https://api.github.com/repos/$REPO/pulls/$PR_NUM/requested_reviewers"
  fi
  exit 0
fi

echo "No 'gh' CLI and no GITHUB_TOKEN set; can't apply labels or request reviewers."
exit 1
