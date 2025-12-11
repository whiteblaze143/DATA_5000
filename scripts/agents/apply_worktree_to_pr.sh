#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [OPTIONS] <worktree_path> <branch_name>

Create a PR from the given worktree branch.

Options:
  --no-push            Do not push the branch to origin
  --no-gh              Do not use 'gh' CLI; force using GitHub API via curl
  --repo OWNER/REPO    Override repository (owner/repo). If omitted, auto-detected
  --base BRANCH        Base branch for PR (default: main)
  --title TEXT         PR title (default: 'Auto PR from <branch>')
  --body TEXT          PR body (default: auto message)
  --draft              Create PR as a draft
  --dry-run            Do not perform network actions (no push, no PR)
  --help               Show this help and exit

Examples:
  $0 --dry-run /path/to/worktree feature-branch
  $0 --repo myorg/myrepo --title "Fix" /path work-branch
EOF
}

# Defaults
NO_PUSH=0
NO_GH=0
DRY_RUN=0
REPO=""
BASE="main"
TITLE=""
BODY=""
DRAFT=0

POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-push) NO_PUSH=1; shift ;;
    --no-gh) NO_GH=1; shift ;;
    --repo) REPO="${2:-}"; shift 2 ;;
    --base) BASE="${2:-}"; shift 2 ;;
    --title) TITLE="${2:-}"; shift 2 ;;
    --body) BODY="${2:-}"; shift 2 ;;
    --draft) DRAFT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help) usage; exit 0;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [ $# -lt 2 ]; then
  echo "Error: missing required arguments." >&2
  usage
  exit 1
fi

WORKTREE_PATH=$1
BRANCH_NAME=$2
TITLE=${TITLE:-"Auto PR from ${BRANCH_NAME}"}
BODY=${BODY:-"Auto PR created for worktree ${WORKTREE_PATH}"}

if [ ! -d "$WORKTREE_PATH" ]; then
  echo "Error: worktree path '$WORKTREE_PATH' does not exist." >&2
  exit 1
fi

pushd "$WORKTREE_PATH" >/dev/null

# Ensure branch exists (create or checkout)
if git rev-parse --verify "$BRANCH_NAME" >/dev/null 2>&1; then
  git checkout "$BRANCH_NAME"
else
  git checkout -b "$BRANCH_NAME"
fi

git add -A
git commit -m "Auto-PR: ${TITLE}" || echo "Nothing to commit"

if [ "$NO_PUSH" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  git push --set-upstream origin "$BRANCH_NAME"
else
  echo "Skipping push (NO_PUSH=$NO_PUSH, DRY_RUN=$DRY_RUN)"
fi

# Helper to detect repo from git remote origin
detect_repo() {
  if [ -n "$REPO" ]; then
    echo "$REPO"
    return
  fi
  # try to parse origin
  if git remote get-url origin >/dev/null 2>&1; then
    url=$(git remote get-url origin)
    # convert ssh or https to owner/repo
    owner_repo=$(python3 - <<PY
import re,sys
u=sys.stdin.read().strip()
u=u.replace('git@github.com:', 'https://github.com/')
u=u.replace('ssh://git@github.com/', 'https://github.com/')
u=u.rstrip('.git')
m=re.search(r'github.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$', u)
if m:
    print(m.group('owner')+'/'+m.group('repo'))
else:
    sys.exit(1)
PY
    ) || true
    echo "$owner_repo"
    return
  fi
  echo ""
}

REPO_DETECT=$(detect_repo)
if [ -z "$REPO_DETECT" ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    REPO_DETECT=""
  else
    echo "Error: unable to determine repository. Use --repo OWNER/REPO" >&2
    popd >/dev/null
    exit 1
  fi
fi

REPO=${REPO_DETECT}

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run: would create PR on ${REPO} (head=${BRANCH_NAME}, base=${BASE})"
  popd >/dev/null
  exit 0
fi

# Create PR using gh if available unless user disabled it
if [ "$NO_GH" -eq 0 ] && command -v gh >/dev/null 2>&1; then
  GH_ARGS=(pr create --title "$TITLE" --body "$BODY" --head "$BRANCH_NAME" --base "$BASE")
  if [ "$DRAFT" -eq 1 ]; then
    GH_ARGS+=(--draft)
  fi
  if [ -n "$REPO" ]; then
    GH_ARGS+=(--repo "$REPO")
  fi
  echo "Using gh to create PR: gh ${GH_ARGS[*]}"
  gh "${GH_ARGS[@]}" || { echo "gh pr create failed" >&2; popd >/dev/null; exit 1; }
else
  # Use GitHub REST API via curl; require GITHUB_TOKEN
  if [ -z "${GITHUB_TOKEN:-}" ] && [ -z "${GH_TOKEN:-}" ]; then
    echo "Error: GITHUB_TOKEN or GH_TOKEN is required for curl-based PR creation." >&2
    popd >/dev/null
    exit 1
  fi
  TOKEN="${GITHUB_TOKEN:-${GH_TOKEN}}"

  # Build JSON payload safely using python
  PAYLOAD=$(python3 - <<PY
import json
payload={'title': '$TITLE', 'head': '$BRANCH_NAME', 'base': '$BASE', 'body': '$BODY', 'draft': True if $DRAFT==1 else False}
print(json.dumps(payload))
PY
  )
  API_URL="https://api.github.com/repos/${REPO}/pulls"
  echo "Using curl to POST to ${API_URL}"
  http_status=$(curl -sS -o /tmp/pr_resp -w '%{http_code}' -X POST -H "Authorization: token ${TOKEN}" -H "Accept: application/vnd.github.v3+json" -d "${PAYLOAD}" "${API_URL}" || true)
  if [ "$http_status" = "201" ]; then
    echo "PR created successfully. Response:"
    cat /tmp/pr_resp
  else
    echo "Failed to create PR (HTTP ${http_status}). Response:"
    cat /tmp/pr_resp >&2
    popd >/dev/null
    rm -f /tmp/pr_resp
    exit 1
  fi
  rm -f /tmp/pr_resp
fi

popd >/dev/null
#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <worktree_path> <branch_name> [pr_title]"
  exit 1
fi
WORKTREE_PATH=$1
BRANCH_NAME=$2
PR_TITLE=${3:-"Auto PR from worktree ${BRANCH_NAME}"}

pushd "$WORKTREE_PATH" >/dev/null
git checkout -b "$BRANCH_NAME" || git checkout "$BRANCH_NAME"
git add -A
git commit -m "Auto-PR: ${PR_TITLE}" || echo "Nothing to commit"
git push origin "$BRANCH_NAME"
if command -v gh >/dev/null 2>&1; then
  gh pr create --fill --title "$PR_TITLE" --body "Auto PR created for worktree ${WORKTREE_PATH}" || true
else
  echo "gh CLI not available; run: gh pr create --fill --title \"${PR_TITLE}\""
fi
popd >/dev/null
