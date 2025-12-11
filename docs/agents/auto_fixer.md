**Auto Fixer Agent**
- **Purpose:** Run `pre-commit` on all files, commit automatic fixes, and open a draft PR.
- **Location:** `scripts/agents/auto_fix_and_pr.sh`
- **Usage:**
  - Dry run: `scripts/agents/auto_fix_and_pr.sh --dry-run`
  - Create PR: `scripts/agents/auto_fix_and_pr.sh --title "..." --body "..."`
- **Notes:**
  - If `gh` CLI is available and authenticated, it's used to create a PR. Otherwise the script will fall back to using `GITHUB_TOKEN` to call the GitHub REST API.
  - Ensure `pre-commit` is installed and configured (`.pre-commit-config.yaml` in repo root).
