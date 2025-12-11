**Agents & Orchestration**

This directory documents the workspace's custom agents and the orchestrator tool.

Agents:
- `Experiment Orchestrator`: Create and orchestrate hyperparameter sweeps across git worktrees.
- `Auto Runner`: Execute small jobs or scripts in isolated worktrees.
- `CI Fixer`: Run tests, capture failing tests, and create a patch/PR.
- `Code Reviewer`: Provide code review guidance focussing on lint, tests, perf.
- `Data Curator`: Download/validate PTB-XL and prepare `data/processed/`.
- `Diagnostic Evaluator`: Run evaluation scripts and generate figures/CSV.
- `Merge Assistant`: Review and merge background agent changes.
- `Worktree Manager`: Manage worktrees for agents.
 - `Results Aggregator`: Summarize per-job outputs under `results/<exp>/` into `aggregated.json` and `aggregated.csv`.

Orchestrator usage
------------------
The orchestrator runs a set of commands for an experiment in isolated worktrees.
Example (dry run):

```
python3 scripts/agents/orchestrator.py --exp_name test_run --commands '["echo hello", "echo world"]' --parallel 2 --use_worktree --dry_run
```

Example (real):

```
python3 scripts/agents/orchestrator.py --exp_name unet_lr_sweep --commands "commands.json" --parallel 2 --use_worktree
```

`commands.json` example:

```
["python run_training.py --exp unet_lr_3e-4 --seed 1", "python run_training.py --exp unet_lr_1e-4 --seed 2"]
```

Notes
-----
- The orchestrator will create `results/<exp_name>/job_<i>.log` for each job.
- For safety, verify runs with `--dry_run` before disabling it.
- Use the `Experiment Orchestrator` agent to dispatch runs as background agents via the Chat UI, choosing Git worktrees for isolation.
 - Use the `Experiment Orchestrator` agent to dispatch runs as background agents via the Chat UI, choosing Git worktrees for isolation.
 - To convert a background worktree into a branch/PR for review, use `scripts/agents/apply_worktree_to_pr.sh <worktree_path> <branch_name> --dry-run` to validate, and remove `--dry-run` to actually push and create PRs (requires `gh` or `GITHUB_TOKEN`).

Applying worktree changes
------------------------
When a background agent completes and you want to bring changes back into the main repo:

1. Use `scripts/agents/apply_worktree_to_pr.sh <worktree_path> <branch_name> [pr_title]` to create a branch, push it, and open a PR (requires `gh` CLI for automatic PR creation).
2. Use the `Merge Assistant` agent to review the PR, run tests, and optionally merge if checks pass.


Agent prompt examples
---------------------
- Experiment Orchestrator: "Run a 3-job LR sweep: `commands.json` contains 3 `python run_training.py --exp` lines; use `--parallel 2`, create results under `results/unet_sweep/` and summarize exit codes."
- Auto Runner: "Run `python scripts/generate_presentation_plots.py` in an isolated worktree `plots-run` and return logs." 
- CI Fixer: "Run `pytest -q`, attach failing tests, propose minimal patch and open PR if `apply_patch=true`."
- Merge Assistant: "Review `worktrees/exp-123`, show diffs, run tests in PR branch, and merge if all checks pass."

Best practices
--------------
- Start with `--dry_run` so the orchestrator prints a plan and you can review it before running.
- Prefer `parallel` limited to available GPUs/CPUs; orchestrator creates per-job worktrees to avoid conflicts.
 - New options:
	 - `--auto_fix_before_pr`: Run the auto-fix agent (`scripts/agents/auto_fix_and_pr.sh`) inside each job's worktree before creating a PR.
	 - `--auto_fix_script`: Path to the auto-fix script (defaults to `scripts/agents/auto_fix_and_pr.sh`).
 - When creating PRs via `--commit_and_pr`, the orchestrator now captures PR URLs emitted by the apply script and saves them into each job's `job_<i>.meta.json` as `pr_url` for later aggregation.
 - New post-PR options:
	 - `--post_pr_labels`: comma-separated list of labels to add to PRs created by the orchestrator
	 - `--post_pr_reviewers`: comma-separated list of reviewers to request for the PR
	 - Post-PR actions call `scripts/agents/auto_review_agent.sh` and respect dry-run behavior.
 - Nightly runs and auto-PRs:
	 - A nightly GitHub Action `.github/workflows/nightly-orchestrator.yml` is provided to run the orchestrator on a schedule and upload `aggregated.json`/`aggregated.csv` as artifacts.
	 - The nightly job will create draft PRs (requires `GITHUB_TOKEN` which is provided by Actions as `secrets.GITHUB_TOKEN`). The action runs a small safe placeholder by default; replace `scripts/agents/commands.nightly.json` with the desired nightly commands for your experiments.
- Use `Merge Assistant` to inspect and apply changes from a background agent worktree back into the main branch.

