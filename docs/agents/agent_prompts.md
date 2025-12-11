## Agent prompt templates

Use these suggested prompts when invoking agents via Chat, background agents, or subagents.

- Experiment Orchestrator
  - Prompt: "Run the following commands as a hyperparameter sweep under `exp_name=unet_sweep`. Use `parallel=2` and `use_worktree=true`. Commands: [<commands.json path>]." 
  - Example: "Run `commands.example.json` with parallel=2 and produce a summary JSON in `results/unet_sweep/summary.json`."

- Auto Runner
  - Prompt: "Run `python scripts/generate_presentation_plots.py` in worktree `plots-20251211` and return logs to `results/plots-20251211/`."

- CI Fixer
  - Prompt: "Run `pytest -q`. When failures arise, propose minimal fix. If `apply_patch=true`, commit changes to a worktree and open a PR."

- Code Reviewer
  - Prompt: "Review `src/models/` for memory/perf issues. Suggest changes and, if `apply_patch=true`, create a PR with changes."

- Data Curator
  - Prompt: "Validate `data/processed_full` and check labels for alignment. If misaligned, suggest script changes to fix label alignment and create a PR with the fix."

- Diagnostic Evaluator
  - Prompt: "Run the ROC/bootstrapping evaluation for `results/eval/baseline` and `results/eval/classifier_true` with `bootstrap_iters=1000`. Generate plots under `docs/figures/` and a summary CSV.":

- Merge Assistant
  - Prompt: "Review worktree `worktrees/exp-123`, run tests, and if all pass, create a PR and merge."

- Results Aggregator
  - Prompt: "Aggregate results in `results/unet_sweep/` and produce `aggregated.json` and `aggregated.csv`. If `--commit_and_pr` is used via the orchestrator, make the PR dry-run first."
