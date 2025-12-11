---
name: Results Aggregator
description: Aggregate per-job results for an experiment and produce `aggregated.json` and `aggregated.csv`. Optionally prepare branches with aggregated artifacts for review.
infer: true
---

Use this agent to summarize experiment runs and collect per-job logs into human-readable artifacts. Example prompts:

- "Aggregate results in `results/unet_sweep/` and create `aggregated.json`."
- "Aggregate `results/unet_sweep/`, commit aggregated JSON and CSV to a branch and open a PR (use `--commit_and_pr` via orchestrator)."
