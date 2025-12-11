**Experiment Metadata and Artifacts**
- **Purpose:** Standardize metadata produced by `scripts/agents/orchestrator.py`
- **Location:** Per-experiment folder `results/<exp_name>/` containing `job_<i>.meta.json`, `job_<i>.log`, `aggregated.json`, and `aggregated.csv`.
- **Metadata fields:**
  - `id`: job id (int)
  - `cmd`: command executed (string)
  - `exit_code`: exit status (int)
  - `log`: path to the full log
  - `start_time`: job start in ISO8601
  - `end_time`: job end in ISO8601
  - `duration_seconds`: elapsed seconds (float)
  - `git_commit`: short SHA of HEAD inside the worktree
  - `git_branch`: branch name of the worktree
  - `pr_url`: optional; PR URL if a PR was created for this worktree
  - `host`: hostname where the job ran
  - `python_version`: python version string
  - `artifacts`: list of files modified since job start (relative to worktree)

- **Artifacts capture:** The orchestrator inspects the worktree for files with modification times greater than job `start_time`; those files are reported in `artifacts` and can be copied to `results/<exp_name>/job_<i>/artifacts/` by downstream scripts if desired.
  - The `results_aggregator.py` includes the list of artifacts in the `aggregated.csv` column `artifacts` as a semicolon-separated list.
