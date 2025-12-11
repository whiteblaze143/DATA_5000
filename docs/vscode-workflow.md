**VS Code Workflow (leveraging Nov 2025 features)**

- **Enable chat sessions & agents:** Open the Command Palette and ensure Chat features are enabled. Workspace settings already enable session list and restore previous session.
- **Use custom agents:** Open Chat -> Agents menu and pick one of the agents (Experiment Orchestrator, Reconstructor Agent, Data Curator, CI Fixer, Code Reviewer, Diagnostic Evaluator, Merge Assistant, Worktree Manager) to plan tasks or execute scripts.
- **Run long jobs as background agents:** When you hand off a job to a background agent, choose the Git worktree option to isolate changes from your main workspace.
- **Orchestrate experiments & sweeps:** Use `Experiment Orchestrator` (agent) or `scripts/agents/orchestrator.py` to dispatch sweeps across multiple isolated worktrees and collect logs in `results/<exp_name>/`.
  - Start with `--dry_run` to validate the plan before executing.
  - For parallel jobs, the orchestrator will create per-job worktrees `worktrees/<exp_name>/job_<i>` to avoid conflicts.
- **Attach context:** Right-click errors, tests, or files and use "Add Context" in chat so the agent has precise info to act on.
- **Approve fetch and external domains carefully:** The first time an agent fetches a URL you'll be prompted to approve the domain; approve trusted domains only.
- **Use tasks for automation:** Use `Terminal -> Run Task...` to run `Train Baseline`, `Run Tests` or `Generate Presentation Plots`. You can also run tasks via background agents.
- **Use Claude skills and subagents:** If you maintain any Claude skills under `${workspaceFolder}.claude/skills/`, agents can load them when `chat.useClaudeSkills` is enabled.

Quick tips
- Open Chat quickly: `Ctrl+Alt+C` (workspace keybinding added).
- Run `pytest -q` for tests and ask the Reconstructor Agent to fix failures by attaching the output.
