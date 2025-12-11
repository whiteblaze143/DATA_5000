#!/usr/bin/env python3
"""
Simple experiment orchestrator: runs commands in isolated git worktrees and collects logs.
Usage (dry run): python scripts/agents/orchestrator.py --exp_name test_run --commands '["echo hi","echo bye"]' --dry_run
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, cwd, logfile):
    with open(logfile, 'wb') as f:
        proc = subprocess.Popen(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for b in proc.stdout:
            f.write(b)
            sys.stdout.buffer.write(b)
        proc.wait()
    return proc.returncode


def create_git_worktree(base, name):
    wt = base / name
    if wt.exists():
        return wt
    cmd = f"git worktree add {wt}"
    subprocess.check_call(cmd, shell=True)
    return wt


def orchestrate(exp_name, commands, parallel=1, use_worktree=True, dry_run=True, results_dir='results'):
    root = Path.cwd()
    results_path = root / results_dir / exp_name
    results_path.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, cmd in enumerate(commands):
        job = {"id": i, "cmd": cmd, "worktree": f"worktrees/{exp_name}/job_{i}" if use_worktree and parallel>1 else f"worktrees/{exp_name}"}
        jobs.append(job)

    plan = {"exp_name": exp_name, "n_jobs": len(jobs), "parallel": parallel, "use_worktree": use_worktree, "jobs": jobs}
    print("Plan:")
    print(json.dumps(plan, indent=2))
    if dry_run:
        return plan

    results = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {}
        for job in jobs:
            if use_worktree:
                wt_path = Path(job['worktree'])
                wt_path.mkdir(parents=True, exist_ok=True)
                # create a git worktree branch for isolation
                branch = f"{exp_name}/job_{job['id']}"
                try:
                    subprocess.check_call(f"git worktree add -b {branch} {wt_path}", shell=True)
                except subprocess.CalledProcessError:
                    # branch or worktree may already exist; try adding without -b
                    try:
                        subprocess.check_call(f"git worktree add {wt_path}", shell=True)
                    except subprocess.CalledProcessError:
                        pass
            else:
                wt_path = root
                branch = None
            logfile = results_path / f"job_{job['id']}.log"
            metafile = results_path / f"job_{job['id']}.meta.json"
            futures[ex.submit(run_cmd, job['cmd'], str(wt_path), str(logfile))] = (job, wt_path, branch, logfile, metafile)

        for fut in as_completed(futures):
            job, wt_path, branch, logfile, metafile = futures[fut]
            rc = fut.result()
            results.append(rc)
            # write per-job metadata
            meta = {"id": job['id'], "cmd": job['cmd'], "exit_code": rc, "log": str(logfile)}
            with open(metafile, 'w') as f:
                json.dump(meta, f)

    summary = {"exp_name": exp_name, "commands": len(commands), "exit_codes": results, "timestamp": datetime.utcnow().isoformat()}
    with open(results_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Done. Summary:", json.dumps(summary))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', required=True)
    p.add_argument('--commands', type=str, default='[]', help='JSON serialized array of commands or path to file')
    p.add_argument('--parallel', type=int, default=1)
    p.add_argument('--use_worktree', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--commit_and_pr', action='store_true', help='Commit worktree and create PR for each successful job (calls apply_worktree_to_pr.sh).')
    p.add_argument('--commit_and_pr_no_dry', action='store_true', help='When used with --commit_and_pr, create PRs without --dry-run on apply_worktree_to_pr.sh')
    p.add_argument('--results_dir', default='results')
    args = p.parse_args()

    if os.path.exists(args.commands):
        with open(args.commands) as f:
            commands = json.load(f)
    else:
        commands = json.loads(args.commands)

    summary = orchestrate(args.exp_name, commands, parallel=args.parallel, use_worktree=args.use_worktree, dry_run=args.dry_run, results_dir=args.results_dir)
    # Optionally create PRs for successful jobs
    if args.commit_and_pr and not args.dry_run:
        root = Path.cwd()
        results_path = root / args.results_dir / args.exp_name
        for i in range(len(commands)):
            meta_path = results_path / f"job_{i}.meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get('exit_code', 1) == 0:
                    worktree = Path(f"worktrees/{args.exp_name}/job_{i}")
                    branch = f"{args.exp_name}/job_{i}"
                    apply_cmd = [
                        str(Path('scripts/agents/apply_worktree_to_pr.sh')),
                        str(worktree),
                        branch,
                        '--repo', os.environ.get('GITHUB_REPO', '')
                    ]
                    if not args.commit_and_pr_no_dry:
                        apply_cmd.append('--dry-run')
                    # run the apply script
                    try:
                        subprocess.check_call(' '.join([str(x) for x in apply_cmd]), shell=True)
                    except subprocess.CalledProcessError as e:
                        print(f"apply_worktree_to_pr failed for job {i}: {e}")
                else:
                    print(f"Skipping PR for job {i} due to non-zero exit code {meta.get('exit_code')}")
            else:
                print(f"Missing meta file for job {i}; skipping PR generation")


if __name__ == '__main__':
    main()
