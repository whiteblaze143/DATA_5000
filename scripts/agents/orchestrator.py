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
from datetime import datetime, timezone


def run_cmd(cmd, cwd, logfile):
    start = datetime.utcnow()
    with open(logfile, 'wb') as f:
        proc = subprocess.Popen(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for b in proc.stdout:
            f.write(b)
            sys.stdout.buffer.write(b)
        proc.wait()
    end = datetime.utcnow()
    return {'rc': proc.returncode, 'start_time': start.isoformat(), 'end_time': end.isoformat()}


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
            res = fut.result()
            rc = res['rc'] if isinstance(res, dict) else res
            start_time = res.get('start_time') if isinstance(res, dict) else None
            end_time = res.get('end_time') if isinstance(res, dict) else None
            results.append(rc)
            # write per-job metadata
            # collect git info and metadata
            meta = {"id": job['id'], "cmd": job['cmd'], "exit_code": rc, "log": str(logfile)}
            if start_time:
                meta['start_time'] = start_time
            if end_time:
                meta['end_time'] = end_time
                try:
                    dur = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()
                    meta['duration_seconds'] = dur
                except Exception:
                    pass
            # git commit and branch
            try:
                commit = subprocess.check_output('git rev-parse --short HEAD', cwd=str(wt_path), shell=True, text=True).strip()
                meta['git_commit'] = commit
            except Exception:
                try:
                    meta['git_commit'] = subprocess.check_output('git rev-parse --short HEAD', shell=True, text=True).strip()
                except Exception:
                    meta['git_commit'] = None
            try:
                gbranch = subprocess.check_output('git rev-parse --abbrev-ref HEAD', cwd=str(wt_path), shell=True, text=True).strip()
                meta['git_branch'] = gbranch
            except Exception:
                try:
                    meta['git_branch'] = subprocess.check_output('git rev-parse --abbrev-ref HEAD', shell=True, text=True).strip()
                except Exception:
                    meta['git_branch'] = None
            # host and python info
            import platform

            meta['host'] = platform.node()
            meta['python_version'] = platform.python_version()

            # collect artifacts: files modified after start_time
            artifacts = []
            try:
                if start_time:
                    # treat start_time as UTC
                    st_ts = datetime.fromisoformat(start_time).replace(tzinfo=timezone.utc).timestamp()
                    for root, dirs, files in os.walk(str(wt_path)):
                        for fn in files:
                            fp = os.path.join(root, fn)
                            try:
                                mtime = os.path.getmtime(fp)
                                if mtime >= st_ts:
                                    rel = os.path.relpath(fp, str(wt_path))
                                    artifacts.append(rel)
                                    # copy artifact into the results folder
                                    art_out_dir = results_path / f"job_{job['id']}" / 'artifacts'
                                    art_out_dir.mkdir(parents=True, exist_ok=True)
                                    import shutil

                                    dest = art_out_dir / rel
                                    dest.parent.mkdir(parents=True, exist_ok=True)
                                    try:
                                        shutil.copy2(fp, dest)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                if artifacts:
                    meta['artifacts'] = artifacts
            except Exception:
                pass
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
    p.add_argument('--auto_fix_before_pr', action='store_true', help='Run the auto-fix agent (`scripts/agents/auto_fix_and_pr.sh`) on the worktree before creating a PR')
    p.add_argument('--auto_fix_script', default='scripts/agents/auto_fix_and_pr.sh', help='Path to the auto-fix script')
    p.add_argument('--results_dir', default='results')
    p.add_argument('--post_pr_labels', type=str, default='', help='Comma-separated labels to add to PRs created by the orchestrator')
    p.add_argument('--post_pr_reviewers', type=str, default='', help='Comma-separated reviewers to request on PRs created by the orchestrator')
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
                    # Optionally run auto-fix agent in the worktree before creating PRs
                    if args.auto_fix_before_pr:
                        auto_fix_script = Path(args.auto_fix_script)
                        auto_fix_cmd = [str(auto_fix_script), '--repo', os.environ.get('GITHUB_REPO', '') , '--branch', branch, '--no-pr']
                        # Respect the 'no_dry' flag: if we are still in dry mode for PR creation, run auto-fix in dry mode as well
                        if not args.commit_and_pr_no_dry:
                            auto_fix_cmd.append('--dry-run')
                        try:
                            print(f"Running auto-fix for job {i}: {' '.join(auto_fix_cmd)}")
                            subprocess.check_call(' '.join(auto_fix_cmd), shell=True)
                        except subprocess.CalledProcessError as e:
                            print(f"Auto-fix failed for job {i}: {e}; proceeding to PR step")
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
                        pr_output = subprocess.check_output(' '.join([str(x) for x in apply_cmd]), shell=True, text=True, stderr=subprocess.STDOUT)
                        print(f"apply_worktree_to_pr output for job {i}: {pr_output}")
                        # capture PR url if present
                        pr_url = None
                        for line in pr_output.splitlines():
                            if line.startswith('http') or 'github.com/' in line:
                                pr_url = line.strip()
                                break
                        if pr_url:
                            # update meta with pr_url
                            with open(meta_path, 'r') as mf:
                                meta = json.load(mf)
                            meta['pr_url'] = pr_url
                            with open(meta_path, 'w') as mf:
                                json.dump(meta, mf)
                        # Optionally run auto-review agent to add labels or request reviewers
                        if pr_url and (args.post_pr_labels or args.post_pr_reviewers):
                            labels = [l for l in args.post_pr_labels.split(',') if l]
                            reviewers = [r for r in args.post_pr_reviewers.split(',') if r]
                            review_cmd = [str(Path('scripts/agents/auto_review_agent.sh')), '--pr-url', pr_url]
                            for l in labels:
                                review_cmd.extend(['--label', l])
                            for r in reviewers:
                                review_cmd.extend(['--reviewer', r])
                            if not args.commit_and_pr_no_dry:
                                review_cmd.append('--dry-run')
                            try:
                                print(f"Running auto-review for job {i}: {' '.join(review_cmd)}")
                                subprocess.check_call(' '.join(review_cmd), shell=True)
                            except subprocess.CalledProcessError as e:
                                print(f"auto-review failed for job {i}: {e}")
                    except subprocess.CalledProcessError as e:
                        print(f"apply_worktree_to_pr failed for job {i}: {e}")
                else:
                    print(f"Skipping PR for job {i} due to non-zero exit code {meta.get('exit_code')}")
            else:
                print(f"Missing meta file for job {i}; skipping PR generation")


if __name__ == '__main__':
    main()
