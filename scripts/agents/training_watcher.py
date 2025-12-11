#!/usr/bin/env python3
"""Watch a training log, restart on failure, run evaluation on completion, and prepare PR summary.

Usage:
  python3 scripts/agents/training_watcher.py --log logs/classifier_train.log --session classifier_train \
      --restart-cmd "source .venv/bin/activate && ./scripts/run_with_retry.sh logs/classifier_train.log python3 scripts/train_classifier.py --data_dir data/processed_full --labels_dir data/processed_full/labels --output models/classifier_full_long --epochs 50 --batch_size 64 --device cuda" \
      --eval-cmd "python3 scripts/evaluate_classifier.py --model_path models/classifier_full_long/best_model.pt --input data/processed_full/test_input.npy --labels data/processed_full/labels/test_labels.npy --save_dir results/eval/classifier_full_long" \
      --target-epoch 50
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def tail_contains(path: Path, pattern: str, lookback_lines=50) -> bool:
    if not path.exists():
        return False
    with open(path, 'r', errors='replace') as f:
        lines = f.readlines()[-lookback_lines:]
    for l in lines:
        if re.search(pattern, l):
            return True
    return False


def last_epoch_from_log(path: Path) -> int:
    if not path.exists():
        return 0
    last = 0
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = re.search(r"Epoch\s*(\d+)(?:/|-)\s*(\d+)", line)
            if m:
                try:
                    e = int(m.group(1))
                    last = max(last, e)
                except Exception:
                    pass
    return last


def tmux_session_exists(name: str) -> bool:
    try:
        out = subprocess.check_output(['tmux', 'ls'], text=True, stderr=subprocess.DEVNULL)
        return any(line.split(':', 1)[0] == name for line in out.splitlines())
    except subprocess.CalledProcessError:
        return False


def start_restart_command(cmd: str, cwd: str = '.') -> None:
    # run command in a new tmux session detached so watcher can continue
    # Use a timestamped session name for clarity
    session = f"training_restart_{int(time.time())}"
    full = f"tmux new-session -d -s {session} 'bash -lc {shlex.quote(cmd)}'"
    subprocess.check_call(full, shell=True, cwd=cwd)


def run_eval(eval_cmd: str):
    print(f"Running evaluation: {eval_cmd}")
    try:
        subprocess.check_call(eval_cmd, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print("Evaluation failed:", e)
        return False


def try_create_pr_from_results(results_dir: Path, title: str, body: str, dry_run=True, branch_prefix='auto/results'):
    # Create a PR draft using gh if present, otherwise write a markdown file for manual PR
    # Make the branch and commit the report file into a branch so PR has content
    repo_root = Path.cwd()
    branch = f"{branch_prefix}/{int(time.time())}"
    try:
        subprocess.check_call(['git', 'checkout', '-b', branch], cwd=str(repo_root))
        report_file = results_dir / 'PR_SUMMARY.md'
        report_file.write_text(body)
        subprocess.check_call(['git', 'add', str(report_file)], cwd=str(repo_root))
        subprocess.check_call(['git', 'commit', '-m', f"Auto PR: results {branch}"], cwd=str(repo_root))
        subprocess.check_call(['git', 'push', '-u', 'origin', branch], cwd=str(repo_root))
    except subprocess.CalledProcessError as e:
        print("Git operations for PR creation failed:", e)
        return None

    if shutil_which('gh') and not dry_run:
        cmd = f"gh pr create --title {shlex.quote(title)} --body {shlex.quote(body)} --base main --head {branch} --draft"
        if dry_run:
            print("Dry-run PR command:", cmd)
            return None
        try:
            out = subprocess.check_output(cmd, shell=True, text=True)
            print(out)
            return out.strip()
        except subprocess.CalledProcessError as e:
            print("gh pr create failed:", e)
            return None
    else:
        md = results_dir / 'PR_DRAFT.md'
        md.write_text(f"# {title}\n\n{body}\n")
        print("Wrote PR draft to", md)
        return str(md)


def shutil_which(name: str) -> bool:
    return any(
        (Path(p) / name).exists()
        for p in os.environ.get('PATH', '').split(':')
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', required=True, help='Path to training log to watch')
    p.add_argument('--session', default='', help='tmux session name that runs training (optional)')
    p.add_argument('--restart-cmd', default='', help='Shell command to restart the training')
    p.add_argument('--eval-cmd', default='', help='Shell command to run evaluation after successful training')
    p.add_argument('--target-epoch', type=int, default=0, help='Epoch to consider training complete')
    p.add_argument('--poll-interval', type=int, default=60, help='Seconds between checks')
    p.add_argument('--results-dir', default='results/watch_reports', help='Where to write reports and PR drafts')
    p.add_argument('--dry-run-pr', action='store_true', help='Do not create PRs automatically; write drafts instead')
    p.add_argument('--create-pr', action='store_true', help='Create a draft PR automatically when training completes')
    p.add_argument('--pr-labels', type=str, default='', help='Comma-separated labels to add to created PR')
    p.add_argument('--pr-reviewers', type=str, default='', help='Comma-separated reviewers to request for the PR')
    args = p.parse_args()

    logp = Path(args.log)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    restart_attempts = 0
    last_epoch = last_epoch_from_log(logp)
    print(f"Starting watcher: last_epoch={last_epoch}")

    while True:
        # Check for completion
        epoch = last_epoch_from_log(logp)
        if args.target_epoch and epoch >= args.target_epoch:
            print(f"Detected completion at epoch {epoch}")
            # run evaluation
            if args.eval_cmd:
                ok = run_eval(args.eval_cmd)
                report = results_dir / f'completed_{int(time.time())}.txt'
                report.write_text(f"Completed epoch {epoch}; eval_ok={ok}\n")
            else:
                report = results_dir / f'completed_{int(time.time())}.txt'
                report.write_text(f"Completed epoch {epoch}; no eval configured\n")

            # prepare PR draft
            title = f"Automated results: classifier training completed ({epoch} epochs)"
            body = report.read_text()
            pr_url = try_create_pr_from_results(results_dir, title, body, dry_run=(not args.create_pr))
            # Optionally add labels and reviewers
            if pr_url and (args.pr_labels or args.pr_reviewers):
                labels = [l for l in args.pr_labels.split(',') if l]
                reviewers = [r for r in args.pr_reviewers.split(',') if r]
                review_cmd = [str(Path('scripts/agents/auto_review_agent.sh')), '--pr-url', pr_url]
                for l in labels:
                    review_cmd.extend(['--label', l])
                for r in reviewers:
                    review_cmd.extend(['--reviewer', r])
                if not args.create_pr:
                    review_cmd.append('--dry-run')
                try:
                    print(f"Running auto-review: {' '.join(review_cmd)}")
                    subprocess.check_call(' '.join(review_cmd), shell=True)
                except subprocess.CalledProcessError as e:
                    print(f"auto-review failed: {e}")
            return

        # Check for failures: look for Traceback or killed
        if tail_contains(logp, r"Traceback \(|Exception:\s"):
            print("Detected exception in log; will attempt to restart")
            restart_attempts += 1
            if args.restart_cmd:
                start_restart_command(args.restart_cmd)
            time.sleep(30)

        # Check session missing
        if args.session and not tmux_session_exists(args.session):
            print(f"tmux session {args.session} not found; attempting restart (attempt {restart_attempts+1})")
            restart_attempts += 1
            if args.restart_cmd:
                start_restart_command(args.restart_cmd)

        # Update last_epoch seen to avoid re-detecting completion
        last_epoch = epoch
        time.sleep(args.poll_interval)


if __name__ == '__main__':
    main()
