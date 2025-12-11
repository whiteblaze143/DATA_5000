import subprocess
import time
from pathlib import Path


def test_watcher_detects_completion(tmp_path):
    log = tmp_path / 'log.txt'
    # write lines simulating epochs
    log.write_text('Epoch 1/3\nEpoch 2/3\nEpoch 3/3\n')
    # run watcher with target 3 and dry-run PR
    cmd = [
        'python3', 'scripts/agents/training_watcher.py',
        '--log', str(log),
        '--target-epoch', '3',
        '--poll-interval', '1',
        '--results-dir', str(tmp_path / 'results'),
        '--dry-run-pr'
    ]
    out = subprocess.check_output(cmd, text=True)
    assert 'Detected completion' in out


def test_try_create_pr_dry_run(tmp_path):
    # Create a git repo then call try_create_pr_from_results in dry-run mode; ensure draft is written
    from scripts.agents import training_watcher as tw
    repo = tmp_path / 'repo'
    repo.mkdir()
    # init git
    subprocess.check_call(['git', 'init'], cwd=str(repo))
    subprocess.check_call(['git', 'config', 'user.email', 'test@example.com'], cwd=str(repo))
    subprocess.check_call(['git', 'config', 'user.name', 'testuser'], cwd=str(repo))
    (repo / 'README.md').write_text('# repo')
    subprocess.check_call(['git', 'add', '.'], cwd=str(repo))
    subprocess.check_call(['git', 'commit', '-m', 'init'], cwd=str(repo))
    # create results folder and call PR function
    rdir = repo / 'results'
    rdir.mkdir()
    title = 'Test Auto PR'
    body = 'This is a test'
    # run in dry-run: no GH and no token set; should write PR_DRAFT.md
    cur = os.getcwd()
    try:
        os.chdir(repo)
        res = tw.try_create_pr_from_results(rdir, title, body, dry_run=True)
        assert res is not None
        assert (rdir / 'PR_SUMMARY.md').exists() or (rdir / 'PR_DRAFT.md').exists()
    finally:
        os.chdir(cur)
