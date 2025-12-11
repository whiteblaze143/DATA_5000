import os
import subprocess
import tempfile
from pathlib import Path


def run(cmd, cwd=None):
    print('RUN:', cmd)
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def test_apply_worktree_dry_run(tmp_path):
    # Setup a bare repo and a clone
    bare = tmp_path / 'bare.git'
    clone = tmp_path / 'clone'
    bare.mkdir()
    run('git init --bare', cwd=str(bare))
    run(f'git clone {bare} {clone}', cwd=str(tmp_path))
    # initial commit
    (clone / 'README.md').write_text('# repo')
    run('git add README.md && git commit -m "init"', cwd=str(clone))
    run('git push origin HEAD:main', cwd=str(clone))
    # create a worktree under clone
    wt = clone / 'worktrees' / 'job_0'
    wt.parent.mkdir(parents=True, exist_ok=True)
    run(f'git worktree add -b test_job {wt}', cwd=str(clone))
    # modify a file in worktree
    (wt / 'notes.txt').write_text('hello')
    # call apply script with dry-run
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'agents' / 'apply_worktree_to_pr.sh'
    cmd = f'bash {script} --dry-run {wt} test_job'
    run(cmd, cwd=str(clone))
