import subprocess
import tempfile
import os
from pathlib import Path


def test_orchestrator_copies_artifacts(tmp_path):
    # create a temporary git repo for the test
    repo_dir = tmp_path / 'repo'
    repo_dir.mkdir()
    cur_dir = os.getcwd()
    try:
        # initialize git repo
        subprocess.check_call(['git', 'init'], cwd=str(repo_dir))
        subprocess.check_call(['git', 'config', 'user.email', 'test@example.com'], cwd=str(repo_dir))
        subprocess.check_call(['git', 'config', 'user.name', 'testuser'], cwd=str(repo_dir))
        (repo_dir / 'README.md').write_text('# test')
        subprocess.check_call(['git', 'add', '.'], cwd=str(repo_dir))
        subprocess.check_call(['git', 'commit', '-m', 'initial'], cwd=str(repo_dir))

        # run orchestrator (non-dry) with a simple command that writes a file
        script_path = Path(os.getcwd()) / 'scripts' / 'agents' / 'orchestrator.py'
        exp_name = 'artifact_test'
        cmd = [
            'python3',
            str(script_path),
            '--exp_name', exp_name,
            '--commands', "[\"bash -lc 'echo hi > generated_artifact.txt'\"]",
            '--use_worktree',
        ]
        subprocess.check_call(cmd, cwd=str(repo_dir))

        # Assert artifacts copied
        artifacts_dir = repo_dir / 'results' / exp_name / 'job_0' / 'artifacts'
        assert artifacts_dir.exists()
        files = list(artifacts_dir.rglob('*'))
        assert any('generated_artifact.txt' in str(p) for p in files)
    finally:
        os.chdir(cur_dir)
