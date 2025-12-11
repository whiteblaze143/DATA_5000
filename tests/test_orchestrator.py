import json
import subprocess
import tempfile
from pathlib import Path


def test_orchestrator_dry_run():
    # Dry run should show a plan and exit 0
    cmd = [
        'python3',
        'scripts/agents/orchestrator.py',
        '--exp_name',
        'test_dry',
        '--commands',
        '["echo hello","echo world"]',
        '--dry_run',
    ]
    out = subprocess.check_output(cmd, text=True)
    assert 'Plan:' in out
    assert 'test_dry' in out


def test_orchestrator_autofix_flag_dry_run():
    # Dry run with auto-fix should still print a plan and not fail
    cmd = [
        'python3',
        'scripts/agents/orchestrator.py',
        '--exp_name',
        'test_autofix',
        '--commands',
        '["echo hi"]',
        '--dry_run',
        '--auto_fix_before_pr',
    ]
    out = subprocess.check_output(cmd, text=True)
    assert 'Plan:' in out
    assert 'test_autofix' in out
