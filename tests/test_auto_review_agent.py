import subprocess
from pathlib import Path


def test_auto_review_agent_dry_run():
    script = Path('scripts') / 'agents' / 'auto_review_agent.sh'
    cmd = ['bash', str(script), '--pr-url', 'https://github.com/foo/bar/pull/1', '--label', 'auto', '--reviewer', 'alice', '--dry-run']
    out = subprocess.check_output(cmd, text=True)
    assert 'Dry run' in out
