import os
import subprocess
import tempfile


def test_auto_fix_dry_run_handles_missing_precommit():
    # Ensure script returns 0 when pre-commit isn't installed (graceful exit)
    script = os.path.join("scripts", "agents", "auto_fix_and_pr.sh")
    res = subprocess.run(["bash", script, "--dry-run"], capture_output=True, text=True)
    assert res.returncode == 0
