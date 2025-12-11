import json
import tempfile
from pathlib import Path
import subprocess


def test_aggregator_handles_enriched_meta(tmp_path):
    exp_dir = tmp_path / 'results' / 'exp_meta'
    exp_dir.mkdir(parents=True)
    summary = {'exp_name': 'exp_meta', 'commands': 1}
    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(summary, f)

    meta = {
        'id': 0,
        'cmd': 'echo hi',
        'exit_code': 0,
        'git_commit': 'abcd123',
        'git_branch': 'exp_meta/job_0',
        'pr_url': 'https://github.com/owner/repo/pull/1',
        'duration_seconds': 1.23,
        'artifacts': ['model.pt', 'metrics.json'],
    }
    with open(exp_dir / 'job_0.meta.json', 'w') as f:
        json.dump(meta, f)

    # run aggregator
    agg = subprocess.check_output(['python3', 'scripts/agents/results_aggregator.py', '--exp_dir', str(exp_dir)], text=True)
    assert 'Wrote' in agg
    out_json = exp_dir / 'aggregated.json'
    out_csv = exp_dir / 'aggregated.csv'
    assert out_json.exists()
    assert out_csv.exists()
    with open(out_json) as f:
        o = json.load(f)
    assert o['n_jobs'] == 1
    assert o['jobs'][0]['pr_url'] == 'https://github.com/owner/repo/pull/1'
