import json
import os
import tempfile
from pathlib import Path
import subprocess


def test_results_aggregator_basic(tmp_path):
    exp_dir = tmp_path / 'results' / 'test_exp'
    exp_dir.mkdir(parents=True)
    summary = {'exp_name': 'test_exp', 'commands': 2, 'exit_codes': [0, 1]}
    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(summary, f)
    meta0 = {'id': 0, 'cmd': 'echo hi', 'exit_code': 0}
    with open(exp_dir / 'job_0.meta.json', 'w') as f:
        json.dump(meta0, f)
    with open(exp_dir / 'job_0.log', 'w') as f:
        f.write('hello\n')
    meta1 = {'id': 1, 'cmd': 'false', 'exit_code': 1}
    with open(exp_dir / 'job_1.meta.json', 'w') as f:
        json.dump(meta1, f)
    # call aggregator
    cmd = ['python3', 'scripts/agents/results_aggregator.py', '--exp_dir', str(exp_dir)]
    subprocess.check_call(cmd)
    assert (exp_dir / 'aggregated.json').exists()
    assert (exp_dir / 'aggregated.csv').exists()
    with open(exp_dir / 'aggregated.json') as f:
        agg = json.load(f)
    assert agg['n_jobs'] == 2
