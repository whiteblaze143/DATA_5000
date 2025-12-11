#!/usr/bin/env python3
"""Aggregate per-job results for an experiment run.
Writes `aggregated.json` and `aggregated.csv` under the experiment results folder.
"""
import argparse
import json
from pathlib import Path
import csv


def aggregate(exp_dir: Path):
    summary_file = exp_dir / 'summary.json'
    if not summary_file.exists():
        raise SystemExit(f"Missing summary.json in {exp_dir}")

    with open(summary_file) as f:
        summary = json.load(f)

    jobs = []
    i = 0
    while True:
        meta = exp_dir / f"job_{i}.meta.json"
        logf = exp_dir / f"job_{i}.log"
        if not meta.exists() and not logf.exists():
            break
        if meta.exists():
            with open(meta) as f:
                m = json.load(f)
        else:
            m = {"id": i, "exit_code": None}
        tail = None
        if logf.exists():
            # read last 2000 chars for a quick tail
            with open(logf, 'rb') as lf:
                lf.seek(0, 2)
                size = lf.tell()
                lf.seek(max(0, size-2000))
                tail = lf.read().decode(errors='replace')
        m['log_tail'] = tail
        jobs.append(m)
        i += 1

    out = {
        'exp_name': summary.get('exp_name'),
        'n_jobs': len(jobs),
        'jobs': jobs,
        'summary': summary,
    }

    out_json = exp_dir / 'aggregated.json'
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)

    out_csv = exp_dir / 'aggregated.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'exit_code', 'cmd'])
        for j in jobs:
            writer.writerow([j.get('id'), j.get('exit_code'), j.get('cmd')])

    print(f"Wrote {out_json} and {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir', required=True, help='Path to results/<exp_name> folder')
    args = p.parse_args()
    aggregate(Path(args.exp_dir))


if __name__ == '__main__':
    main()
