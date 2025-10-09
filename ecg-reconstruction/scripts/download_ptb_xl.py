#!/usr/bin/env python3
"""
Download PTB-XL dataset from PhysioNet
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def download_ptb_xl(output_dir, subset=None):
    """
    Download PTB-XL dataset from PhysioNet

    Args:
        output_dir: Directory to save the dataset
        subset: If specified, download only a subset (e.g., 'records100' for 100 Hz data)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"

    # Files to download
    files_to_download = [
        "ptbxl_database.csv",
        "scp_statements.csv",
        "records500/RECORDS",  # 500 Hz records
    ]

    if subset == 'records100':
        # Download only 100 Hz records (smaller subset)
        files_to_download.extend([
            "records100/RECORDS",
        ])
        # Add first few record files for testing
        for i in range(1, 11):  # Download first 10 records as example
            files_to_download.append(f"records100/{i:05d}_hr.dat")
            files_to_download.append(f"records100/{i:05d}_hr.hea")
    else:
        # Download all records (this will be very large)
        print("Warning: Downloading full PTB-XL dataset (~50GB)")
        files_to_download.append("records500/RECORDS")

    print(f"Downloading PTB-XL to: {output_dir}")

    for file_path in files_to_download:
        url = base_url + file_path
        local_path = output_dir / file_path

        # Create directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading: {file_path}")

        try:
            # Use curl or wget to download
            if sys.platform == "win32":
                # Windows - use curl (usually available in Windows 10+)
                cmd = ["curl", "-o", str(local_path), url]
            else:
                # Linux/Mac - use wget
                cmd = ["wget", "-O", str(local_path), url]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error downloading {file_path}: {result.stderr}")
                continue

            print(f"✓ Downloaded: {file_path}")

        except Exception as e:
            print(f"Error downloading {file_path}: {e}")
            continue

    print("Download complete!")
    print(f"Data saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Download PTB-XL dataset')
    parser.add_argument('--output_dir', type=str, default='data/ptb_xl',
                       help='Output directory for PTB-XL data')
    parser.add_argument('--subset', type=str, choices=['records100', None],
                       help='Download subset (records100 for smaller 100Hz data)')

    args = parser.parse_args()

    download_ptb_xl(args.output_dir, args.subset)

if __name__ == '__main__':
    main()