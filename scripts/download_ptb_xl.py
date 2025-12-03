#!/usr/bin/env python3
"""
Download PTB-XL dataset from PhysioNet using wfdb library.

This script properly downloads the PTB-XL dataset which contains 21,799 
12-lead ECG records of 10 seconds duration sampled at 100Hz and 500Hz.

Usage:
    python scripts/download_ptb_xl.py --output_dir data/ptb_xl
    python scripts/download_ptb_xl.py --output_dir data/ptb_xl --records100_only
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import wfdb
except ImportError:
    print("Error: wfdb not installed. Run: pip install wfdb")
    sys.exit(1)


def download_ptb_xl(output_dir: str, records100_only: bool = False):
    """
    Download PTB-XL dataset from PhysioNet using wfdb.
    
    Args:
        output_dir: Directory to save the dataset
        records100_only: If True, download only 100Hz records (smaller, ~500MB)
                        If False, download 500Hz records (~2GB)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    db_name = "ptb-xl/1.0.3"
    
    print("=" * 60)
    print("PTB-XL Dataset Download")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Record type: {'100Hz (smaller)' if records100_only else '500Hz (full quality)'}")
    print()
    
    # Download metadata files first
    print("Step 1/2: Downloading metadata files...")
    metadata_files = [
        "ptbxl_database.csv",
        "scp_statements.csv",
    ]
    
    for fname in metadata_files:
        local_path = output_dir / fname
        if local_path.exists():
            print(f"  ✓ {fname} (already exists)")
            continue
            
        try:
            # Download using wfdb
            wfdb.dl_files(db_name, output_dir, [fname])
            print(f"  ✓ {fname}")
        except Exception as e:
            print(f"  ✗ {fname}: {e}")
            # Try alternative method
            try:
                import urllib.request
                url = f"https://physionet.org/files/{db_name}/{fname}"
                urllib.request.urlretrieve(url, local_path)
                print(f"  ✓ {fname} (via urllib)")
            except Exception as e2:
                print(f"  ✗ Failed to download {fname}: {e2}")
    
    # Download ECG records
    print("\nStep 2/2: Downloading ECG records...")
    
    if records100_only:
        # Download 100Hz records (smaller)
        record_dir = "records100"
        print("  Downloading 100Hz records (~500MB)...")
    else:
        # Download 500Hz records (full quality)
        record_dir = "records500"
        print("  Downloading 500Hz records (~2GB)...")
    
    try:
        # Use wfdb.dl_database to download all records in the folder
        # This handles the subdirectory structure properly
        wfdb.dl_database(
            db_name,
            dl_dir=str(output_dir),
            records=None,  # Download all
            annotators=None,
            keep_subdirs=True,
            overwrite=False
        )
        print("  ✓ ECG records downloaded successfully")
    except Exception as e:
        print(f"  ✗ Error downloading records: {e}")
        print("\nTrying alternative download method...")
        
        # Alternative: Download records individually based on RECORDS file
        try:
            import pandas as pd
            import urllib.request
            from tqdm import tqdm
            
            # Read metadata to get record list
            metadata_path = output_dir / "ptbxl_database.csv"
            if metadata_path.exists():
                df = pd.read_csv(metadata_path)
                
                if records100_only:
                    records = df['filename_lr'].tolist()  # 100Hz
                else:
                    records = df['filename_hr'].tolist()  # 500Hz
                
                print(f"  Found {len(records)} records to download")
                
                for record in tqdm(records[:100], desc="Downloading"):  # Start with first 100
                    for ext in ['.dat', '.hea']:
                        fname = record + ext
                        local_path = output_dir / fname
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        if not local_path.exists():
                            url = f"https://physionet.org/files/{db_name}/{fname}"
                            try:
                                urllib.request.urlretrieve(url, local_path)
                            except:
                                pass
                                
        except Exception as e2:
            print(f"  Alternative method also failed: {e2}")
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    # Check what was downloaded
    metadata_csv = output_dir / "ptbxl_database.csv"
    if metadata_csv.exists():
        import pandas as pd
        df = pd.read_csv(metadata_csv)
        print(f"Metadata: {len(df)} ECG records in database")
    
    # Count actual record files
    for record_type in ['records100', 'records500']:
        record_path = output_dir / record_type
        if record_path.exists():
            hea_files = list(record_path.rglob("*.hea"))
            print(f"{record_type}: {len(hea_files)} record files downloaded")
    
    print(f"\nData saved to: {output_dir}")
    print("\nNext step: Process data with:")
    print(f"  python data/get_data.py --ptb_xl_path {output_dir} --output_path data/processed")

def main():
    parser = argparse.ArgumentParser(
        description='Download PTB-XL dataset from PhysioNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download full 500Hz dataset (~2GB)
    python scripts/download_ptb_xl.py --output_dir data/ptb_xl
    
    # Download smaller 100Hz dataset (~500MB) 
    python scripts/download_ptb_xl.py --output_dir data/ptb_xl --records100_only
        """
    )
    parser.add_argument('--output_dir', type=str, default='data/ptb_xl',
                       help='Output directory for PTB-XL data')
    parser.add_argument('--records100_only', action='store_true',
                       help='Download only 100Hz records (smaller, faster)')

    args = parser.parse_args()

    download_ptb_xl(args.output_dir, args.records100_only)


if __name__ == '__main__':
    main()