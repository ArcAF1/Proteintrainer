#!/usr/bin/env python3
"""Quick verification of the three essential datasets."""

import json
from pathlib import Path
from src.verify_downloads import verify_url, format_size

def main():
    print("Verifying Essential Datasets")
    print("=" * 40)
    
    # Load data sources
    with open("src/data_sources.json") as f:
        sources = json.load(f)
    
    # Check only the three essential ones
    essential = ['chembl_sqlite', 'hetionet', 'clinical_trials']
    
    total_size = 0
    all_working = True
    
    for name in essential:
        if name in sources:
            info = sources[name]
            url = info['url']
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Expected size: {info['size']}")
            print(f"  Checking URL...", end="", flush=True)
            
            is_working, status, size = verify_url(url)
            
            if is_working:
                print(f" ✅ Working ({format_size(size)})")
                total_size += size
            else:
                print(f" ❌ {status}")
                all_working = False
    
    print("\n" + "=" * 40)
    print(f"Total download size: {format_size(total_size)}")
    print(f"Status: {'✅ All essential datasets available' if all_working else '❌ Some datasets unavailable'}")
    
    # Check local downloads
    print("\nLocal files:")
    data_dir = Path("data")
    if data_dir.exists():
        for name in essential:
            local_files = list(data_dir.glob(f"*{name}*"))
            if local_files:
                print(f"  ✅ {name}: Found {len(local_files)} file(s)")
            else:
                print(f"  ⏳ {name}: Not downloaded yet")
    else:
        print("  ⏳ No data directory yet")

if __name__ == "__main__":
    main() 