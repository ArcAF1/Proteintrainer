#!/usr/bin/env python3
"""Verify all biomedical dataset URLs are accessible and working."""

import json
import requests
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Tuple

def verify_url(url: str, timeout: int = 10) -> Tuple[bool, str, int]:
    """Verify a URL is accessible and get its size."""
    try:
        # For some URLs we need special headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        # First try HEAD request
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        
        # Some servers don't support HEAD, try GET with stream
        if response.status_code >= 400:
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.close()
        
        if response.status_code == 200:
            size = int(response.headers.get('content-length', 0))
            return True, "OK", size
        else:
            return False, f"HTTP {response.status_code}", 0
            
    except requests.exceptions.Timeout:
        return False, "Timeout", 0
    except requests.exceptions.ConnectionError:
        return False, "Connection Error", 0
    except Exception as e:
        return False, str(e), 0


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def main():
    """Test all dataset URLs."""
    print("=" * 80)
    print("BIOMEDICAL DATASET URL VERIFICATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Load data sources
    with open(Path(__file__).parent / "data_sources.json") as f:
        sources = json.load(f)
        sources.pop("_comment", None)
    
    results = []
    total_size = 0
    working_count = 0
    
    print("Testing URLs...\n")
    
    for name, info in sources.items():
        url = info["url"]
        print(f"Testing {name}...", end="", flush=True)
        
        is_working, status, size = verify_url(url)
        
        if is_working:
            print(f" ✅ {status} ({format_size(size)})")
            working_count += 1
            total_size += size
        else:
            print(f" ❌ {status}")
        
        results.append({
            "name": name,
            "url": url,
            "working": is_working,
            "status": status,
            "size": size,
            "description": info.get("description", "")
        })
        
        # Small delay to avoid overwhelming servers
        time.sleep(0.5)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total datasets: {len(sources)}")
    print(f"Working URLs: {working_count}/{len(sources)} ({working_count/len(sources)*100:.1f}%)")
    print(f"Total size (accessible): {format_size(total_size)}")
    print()
    
    # List failed URLs
    failed = [r for r in results if not r["working"]]
    if failed:
        print("Failed URLs:")
        for r in failed:
            print(f"  - {r['name']}: {r['status']}")
            print(f"    URL: {r['url']}")
        print()
    
    # List working datasets by size
    print("Working datasets (sorted by size):")
    working = [r for r in results if r["working"]]
    working.sort(key=lambda x: x["size"], reverse=True)
    
    for r in working:
        print(f"  - {r['name']}: {format_size(r['size'])}")
        print(f"    {r['description']}")
    
    # Write detailed report
    report_path = Path("dataset_verification_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_datasets": len(sources),
                "working_count": working_count,
                "total_size_bytes": total_size,
                "total_size_human": format_size(total_size)
            },
            "results": results
        }, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    # Special notes for certain datasets
    print("\n" + "=" * 80)
    print("SPECIAL NOTES")
    print("=" * 80)
    
    if "pubmed_baseline" in sources:
        print("• PubMed Baseline: The URL shown is for file #1. There are 972+ files total.")
        print("  Full dataset would be ~20GB+")
    
    if "pmc_open_access" in sources:
        print("• PMC Open Access: Multiple files available. Check filelist.csv for all files.")
        print("  Full dataset would be ~100GB+")
    
    if "pubchem_compounds" in sources:
        print("• PubChem: The URL shown is for the first batch. Hundreds of files available.")
        print("  Full dataset would be ~300GB+")
    
    if "drugbank_open" in sources:
        print("• DrugBank: Requires free academic registration at https://go.drugbank.com")
    
    print("\n✅ Verification complete!")


if __name__ == "__main__":
    main() 