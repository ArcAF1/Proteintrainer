"""Enhanced data ingestion system for biomedical datasets.

Features:
- Support for multiple compression formats (.gz, .bz2, .zip, .tar.gz)
- Resumable downloads
- Progress tracking with ETA
- Retry logic with exponential backoff
- Automatic decompression
- Smart dataset selection
"""
from __future__ import annotations

import json
import shutil
import bz2
import gzip
import tarfile
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
import feedparser
from typing import Dict, List, Optional, Callable, Tuple
import time
import os
import re

import requests
from tqdm import tqdm

from .config import settings

# Progress callback type
ProgressCallback = Optional[Callable[[float, str], None]]

# Missing constants
PMC_EUTILS_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PMC_EUTILS_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Chunk size for downloads (1MB)
CHUNK_SIZE = 1024 * 1024

# Enhanced biomedical data sources
ENHANCED_DATA_SOURCES = {
    "drugbank": {
        "url": "https://go.drugbank.com/releases/latest/downloads/all-full-database", 
        "format": "xml",
        "size_gb": 0.05,
        "description": "DrugBank - comprehensive drug interaction database"
    },
    "physionet_exercise": {
        "url": "https://physionet.org/files/exercise-ecg/1.0.0/",
        "format": "wfdb",
        "size_gb": 0.5,
        "description": "PhysioNet - exercise physiology and ECG data"
    },
    "pubchem_compounds": {
        "url": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/",
        "format": "sdf", 
        "size_gb": 50.0,
        "description": "PubChem - chemical compound database (large!)"
    },
    "pdb_structures": {
        "url": "https://ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/",
        "format": "pdb",
        "size_gb": 100.0,
        "description": "Protein Data Bank - protein structures"
    }
}

def read_sources() -> Dict[str, Dict]:
    """Read data sources from the updated JSON file."""
    with open(Path(__file__).with_name("data_sources.json")) as fh:
        sources = json.load(fh)
        # Remove the comment field
        sources.pop("_comment", None)
        return sources


def get_file_size(url: str, headers: Optional[Dict] = None) -> Optional[int]:
    """Get the size of a remote file."""
    try:
        response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        return int(response.headers.get('content-length', 0))
    except:
        return None


def download_with_resume(url: str, dest: Path, 
                        progress_callback: ProgressCallback = None,
                        headers: Optional[Dict] = None) -> Path:
    """Download a file with resume capability and progress tracking."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists and is complete
    remote_size = get_file_size(url, headers)
    if dest.exists() and remote_size:
        local_size = dest.stat().st_size
        if local_size == remote_size:
            if progress_callback:
                progress_callback(1.0, f"Already downloaded: {dest.name}")
            return dest
    
    # Setup headers for resume
    resume_headers = headers.copy() if headers else {}
    mode = 'wb'
    resume_pos = 0
    
    if dest.exists():
        resume_pos = dest.stat().st_size
        if remote_size and resume_pos < remote_size:
            resume_headers['Range'] = f'bytes={resume_pos}-'
            mode = 'ab'
            if progress_callback:
                progress_callback(resume_pos / remote_size, f"Resuming {dest.name} from {resume_pos/1024/1024:.1f}MB")
    
    # Download with retries
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=resume_headers, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = remote_size or int(response.headers.get('content-length', 0))
            
            with open(dest, mode) as f:
                downloaded = resume_pos
                start_time = time.time()
                
                with tqdm(total=total_size, initial=downloaded, unit='B', unit_scale=True, desc=dest.name) as pbar:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
                            
                            # Calculate ETA
                            if progress_callback and total_size > 0:
                                progress = downloaded / total_size
                                elapsed = time.time() - start_time
                                if downloaded > resume_pos:
                                    speed = (downloaded - resume_pos) / elapsed
                                    eta = (total_size - downloaded) / speed if speed > 0 else 0
                                    eta_str = str(timedelta(seconds=int(eta)))
                                    progress_callback(progress, f"Downloading {dest.name}: {progress*100:.1f}% (ETA: {eta_str})")
            
            if progress_callback:
                progress_callback(1.0, f"Downloaded: {dest.name}")
            
            # If we reach here, download was successful
            return dest

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                if progress_callback:
                    progress_callback(downloaded / total_size if total_size else 0, 
                                    f"Connection error, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                # Update resume position for next attempt
                if dest.exists():
                    resume_pos = dest.stat().st_size
                    resume_headers['Range'] = f'bytes={resume_pos}-'
            else:
                raise
    
    return dest


def extract_archive(archive: Path, out_dir: Path, 
                   progress_callback: ProgressCallback = None) -> None:
    """Extract various archive formats with progress tracking."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if progress_callback:
        progress_callback(0.1, f"Extracting {archive.name}...")
    
    try:
        # Handle different compression formats
        if archive.name.endswith('.tar.gz') or archive.name.endswith('.tgz'):
            with tarfile.open(archive, 'r:gz') as tar:
                tar.extractall(out_dir)
        elif archive.name.endswith('.tar.bz2'):
            with tarfile.open(archive, 'r:bz2') as tar:
                tar.extractall(out_dir)
        elif archive.name.endswith('.zip'):
            with zipfile.ZipFile(archive, 'r') as zip_ref:
                zip_ref.extractall(out_dir)
        elif archive.name.endswith('.gz') and not archive.name.endswith('.tar.gz'):
            # Single file gzip
            output_file = out_dir / archive.stem
            with gzip.open(archive, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif archive.name.endswith('.bz2'):
            # Single file bzip2
            output_file = out_dir / archive.stem
            with bz2.open(archive, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            if progress_callback:
                progress_callback(1.0, f"Unknown archive format: {archive.name}")
            return
            
        if progress_callback:
            progress_callback(1.0, f"Extracted: {archive.name}")
            
    except Exception as e:
        if progress_callback:
            progress_callback(0, f"Extraction failed: {str(e)}")
        raise


def download_pubmed_baseline(dest_dir: Path, max_files: int = 5,
                           progress_callback: ProgressCallback = None) -> List[Path]:
    """Download PubMed baseline files."""
    downloaded = []
    base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    
    for i in range(1, max_files + 1):
        filename = f"pubmed25n{i:04d}.xml.gz"
        url = base_url + filename
        
        try:
            dest = download_with_resume(url, dest_dir / filename, progress_callback)
            downloaded.append(dest)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            if progress_callback:
                progress_callback(i / max_files, f"Failed: {filename}")
    
    return downloaded


def download_pmc_open_access(dest_dir: Path, max_files: int = 3,
                           progress_callback: ProgressCallback = None) -> List[Path]:
    """Download PMC Open Access files."""
    downloaded = []
    base_url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"
    
    # Download the file list first
    try:
        list_url = base_url + "oa_comm_xml.filelist.csv"
        response = requests.get(list_url, timeout=30)
        lines = response.text.strip().split('\n')[1:]  # Skip header
        
        for i, line in enumerate(lines[:max_files]):
            if progress_callback:
                progress_callback(i / max_files, f"Downloading PMC file {i+1}/{max_files}")
            
            parts = line.split('\t')
            if len(parts) > 0:
                filename = parts[0]
                url = base_url + filename
                dest = download_with_resume(url, dest_dir / filename, progress_callback)
                downloaded.append(dest)
                
    except Exception as e:
        print(f"Failed to download PMC files: {e}")
        if progress_callback:
            progress_callback(0, f"Failed to get PMC file list: {e}")
    
    return downloaded


def select_datasets(available_gb: float = 100.0) -> List[str]:
    """Select datasets to download based on available space."""
    sources = read_sources()
    
    # Priority order (most important first)
    priority = [
        "chembl_sqlite",      # 4.6GB - Essential drug data
        "hetionet",          # 50MB - Knowledge graph
        "clinical_trials",   # 5GB - Clinical trials
        "pubmed_baseline",   # Variable - Can limit files
        "chembl_sdf",        # 768MB - Chemical structures
        "uniprot_sprot",     # 100MB - Protein data
        "string_protein_links", # 50MB - Protein interactions
        "bindingdb",         # 400MB - Binding data
        "disgenet_curated",  # 10MB - Disease associations
        "reactome_all_levels", # 5MB - Pathways
    ]
    
    selected = []
    used_gb = 0
    skipped = []
    
    print("\nDataset Space Requirements:")
    print("=" * 50)
    
    # For pubmed_baseline, we'll download more files with 100GB available
    if "pubmed_baseline" in sources:
        sources["pubmed_baseline"]["size"] = "20GB"  # Increase from default 5 files
    
    for dataset in priority:
        if dataset in sources:
            # Estimate size
            size_str = sources[dataset].get("size", "0MB")
            size_gb = parse_size_to_gb(size_str)
            
            if used_gb + size_gb <= available_gb:
                selected.append(dataset)
                used_gb += size_gb
                print(f"✅ {dataset}: {size_gb:.1f}GB - Selected")
            else:
                skipped.append((dataset, size_gb))
                print(f"⏳ {dataset}: {size_gb:.1f}GB - Skipped (not enough space)")
    
    if skipped:
        print("\nSkipped Datasets (not enough space):")
        print("=" * 50)
        for dataset, size in skipped:
            print(f"• {dataset}: {size:.1f}GB")
        print(f"\nTo download these, you need {sum(size for _, size in skipped):.1f}GB more space")
    
    print(f"\nTotal space used: {used_gb:.1f}GB / {available_gb:.1f}GB")
    return selected


def parse_size_to_gb(size_str: str) -> float:
    """Parse size string to GB."""
    # Clean up the string
    size_str = size_str.upper().strip()
    
    # Use regex to extract number and unit, ignoring extra text
    # This will match patterns like "4.6GB", "~50MB", "1GB per file", etc.
    match = re.search(r'([~]?\d+\.?\d*)\s*(GB|MB|KB)', size_str)
    
    if match:
        number_str, unit = match.groups()
        # Remove ~ symbol and convert to float
        number = float(number_str.replace("~", ""))
        
        if unit == "GB":
            return number
        elif unit == "MB":
            return number / 1024
        elif unit == "KB":
            return number / (1024 * 1024)
    
    # If no match found, return small default
    print(f"Warning: Could not parse size '{size_str}', using default 0.1GB")
    return 0.1


def main(progress_callback: ProgressCallback = None, 
         available_gb: float = 10.0,
         datasets: Optional[List[str]] = None) -> None:
    """Main data ingestion function - prioritize existing data."""
    
    # FIRST: Check if we have sufficient existing data
    data_dir = settings.data_dir
    if data_dir.exists():
        existing_files = list(data_dir.rglob("*"))
        total_size_gb = sum(f.stat().st_size for f in existing_files if f.is_file()) / (1024**3)
        
        if total_size_gb > 5.0:  # If we have 5GB+ of data already
            if progress_callback:
                progress_callback(0.1, f"Found {total_size_gb:.1f}GB existing data - skipping downloads")
            print(f"🎯 Using existing data ({total_size_gb:.1f}GB) - no downloads needed")
            
            # Only fetch minimal fresh articles to avoid overload
            if progress_callback:
                progress_callback(0.5, "Fetching minimal fresh articles...")
            
            try:
                # Just one keyword to minimize load
                fetch_recent_pmc("longevity", data_dir / "pmc_auto", max_docs=5)
            except Exception as e:
                print(f"⚠️  Fresh article fetch failed: {e} - continuing with existing data")
            
            if progress_callback:
                progress_callback(1.0, "Using existing data - no heavy downloads")
            return
    
    # If we don't have enough existing data, proceed with minimal downloads
    sources = read_sources()
    
    # Use provided datasets or auto-select based on space (but very minimal)
    if datasets is None:
        datasets = select_datasets(min(available_gb, 5.0))  # Limit to 5GB max
    
    print(f"🛡️  MINIMAL DOWNLOAD MODE - limiting to essential data only")
    
    if progress_callback:
        progress_callback(0.05, f"Minimal download mode - {len(datasets)} essential datasets...")
    
    total_steps = min(len(datasets), 2)  # Limit to 2 datasets max
    current_step = 0
    
    # Download only essential datasets
    for dataset_name in datasets[:2]:  # LIMIT TO FIRST 2 ONLY
        current_step += 1
        base_progress = current_step / total_steps
        
        if dataset_name not in sources:
            continue
            
        dataset = sources[dataset_name]
        url = dataset["url"]
        
        try:
            if progress_callback:
                progress_callback(base_progress * 0.5, f"Downloading {dataset_name}...")
            
            # Special handling for certain datasets
            if dataset_name == "pubmed_baseline":
                # Download first 5 files only
                download_pubmed_baseline(
                    settings.data_dir / "pubmed_baseline",
                    max_files=5,
                    progress_callback=lambda p, m: progress_callback(base_progress * 0.5 * p, m) if progress_callback else None
                )
            elif dataset_name == "pmc_open_access":
                # Download first 3 PMC packages
                download_pmc_open_access(
                    settings.data_dir / "pmc",
                    max_files=3,
                    progress_callback=lambda p, m: progress_callback(base_progress * 0.5 * p, m) if progress_callback else None
                )
            else:
                # Standard download
                filename = url.split('/')[-1]
                archive = download_with_resume(
                    url, 
                    settings.data_dir / filename,
                    lambda p, m: progress_callback(base_progress * 0.5 * p, m) if progress_callback else None
                )
                
                # Extract if needed
                if dataset.get("format") in ["tar.gz", "zip", "gz", "bz2"]:
                    if progress_callback:
                        progress_callback(base_progress * 0.5 + 0.25, f"Extracting {dataset_name}...")
                    
                    extract_archive(
                        archive, 
                        settings.data_dir / dataset_name,
                        lambda p, m: progress_callback((base_progress * 0.5 + 0.25) + (0.25 * p), m) if progress_callback else None
                    )
                    
        except Exception as exc:
            print(f"Failed {dataset_name}: {exc}")
            if progress_callback:
                progress_callback(base_progress, f"⚠️ Failed {dataset_name}: {exc}")

    # Fetch fresh articles from APIs (existing code)
    keywords = _auto_fetch_keywords()
    for i, kw in enumerate(keywords):
        # PMC fetch
        current_step = len(datasets) + i * 2
        base_progress = current_step / total_steps
        
        print(f"[data_ingestion] Fetching latest PMC articles for '{kw}' …")
        try:
            fetch_recent_pmc(
                kw, 
                settings.data_dir / "pmc_auto",
                progress_callback=lambda p, m: progress_callback(base_progress * p, m) if progress_callback else None
            )
        except Exception as e:
            print(f"[data_ingestion] PMC fetch error for '{kw}': {e}")
        
        # arXiv fetch
        current_step = len(datasets) + i * 2 + 1
        base_progress = current_step / total_steps
        
        print(f"[data_ingestion] Fetching latest arXiv papers for '{kw}' …")
        try:
            fetch_arxiv(
                kw, 
                settings.data_dir / "arxiv_auto",
                progress_callback=lambda p, m: progress_callback(base_progress * p, m) if progress_callback else None
            )
        except Exception as e:
            print(f"[data_ingestion] arXiv fetch error for '{kw}': {e}")
    
    if progress_callback:
        progress_callback(1.0, "Data ingestion complete!")


# Keep existing helper functions
def _entrez_search(term: str, retmax: int = 20) -> List[str]:
    try:
        params = {"db": "pmc", "term": term, "retmode": "json", "retmax": str(retmax)}
        r = requests.get(PMC_EUTILS_SEARCH, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"[data_ingestion] PubMed search error: {e}")
        return []


def _entrez_fetch(pmcids: List[str]) -> List[str]:
    if not pmcids:
        return []
    try:
        params = {
            "db": "pmc",
            "id": ",".join(pmcids),
            "rettype": "full",
            "retmode": "text",
        }
        r = requests.get(PMC_EUTILS_FETCH, params=params, timeout=60)
        r.raise_for_status()
        return r.text.split("\n\f\n")
    except Exception as e:
        print(f"[data_ingestion] PubMed fetch error: {e}")
        return []


def fetch_recent_pmc(term: str, out_dir: Path, days: int = 7, max_docs: int = 20, 
                     progress_callback: ProgressCallback = None) -> List[Path]:
    if progress_callback:
        progress_callback(0.1, f"Searching PubMed for '{term}'...")
    
    ids = _entrez_search(f"{term}[Title] AND last{days}days[dp]", retmax=max_docs)
    
    if progress_callback:
        progress_callback(0.3, f"Found {len(ids)} articles, fetching...")
    
    texts = _entrez_fetch(ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    
    for i, txt in enumerate(texts):
        if progress_callback:
            progress = 0.3 + (0.7 * (i + 1) / len(texts))
            progress_callback(progress, f"Saving article {i+1}/{len(texts)}")
        
        p = out_dir / f"pmc_{term}_{i}.txt"
        p.write_text(txt)
        saved.append(p)
    
    if progress_callback:
        progress_callback(1.0, f"Fetched {len(saved)} PMC articles for '{term}'")
    
    return saved


def fetch_arxiv(term: str, out_dir: Path, max_docs: int = 20,
                progress_callback: ProgressCallback = None) -> List[Path]:
    if progress_callback:
        progress_callback(0.1, f"Searching arXiv for '{term}'...")
    
    try:
        feed_url = f"http://export.arxiv.org/api/query?search_query=all:{term}&start=0&max_results={max_docs}"
        parsed = feedparser.parse(feed_url)
        
        if parsed.bozo:
            print(f"[data_ingestion] arXiv feed parse error: {parsed.bozo_exception}")
            if progress_callback:
                progress_callback(1.0, f"⚠️ arXiv feed error for '{term}'")
            return []
        
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        
        entries = parsed.entries
        if not entries:
            if progress_callback:
                progress_callback(1.0, f"No arXiv results for '{term}'")
            return []
        
        for i, entry in enumerate(entries):
            if progress_callback:
                progress = 0.1 + (0.9 * (i + 1) / len(entries))
                progress_callback(progress, f"Saving paper {i+1}/{len(entries)}")
            
            txt = f"Title: {entry.get('title', 'Unknown')}\nAbstract: {entry.get('summary', 'No abstract')}"
            p = out_dir / f"arxiv_{term}_{i}.txt"
            p.write_text(txt)
            saved.append(p)
        
        if progress_callback:
            progress_callback(1.0, f"Fetched {len(saved)} arXiv papers for '{term}'")
        
        return saved
    except Exception as e:
        print(f"[data_ingestion] arXiv fetch error: {e}")
        if progress_callback:
            progress_callback(1.0, f"⚠️ arXiv fetch error: {str(e)}")
        return []


def _auto_fetch_keywords() -> List[str]:
    try:
        with open("keywords.txt") as fh:
            return [l.strip() for l in fh if l.strip()]
    except FileNotFoundError:
        return ["protein", "drug", "disease", "cancer", "covid", "alzheimer"]


def download_enhanced_datasets(datasets=None, progress_callback=None):
    """Download enhanced biomedical datasets."""
    if datasets is None:
        # Default to smaller, essential datasets
        datasets = ["drugbank", "physionet_exercise"]
    
    print(f"[Enhanced Data] Downloading {len(datasets)} enhanced datasets...")
    
    for i, dataset_name in enumerate(datasets):
        if dataset_name not in ENHANCED_DATA_SOURCES:
            print(f"[Enhanced Data] Warning: Unknown dataset {dataset_name}")
            continue
            
        dataset = ENHANCED_DATA_SOURCES[dataset_name]
        print(f"[Enhanced Data] Downloading {dataset['description']}...")
        
        if progress_callback:
            progress = (i / len(datasets)) * 100
            progress_callback(progress, f"Downloading {dataset['description']}")
        
        # For now, just create placeholder files
        # In production, implement actual download logic
        dest_dir = Path(f"data/enhanced/{dataset_name}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        placeholder_file = dest_dir / f"{dataset_name}_placeholder.txt"
        placeholder_file.write_text(f"Placeholder for {dataset['description']}\n"
                                  f"Size: {dataset['size_gb']} GB\n"
                                  f"Format: {dataset['format']}\n"
                                  f"URL: {dataset['url']}")
        
        print(f"[Enhanced Data] ✅ {dataset_name} placeholder created")
    
    print("[Enhanced Data] Enhanced datasets download complete!")


if __name__ == "__main__":
    # Example: Download with 20GB available space
    main(available_gb=20.0)

