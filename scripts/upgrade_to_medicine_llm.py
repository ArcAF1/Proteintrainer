#!/usr/bin/env python3
"""Upgrade to Medicine LLM 13B for better biomedical performance."""

import sys
import time
import requests
from pathlib import Path
from tqdm import tqdm

def download_with_progress(url: str, dest: Path, progress_callback=None):
    """Simple download function with progress tracking."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and get sizes
    local_size = dest.stat().st_size if dest.exists() else 0
    
    # Get remote file size
    response = requests.head(url, allow_redirects=True)
    remote_size = int(response.headers.get('content-length', 0))
    
    # Check if already complete
    if local_size == remote_size and local_size > 0:
        if progress_callback:
            progress_callback(100.0, "Already downloaded")
        return dest
    
    # Setup for resume download
    headers = {}
    mode = 'wb'
    if local_size > 0 and local_size < remote_size:
        headers['Range'] = f'bytes={local_size}-'
        mode = 'ab'
        print(f"Resuming download from {local_size / (1024**3):.1f} GB...")
    
    # Download with progress
    response = requests.get(url, headers=headers, stream=True, timeout=300)
    response.raise_for_status()
    
    chunk_size = 1024 * 1024  # 1MB chunks
    downloaded = local_size
    
    with open(dest, mode) as f:
        with tqdm(total=remote_size, initial=downloaded, unit='B', unit_scale=True, desc=dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))
                    
                    if progress_callback:
                        progress = (downloaded / remote_size) * 100
                        progress_callback(progress, f"Downloaded {downloaded / (1024**3):.1f} GB")
    
    return dest

def main():
    """Download and setup Medicine LLM 13B."""
    print("🧬 Upgrading to Medicine LLM 13B")
    print("=" * 50)
    print()
    
    # Medicine LLM configuration
    medicine_llm = {
        "url": "https://huggingface.co/TheBloke/medicine-LLM-13B-GGUF/resolve/main/medicine-llm-13b.Q4_K_M.gguf",
        "filename": "medicine-llm-13b.Q4_K_M.gguf",
        "size_gb": 7.9,
        "description": "Medicine LLM 13B - Specialized biomedical model"
    }
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / medicine_llm["filename"]
    
    # Check if already downloaded
    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        if size_gb > 7.0:  # At least 7GB means complete
            print(f"✅ Medicine LLM already downloaded ({size_gb:.1f} GB)")
            print(f"   Located at: {model_path}")
            print()
            print("🎯 Your system will automatically use this model on next startup!")
            return
    
    print(f"📥 Downloading {medicine_llm['description']}")
    print(f"   Size: {medicine_llm['size_gb']} GB")
    print(f"   This will take 5-15 minutes depending on your internet speed...")
    print()
    
    # Progress callback
    def progress_callback(percentage, message):
        if percentage < 100:
            print(f"\r🔄 Progress: {percentage:.1f}% - {message}", end="", flush=True)
    
    try:
        # Download the model
        download_with_progress(
            url=medicine_llm["url"],
            dest=model_path,
            progress_callback=progress_callback
        )
        
        print()  # New line after progress
        print(f"✅ Medicine LLM downloaded successfully!")
        print(f"   Size: {model_path.stat().st_size / (1024**3):.1f} GB")
        print()
        
        # Verify download
        if model_path.stat().st_size > 7 * (1024**3):  # At least 7GB
            print("🎯 **Upgrade Complete!**")
            print()
            print("**What's improved:**")
            print("• 📚 Trained specifically on medical literature")
            print("• 🧠 Better understanding of biomedical concepts")
            print("• 💊 Enhanced drug interaction knowledge")
            print("• 🔬 Improved clinical reasoning")
            print("• 📊 More accurate medical terminology")
            print()
            print("**Your system will automatically use Medicine LLM on next startup!**")
            print()
            print("To test: Run ./start.command and ask a medical question")
            
        else:
            print("❌ Download may be incomplete. Please try again.")
            
    except KeyboardInterrupt:
        print("\n\n❌ Download cancelled by user")
        if model_path.exists():
            model_path.unlink()  # Remove partial download
            
    except Exception as e:
        print(f"\n\n❌ Download failed: {e}")
        if model_path.exists():
            model_path.unlink()  # Remove partial download

if __name__ == "__main__":
    main() 