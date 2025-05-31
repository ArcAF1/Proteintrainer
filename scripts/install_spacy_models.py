#!/usr/bin/env python3
"""Install SpaCy models for biomedical NER."""

import subprocess
import sys

def install_models():
    """Install SpaCy models."""
    models = [
        # General scientific model
        ("en_core_sci_sm", "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"),
        # Biomedical NER model
        ("en_ner_bc5cdr_md", "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"),
    ]
    
    for model_name, url in models:
        print(f"\nInstalling {model_name}...")
        try:
            # Try direct pip install from URL
            subprocess.run([
                sys.executable, "-m", "pip", "install", url
            ], check=True)
            print(f"✓ Successfully installed {model_name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {model_name}: {e}")
            print("  Trying alternative method...")
            
            # Try using spacy download
            try:
                subprocess.run([
                    sys.executable, "-m", "spacy", "download", model_name
                ], check=True)
                print(f"✓ Successfully installed {model_name} (alternative method)")
            except subprocess.CalledProcessError:
                print(f"✗ Could not install {model_name}")
                print(f"  You may need to manually download from: {url}")

if __name__ == "__main__":
    install_models() 