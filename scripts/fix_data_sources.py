#!/usr/bin/env python3
"""Populate src/data_sources.json med riktiga URL:er.

Interaktivt läge:
    python scripts/fix_data_sources.py

Auto-läge (skriver standardlänkar):
    python scripts/fix_data_sources.py --auto
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

DEFAULT_LINKS = {
    "pubmed_central": "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/non_comm_use.A-B.xml.tar.gz",
    "drugbank": "https://github.com/biothings/drugbank-downloader/releases/download/v1.0/drugbank_dump.xml.zip",
    "clinical_trials": "https://clinicaltrials.gov/AllPublicXML.zip",
}

DST = Path(__file__).resolve().parents[1] / "src" / "data_sources.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="skriv default-länkar utan frågor")
    args = ap.parse_args()

    if not DST.exists():
        raise FileNotFoundError(DST)

    shutil.copy(DST, DST.with_suffix(".json.bak"))  # backup

    links = DEFAULT_LINKS if args.auto else {
        k: (input(f"URL för {k} [{v}]: ") or v).strip() for k, v in DEFAULT_LINKS.items()
    }

    DST.write_text(json.dumps(links, indent=2))
    print(f"✅ Uppdaterade {DST} med {len(links)} länkar")


if __name__ == "__main__":
    main() 