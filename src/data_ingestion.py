"""Download and extract datasets.
Usage example:
    python src/data_ingestion.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict

import requests
from tqdm import tqdm

from .config import settings


def read_sources() -> Dict[str, str]:
    with open(Path(__file__).with_name("data_sources.json")) as fh:
        return json.load(fh)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in tqdm(r.iter_content(chunk_size=8192), desc=f"Downloading {dest.name}"):
                if chunk:
                    fh.write(chunk)
    return dest


def extract_archive(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.unpack_archive(str(archive), str(out_dir))
    except shutil.ReadError:
        print(f"Unsupported archive format for {archive.name}")


def main() -> None:
    sources = read_sources()
    for name, url in sources.items():
        try:
            archive = download(url, settings.data_dir / f"{name}.zip")
            extract_archive(archive, settings.data_dir / name)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed {name}: {exc}")
            print("Check the URL or download manually.")


if __name__ == "__main__":
    main()
