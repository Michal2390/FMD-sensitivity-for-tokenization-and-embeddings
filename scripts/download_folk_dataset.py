#!/usr/bin/env python3
"""Download the Nottingham MIDI dataset for the folk category.

The Nottingham dataset contains ~1200 folk tunes in MIDI format.
Source: https://github.com/jukedeck/nottingham-dataset
"""

import os
import sys
import zipfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def download_nottingham(output_dir: str = "data/raw/folk"):
    """Download and extract Nottingham MIDI dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if already has MIDI files
    existing = list(output_path.glob("**/*.mid")) + list(output_path.glob("**/*.midi"))
    if len(existing) >= 50:
        print(f"Folk dataset already exists with {len(existing)} MIDI files at {output_path}")
        return

    try:
        import urllib.request
    except ImportError:
        print("ERROR: urllib not available")
        return

    # Download from GitHub archive
    url = "https://github.com/jukedeck/nottingham-dataset/archive/refs/heads/master.zip"
    print(f"Downloading Nottingham dataset from {url}...")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(url, tmp_path)
        print(f"Downloaded to {tmp_path}")

        # Extract MIDI files
        with zipfile.ZipFile(tmp_path, "r") as zf:
            midi_count = 0
            for name in zf.namelist():
                if name.lower().endswith((".mid", ".midi")):
                    # Extract to flat directory
                    filename = Path(name).name
                    target = output_path / filename
                    with zf.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    midi_count += 1

        print(f"Extracted {midi_count} MIDI files to {output_path}")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    download_nottingham()

