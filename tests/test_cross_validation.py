"""Tests for cross-dataset validation modules."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config():
    """Minimal config for testing loaders."""
    return {
        "data": {
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "embeddings_dir": "data/embeddings",
            "datasets": [],
        },
        "lakh": {
            "genres": ["rock", "jazz", "electronic", "country"],
            "max_per_genre": 50,
            "seed": 42,
            "lmd_matched_dir": "data/raw/lakh/lmd_matched",
            "tagtraum_file": "data/raw/lakh/msd_tagtraum_cd2.cls",
        },
        "cross_validation": {
            "tagtraum_cd1": {
                "url": "https://www.tagtraum.com/genres/msd_tagtraum_cd1.cls.zip",
                "file": "data/raw/lakh/msd_tagtraum_cd1.cls",
                "max_per_genre": 500,
            },
            "midicaps": {
                "hf_repo": "amaai-lab/MidiCaps",
                "data_dir": "data/raw/midicaps",
                "metadata_file": "data/raw/midicaps/metadata.csv",
                "genres": ["rock", "jazz", "electronic", "country"],
                "max_per_genre": 120,
                "seed": 42,
            },
            "subsample_size": 100,
            "n_repeats": 10,
            "seed": 42,
        },
    }


# ──────────────────────────────────────────────────────────────────────
# LakhGenreLoader CD1 tests
# ──────────────────────────────────────────────────────────────────────

def test_lakh_loader_cd1_init(base_config):
    """Test LakhGenreLoader initialises correctly with CD1."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from data.lakh_genre_loader import LakhGenreLoader

    loader = LakhGenreLoader(base_config, tagtraum_version="cd1")
    assert loader.tagtraum_version == "cd1"
    assert "cd1" in str(loader.tagtraum_file)
    assert loader._dataset_prefix == "lakh_cd1"


def test_lakh_loader_cd2_init(base_config):
    """Test LakhGenreLoader defaults to CD2."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from data.lakh_genre_loader import LakhGenreLoader

    loader = LakhGenreLoader(base_config, tagtraum_version="cd2")
    assert loader.tagtraum_version == "cd2"
    assert "cd2" in str(loader.tagtraum_file)
    assert loader._dataset_prefix == "lakh"


def test_lakh_loader_backward_compatible(base_config):
    """Test LakhGenreLoader works without tagtraum_version arg (backward compat)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from data.lakh_genre_loader import LakhGenreLoader

    loader = LakhGenreLoader(base_config)
    assert loader.tagtraum_version == "cd2"


# ──────────────────────────────────────────────────────────────────────
# MidiCapsGenreLoader tests
# ──────────────────────────────────────────────────────────────────────

def test_midicaps_loader_init(base_config):
    """Test MidiCapsGenreLoader initialises correctly."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from data.midicaps_loader import MidiCapsGenreLoader

    loader = MidiCapsGenreLoader(base_config)
    assert loader.max_per_genre == 120
    assert "rock" in loader.genres
    assert "jazz" in loader.genres


def test_midicaps_genre_map(base_config):
    """Test genre mapping covers expected tags."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from data.midicaps_loader import MIDICAPS_GENRE_MAP

    assert MIDICAPS_GENRE_MAP["rock"] == "rock"
    assert MIDICAPS_GENRE_MAP["jazz"] == "jazz"
    assert MIDICAPS_GENRE_MAP["electronic"] == "electronic"
    assert MIDICAPS_GENRE_MAP["country"] == "country"
    assert MIDICAPS_GENRE_MAP["electro/dance"] == "electronic"
    assert MIDICAPS_GENRE_MAP["country/folk"] == "country"
    assert MIDICAPS_GENRE_MAP["bebop"] == "jazz"


def test_midicaps_csv_parsing(base_config, tmp_path):
    """Test CSV metadata parsing."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from data.midicaps_loader import MidiCapsGenreLoader

    # Create fake CSV metadata
    csv_path = tmp_path / "metadata.csv"
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()

    # Create fake MIDI files
    for i in range(5):
        (midi_dir / f"rock_{i}.mid").write_bytes(b"\x00" * 100)
    for i in range(3):
        (midi_dir / f"jazz_{i}.mid").write_bytes(b"\x00" * 100)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "genre"])
        writer.writeheader()
        for i in range(5):
            writer.writerow({"filename": f"midi/rock_{i}.mid", "genre": "Rock"})
        for i in range(3):
            writer.writerow({"filename": f"midi/jazz_{i}.mid", "genre": "Jazz"})

    base_config["cross_validation"]["midicaps"]["data_dir"] = str(tmp_path)
    base_config["cross_validation"]["midicaps"]["metadata_file"] = str(csv_path)

    loader = MidiCapsGenreLoader(base_config)
    files = loader.load_genre_files()
    assert "rock" in files
    assert len(files["rock"]) == 5
    assert "jazz" in files
    assert len(files["jazz"]) == 3
