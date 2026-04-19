"""MidiCaps dataset genre loader.

Downloads and parses the MidiCaps dataset (amaai-lab/MidiCaps) from
HuggingFace, filtering MIDI files by genre tags in metadata to produce
per-genre lists of validated MIDI file paths for cross-dataset validation.

MidiCaps metadata CSV contains columns including 'genre' with tags like
"Rock", "Jazz", "Electronic", "Country/Folk", etc.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger


# Genre tag mapping: MidiCaps tag variants → canonical genre names
MIDICAPS_GENRE_MAP: Dict[str, str] = {
    # Rock
    "rock": "rock",
    "rock/pop": "rock",
    "hard rock": "rock",
    "soft rock": "rock",
    "indie rock": "rock",
    "alternative rock": "rock",
    "punk rock": "rock",
    "classic rock": "rock",
    # Jazz
    "jazz": "jazz",
    "smooth jazz": "jazz",
    "jazz fusion": "jazz",
    "bebop": "jazz",
    "swing": "jazz",
    "bossa nova": "jazz",
    "latin jazz": "jazz",
    # Electronic
    "electronic": "electronic",
    "electro/dance": "electronic",
    "dance": "electronic",
    "techno": "electronic",
    "house": "electronic",
    "trance": "electronic",
    "edm": "electronic",
    "ambient": "electronic",
    "synthwave": "electronic",
    "drum and bass": "electronic",
    "dubstep": "electronic",
    "electro": "electronic",
    # Country
    "country": "country",
    "country/folk": "country",
    "folk/country": "country",
    "bluegrass": "country",
    "americana": "country",
    "folk": "country",
}


class MidiCapsGenreLoader:
    """Load and filter MidiCaps MIDI files by genre tags."""

    def __init__(self, config: Dict):
        self.config = config
        cv_cfg = config.get("cross_validation", {}).get("midicaps", {})

        self.hf_repo: str = cv_cfg.get("hf_repo", "amaai-lab/MidiCaps")
        self.data_dir = Path(cv_cfg.get("data_dir", "data/raw/midicaps"))
        self.metadata_file = Path(cv_cfg.get("metadata_file", "data/raw/midicaps/metadata.csv"))
        self.genres: List[str] = [
            g.lower() for g in cv_cfg.get("genres", ["rock", "jazz", "electronic", "country"])
        ]
        self.max_per_genre: int = int(cv_cfg.get("max_per_genre", 120))
        self.seed: int = int(cv_cfg.get("seed", 42))

        # Allow custom genre mapping overrides from config
        extra_map = cv_cfg.get("genre_map", {})
        self.genre_map = {**MIDICAPS_GENRE_MAP, **{k.lower(): v.lower() for k, v in extra_map.items()}}

        logger.info(
            f"MidiCapsGenreLoader initialised: genres={self.genres}, "
            f"max_per_genre={self.max_per_genre}, data_dir={self.data_dir}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_data(self) -> bool:
        """Download MidiCaps dataset if not already present.

        Uses huggingface_hub if available, otherwise falls back to
        direct URL download.

        Returns True when data directory contains MIDI files.
        """
        if self._data_ready():
            logger.info(f"MidiCaps data already present at {self.data_dir}")
            return True

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Try huggingface_hub first
        try:
            return self._download_via_hf_hub()
        except Exception as exc:
            logger.warning(f"huggingface_hub download failed: {exc}")

        # Fallback: direct git clone (shallow)
        try:
            return self._download_via_git()
        except Exception as exc:
            logger.error(f"Git clone also failed: {exc}")
            return False

    def load_genre_files(self) -> Dict[str, List[Path]]:
        """Return ``{genre: [midi_path, …]}`` for configured genres.

        Parses metadata CSV/JSON, maps genre tags to canonical names,
        validates MIDI files, and returns up to max_per_genre per genre.
        """
        tag_to_files = self._parse_metadata()

        # Aggregate files per canonical genre
        genre_to_files: Dict[str, List[Path]] = defaultdict(list)
        for tag, midi_paths in tag_to_files.items():
            tag_lower = tag.lower().strip()
            canonical = self.genre_map.get(tag_lower)
            if canonical and canonical in self.genres:
                genre_to_files[canonical].extend(midi_paths)

        # Log all unique tags for debugging
        logger.info(f"MidiCaps unique genre tags found: {sorted(tag_to_files.keys())}")

        rng = np.random.default_rng(self.seed)
        result: Dict[str, List[Path]] = {}

        for genre in self.genres:
            candidates = genre_to_files.get(genre, [])
            # Deduplicate
            candidates = list(set(candidates))
            rng.shuffle(candidates)  # type: ignore[arg-type]

            valid_paths: List[Path] = []
            skipped = 0
            for path in candidates:
                if len(valid_paths) >= self.max_per_genre:
                    break
                if self._validate_midi(path):
                    valid_paths.append(path)
                else:
                    skipped += 1

            logger.info(
                f"MidiCaps genre '{genre}': {len(valid_paths)} valid MIDI files "
                f"(from {len(candidates)} candidates, skipped {skipped})"
            )
            result[genre] = valid_paths

        return result

    def populate_raw_datasets(self) -> Dict[str, int]:
        """Copy genre MIDI subsets into ``data/raw/midicaps_<genre>/`` directories.

        Makes them accessible through the standard DatasetManager interface.
        Returns dict ``{dataset_name: n_files_copied}``.
        """
        genre_files = self.load_genre_files()
        raw_dir = Path(self.config["data"]["raw_data_dir"])
        counts: Dict[str, int] = {}

        for genre, paths in genre_files.items():
            dataset_name = f"midicaps_{genre}"
            dest_dir = raw_dir / dataset_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            copied = 0
            for src in paths:
                dst = dest_dir / src.name
                # Avoid name collisions by adding hash prefix
                if dst.exists():
                    import hashlib
                    h = hashlib.md5(str(src).encode()).hexdigest()[:8]
                    dst = dest_dir / f"{h}_{src.name}"
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    copied += 1
                    continue
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except Exception as exc:
                    logger.warning(f"Failed to copy {src} → {dst}: {exc}")

            logger.info(f"Populated {dataset_name}: {copied} files in {dest_dir}")
            counts[dataset_name] = copied

        return counts

    def discover_genre_tags(self) -> Dict[str, int]:
        """Return all unique genre tags and their counts from metadata.

        Useful for exploring MidiCaps tag vocabulary before running experiments.
        """
        tag_to_files = self._parse_metadata()
        return {tag: len(files) for tag, files in sorted(tag_to_files.items())}

    # ------------------------------------------------------------------
    # Metadata parsing
    # ------------------------------------------------------------------

    def _parse_metadata(self) -> Dict[str, List[Path]]:
        """Parse MidiCaps metadata → ``{genre_tag: [midi_path, …]}``.

        Supports JSONL format (primary, as used by MidiCaps train.json),
        CSV, and standard JSON fallback.
        """
        result: Dict[str, List[Path]] = defaultdict(list)

        # Try JSONL first (MidiCaps native format)
        jsonl_candidates = list(self.data_dir.rglob("*.json"))
        if self.metadata_file.exists() and str(self.metadata_file).endswith(".json"):
            jsonl_candidates.insert(0, self.metadata_file)
        # Deduplicate
        seen = set()
        unique_jsonl = []
        for p in jsonl_candidates:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                unique_jsonl.append(p)

        for json_path in unique_jsonl:
            parsed = self._parse_jsonl_metadata(json_path)
            if parsed:
                for tag, paths in parsed.items():
                    result[tag].extend(paths)
                return dict(result)

        # Try CSV
        csv_candidates = list(self.data_dir.rglob("*.csv"))
        if self.metadata_file.exists() and str(self.metadata_file).endswith(".csv"):
            csv_candidates.insert(0, self.metadata_file)

        for csv_path in csv_candidates:
            parsed = self._parse_csv_metadata(csv_path)
            if parsed:
                for tag, paths in parsed.items():
                    result[tag].extend(paths)
                return dict(result)

        # Fallback: scan directory for MIDI files without genre filtering
        logger.warning(
            "No metadata file found — falling back to directory-based genre detection"
        )
        return self._parse_directory_structure()

    def _parse_jsonl_metadata(self, jsonl_path: Path) -> Optional[Dict[str, List[Path]]]:
        """Parse JSONL metadata (MidiCaps train.json format).

        Each line is a JSON object with keys: location, genre (list), etc.
        Genre is a list of strings, e.g. ["electronic", "classical"].
        Location is relative path, e.g. "lmd_full/1/abc123.mid".
        """
        result: Dict[str, List[Path]] = defaultdict(list)
        count = 0
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(item, dict):
                        continue

                    # Extract genre tags
                    genre_raw = item.get("genre") or item.get("genres") or []
                    if isinstance(genre_raw, str):
                        tags = [genre_raw.strip()]
                    elif isinstance(genre_raw, list):
                        tags = [str(t).strip() for t in genre_raw]
                    else:
                        continue

                    # Extract file location
                    location = item.get("location") or item.get("path") or item.get("filename")
                    if not location:
                        continue

                    # Resolve MIDI path — try multiple strategies
                    midi_path = self._resolve_midi_path(str(location))
                    if midi_path is None:
                        continue

                    for tag in tags:
                        if tag:
                            result[tag].append(midi_path)
                    count += 1

            if result:
                logger.info(
                    f"Parsed MidiCaps JSONL: {jsonl_path.name} "
                    f"({count} entries with valid paths, "
                    f"{sum(len(v) for v in result.values())} genre-file pairs)"
                )
            return dict(result) if result else None

        except Exception as exc:
            logger.debug(f"Could not parse JSONL {jsonl_path}: {exc}")
            return None

    def _resolve_midi_path(self, location: str) -> Optional[Path]:
        """Resolve a MidiCaps location string to an actual file path.

        MidiCaps uses paths like "lmd_full/1/abc123def456.mid".
        We try:
        1. Direct path under data_dir
        2. Just the filename under data_dir (recursive search)
        3. The path with lmd_full prefix stripped
        """
        # Strategy 1: direct path under data_dir
        candidate = self.data_dir / location
        if candidate.exists():
            return candidate

        # Strategy 2: strip prefix and try
        # e.g. "lmd_full/1/abc.mid" -> just search for "abc.mid"
        fname = Path(location).name
        candidate = self.data_dir / fname
        if candidate.exists():
            return candidate

        # Strategy 3: search in subdirectories
        # Use the hash prefix structure: lmd_full/<prefix>/<hash>.mid
        parts = location.split("/")
        if len(parts) >= 2:
            # Try data_dir/lmd_full/prefix/file
            for subdir_name in ("lmd_full", "midicaps", "midi"):
                candidate = self.data_dir / subdir_name
                if candidate.is_dir():
                    full = self.data_dir / location
                    if full.exists():
                        return full
                    # try just last 2 parts
                    partial = candidate / "/".join(parts[-2:])
                    if partial.exists():
                        return partial

        return None

    def _parse_csv_metadata(self, csv_path: Path) -> Optional[Dict[str, List[Path]]]:
        """Parse a CSV metadata file with genre column."""
        result: Dict[str, List[Path]] = defaultdict(list)
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return None

                # Find genre column (various naming conventions)
                genre_col = None
                file_col = None
                for col in reader.fieldnames:
                    col_lower = col.lower().strip()
                    if col_lower in ("genre", "genres", "genre_tag", "genre_tags", "tag", "tags"):
                        genre_col = col
                    if col_lower in ("file", "filename", "file_name", "path", "midi_path",
                                     "midi_filename", "location", "fname"):
                        file_col = col

                if not genre_col:
                    return None

                for row in reader:
                    genre_raw = row.get(genre_col, "").strip()
                    if not genre_raw:
                        continue

                    # Handle comma/semicolon-separated multi-genre tags
                    tags = [t.strip() for t in genre_raw.replace(";", ",").split(",") if t.strip()]

                    # Resolve MIDI file path
                    midi_path = None
                    if file_col and row.get(file_col):
                        candidate = self.data_dir / row[file_col].strip()
                        if candidate.exists():
                            midi_path = candidate
                        else:
                            # Try searching
                            fname = Path(row[file_col].strip()).name
                            found = list(self.data_dir.rglob(fname))
                            if found:
                                midi_path = found[0]

                    if midi_path is None:
                        continue

                    for tag in tags:
                        result[tag].append(midi_path)

            if result:
                logger.info(f"Parsed MidiCaps CSV metadata: {csv_path} ({sum(len(v) for v in result.values())} entries)")
            return dict(result) if result else None

        except Exception as exc:
            logger.debug(f"Could not parse CSV {csv_path}: {exc}")
            return None

    def _parse_json_metadata(self, json_path: Path) -> Optional[Dict[str, List[Path]]]:
        """Parse a JSON metadata file with genre info."""
        result: Dict[str, List[Path]] = defaultdict(list)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else data.get("data", data.get("items", []))
            if not isinstance(items, list):
                return None

            for item in items:
                if not isinstance(item, dict):
                    continue

                genre_raw = item.get("genre") or item.get("genres") or item.get("tag") or ""
                if isinstance(genre_raw, list):
                    tags = [str(t).strip() for t in genre_raw]
                else:
                    tags = [t.strip() for t in str(genre_raw).replace(";", ",").split(",") if t.strip()]

                file_key = None
                for k in ("file", "filename", "path", "midi_path", "fname", "location"):
                    if k in item:
                        file_key = k
                        break

                if not file_key:
                    continue

                candidate = self.data_dir / str(item[file_key]).strip()
                if not candidate.exists():
                    fname = Path(str(item[file_key]).strip()).name
                    found = list(self.data_dir.rglob(fname))
                    if found:
                        candidate = found[0]
                    else:
                        continue

                for tag in tags:
                    result[tag].append(candidate)

            if result:
                logger.info(f"Parsed MidiCaps JSON metadata: {json_path} ({sum(len(v) for v in result.values())} entries)")
            return dict(result) if result else None

        except Exception as exc:
            logger.debug(f"Could not parse JSON {json_path}: {exc}")
            return None

    def _parse_directory_structure(self) -> Dict[str, List[Path]]:
        """Fallback: infer genres from subdirectory names."""
        result: Dict[str, List[Path]] = defaultdict(list)
        for subdir in sorted(self.data_dir.iterdir()):
            if subdir.is_dir():
                genre_name = subdir.name.lower()
                midi_files = sorted(subdir.rglob("*.mid")) + sorted(subdir.rglob("*.midi"))
                if midi_files:
                    result[genre_name] = midi_files
                    logger.info(f"MidiCaps directory genre '{genre_name}': {len(midi_files)} files")
        return dict(result)

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _data_ready(self) -> bool:
        """Check if data directory has MIDI files."""
        if not self.data_dir.exists():
            return False
        midi_count = len(list(self.data_dir.rglob("*.mid"))) + len(list(self.data_dir.rglob("*.midi")))
        return midi_count > 0

    def _download_via_hf_hub(self) -> bool:
        """Download dataset using huggingface_hub."""
        from huggingface_hub import snapshot_download

        logger.info(f"Downloading MidiCaps from HuggingFace: {self.hf_repo}")
        snapshot_download(
            repo_id=self.hf_repo,
            repo_type="dataset",
            local_dir=str(self.data_dir),
            ignore_patterns=["*.git*", "*.md"],
        )
        ok = self._data_ready()
        if ok:
            logger.info(f"MidiCaps downloaded successfully to {self.data_dir}")
        return ok

    def _download_via_git(self) -> bool:
        """Fallback: shallow git clone."""
        import subprocess

        url = f"https://huggingface.co/datasets/{self.hf_repo}"
        logger.info(f"Cloning MidiCaps via git: {url}")
        # Remove empty dir for git clone
        if self.data_dir.exists() and not any(self.data_dir.iterdir()):
            shutil.rmtree(self.data_dir)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(self.data_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error(f"Git clone failed: {result.stderr.strip()}")
            return False
        return self._data_ready()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_midi(path: Path) -> bool:
        """Quick check: file exists, has .mid/.midi extension, is non-empty."""
        if not path.exists():
            return False
        if path.suffix.lower() not in {".mid", ".midi"}:
            return False
        try:
            return path.stat().st_size > 0
        except OSError:
            return False


