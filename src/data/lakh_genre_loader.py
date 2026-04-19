"""Lakh MIDI Dataset genre loader with Tagtraum CD2 annotations.

Downloads and parses the Lakh MIDI Dataset (matched subset) together with
Tagtraum genre annotations from the Million Song Dataset, producing
per-genre lists of validated MIDI file paths.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


class LakhGenreLoader:
    """Load and filter Lakh MIDI files by genre using Tagtraum labels.

    Supports both Tagtraum CD2 (default) and CD1 annotations for
    cross-annotation validation.  Set ``tagtraum_version="cd1"`` or
    configure via ``cross_validation.tagtraum_cd1`` in config.
    """

    # Class-level URL defaults for each Tagtraum version
    _TAGTRAUM_URLS = {
        "cd2": "https://www.tagtraum.com/genres/msd_tagtraum_cd2.cls.zip",
        "cd1": "https://www.tagtraum.com/genres/msd_tagtraum_cd1.cls.zip",
    }

    def __init__(self, config: Dict, tagtraum_version: str = "cd2"):
        self.config = config
        self.tagtraum_version = tagtraum_version.lower()
        lakh_cfg = config.get("lakh", {})

        self.lmd_matched_url: str = lakh_cfg.get(
            "lmd_matched_url",
            "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz",
        )

        # Resolve Tagtraum URL and file path based on version
        if self.tagtraum_version == "cd1":
            cv_cd1 = config.get("cross_validation", {}).get("tagtraum_cd1", {})
            self.tagtraum_url: str = cv_cd1.get(
                "url", self._TAGTRAUM_URLS["cd1"]
            )
            self.tagtraum_file = Path(cv_cd1.get(
                "file", "data/raw/lakh/msd_tagtraum_cd1.cls"
            ))
            self.max_per_genre: int = int(cv_cd1.get(
                "max_per_genre", lakh_cfg.get("max_per_genre", 500)
            ))
            self._dataset_prefix = "lakh_cd1"
        else:
            self.tagtraum_url: str = lakh_cfg.get(
                "tagtraum_url",
                self._TAGTRAUM_URLS["cd2"],
            )
            self.tagtraum_file = Path(lakh_cfg.get(
                "tagtraum_file", "data/raw/lakh/msd_tagtraum_cd2.cls"
            ))
            self.max_per_genre: int = int(lakh_cfg.get("max_per_genre", 500))
            self._dataset_prefix = "lakh"

        self.lmd_matched_dir = Path(lakh_cfg.get("lmd_matched_dir", "data/raw/lakh/lmd_matched"))
        self.genres: List[str] = [g.lower() for g in lakh_cfg.get("genres", ["rock", "classical"])]
        self.seed: int = int(lakh_cfg.get("seed", 42))

        logger.info(
            f"LakhGenreLoader initialised: version={self.tagtraum_version}, "
            f"genres={self.genres}, max_per_genre={self.max_per_genre}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_data(self) -> bool:
        """Download Lakh matched + Tagtraum file if not already present.

        Returns True when both the MIDI directory and annotations file exist
        (or have been downloaded successfully).
        """
        ok = True
        if not self._tagtraum_ready():
            ok = ok and self._download_tagtraum()
        if not self._lmd_ready():
            ok = ok and self._download_lmd()
        return ok

    def load_genre_files(self) -> Dict[str, List[Path]]:
        """Return ``{genre: [midi_path, …]}`` for configured genres.

        Files are validated (non-empty ``.mid``/``.midi``).  Up to
        ``max_per_genre`` files are returned per genre (shuffled
        deterministically with ``seed``).
        """
        track_to_genre = self._parse_tagtraum()
        genre_to_tracks: Dict[str, List[str]] = defaultdict(list)
        for track_id, genre in track_to_genre.items():
            genre_lower = genre.lower()
            if genre_lower in self.genres:
                genre_to_tracks[genre_lower].append(track_id)

        for g in self.genres:
            logger.info(f"Tagtraum tracks labelled '{g}': {len(genre_to_tracks.get(g, []))}")

        result: Dict[str, List[Path]] = {}
        import numpy as np

        rng = np.random.default_rng(self.seed)

        for genre in self.genres:
            candidates = genre_to_tracks.get(genre, [])
            rng.shuffle(candidates)  # type: ignore[arg-type]
            valid_paths: List[Path] = []
            skipped = 0
            for track_id in candidates:
                if len(valid_paths) >= self.max_per_genre:
                    break
                midi_path = self._track_id_to_midi_path(track_id)
                if midi_path is None:
                    skipped += 1
                    continue
                if not self._validate_midi(midi_path):
                    skipped += 1
                    continue
                valid_paths.append(midi_path)

            logger.info(
                f"Genre '{genre}': {len(valid_paths)} valid MIDI files "
                f"(skipped {skipped})"
            )
            result[genre] = valid_paths

        return result

    # ------------------------------------------------------------------
    # Tagtraum parsing
    # ------------------------------------------------------------------

    def _parse_tagtraum(self) -> Dict[str, str]:
        """Parse ``msd_tagtraum_cd2.cls`` → ``{track_id: genre}``.

        File format (TSV, comment lines start with ``#``):
            TRXXXXX<TAB>genre
        Some tracks have multiple genre lines; we keep the first.
        """
        mapping: Dict[str, str] = {}
        path = self.tagtraum_file
        if not path.exists():
            logger.error(f"Tagtraum file not found: {path}")
            return mapping

        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                track_id = parts[0].strip()
                genre = parts[1].strip()
                if track_id and genre and track_id not in mapping:
                    mapping[track_id] = genre

        logger.info(f"Parsed {len(mapping)} track→genre entries from Tagtraum")
        return mapping

    # ------------------------------------------------------------------
    # Track ID → MIDI path mapping
    # ------------------------------------------------------------------

    def _track_id_to_midi_path(self, track_id: str) -> Optional[Path]:
        """Map MSD track ID to a MIDI path inside ``lmd_matched``.

        Lakh matched stores files as::

            lmd_matched/<char2>/<char3>/<char4>/<track_id>/<hash>.mid

        e.g. ``lmd_matched/R/R/U/TRRRUTV12903CEA11B/abc123.mid``

        where char2/3/4 are the 2nd, 3rd, 4th characters of the track ID
        (skipping the leading 'TR').
        """
        if len(track_id) < 5:
            return None

        # Characters at index 2, 3, 4 form the directory path
        track_dir = (
            self.lmd_matched_dir
            / track_id[2]
            / track_id[3]
            / track_id[4]
            / track_id
        )

        if track_dir.is_dir():
            midis = sorted(track_dir.glob("*.mid")) + sorted(track_dir.glob("*.midi"))
            if midis:
                return midis[0]

        return None

    # ------------------------------------------------------------------
    # Validation helpers
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

    def _tagtraum_ready(self) -> bool:
        return self.tagtraum_file.exists() and self.tagtraum_file.stat().st_size > 0

    def _lmd_ready(self) -> bool:
        if not self.lmd_matched_dir.exists():
            return False
        # Quick heuristic: at least some subdirectories
        return any(self.lmd_matched_dir.iterdir())

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _download_tagtraum(self) -> bool:
        """Download Tagtraum CD2 genre annotation file (supports .zip)."""
        self.tagtraum_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading Tagtraum annotations: {self.tagtraum_url}")
        try:
            if self.tagtraum_url.endswith(".zip"):
                import zipfile
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_path = Path(tmp_dir) / "tagtraum.zip"
                    urllib.request.urlretrieve(self.tagtraum_url, str(zip_path))
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        # Find the .cls file inside
                        cls_names = [n for n in zf.namelist() if n.endswith(".cls")]
                        if not cls_names:
                            logger.error("No .cls file found in Tagtraum ZIP")
                            return False
                        with zf.open(cls_names[0]) as src, open(self.tagtraum_file, "wb") as dst:
                            dst.write(src.read())
            else:
                urllib.request.urlretrieve(self.tagtraum_url, str(self.tagtraum_file))
            logger.info(f"Tagtraum file saved to {self.tagtraum_file}")
            return True
        except Exception as exc:
            logger.error(f"Failed to download Tagtraum: {exc}")
            return False

    def _download_lmd(self) -> bool:
        """Download and extract Lakh MIDI matched dataset."""
        self.lmd_matched_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading Lakh MIDI matched: {self.lmd_matched_url}")
        logger.info("This is a ~1.6 GB download and may take a while...")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                archive_path = Path(tmp_dir) / "lmd_matched.tar.gz"
                urllib.request.urlretrieve(self.lmd_matched_url, str(archive_path))
                logger.info("Download complete. Extracting archive...")
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(self.lmd_matched_dir.parent)
            logger.info(f"Lakh MIDI extracted to {self.lmd_matched_dir}")
            return True
        except Exception as exc:
            logger.error(f"Failed to download/extract Lakh: {exc}")
            return False

    # ------------------------------------------------------------------
    # Convenience: copy genre subset to data/raw/<dataset_name>
    # ------------------------------------------------------------------

    def populate_raw_datasets(self) -> Dict[str, int]:
        """Copy genre MIDI subsets into ``data/raw/lakh_<genre>/`` directories.

        This makes them accessible through the standard ``DatasetManager``
        interface used by the rest of the pipeline.

        Returns dict ``{dataset_name: n_files_copied}``.
        """
        genre_files = self.load_genre_files()
        raw_dir = Path(self.config["data"]["raw_data_dir"])
        counts: Dict[str, int] = {}

        for genre, paths in genre_files.items():
            dataset_name = f"{self._dataset_prefix}_{genre}"
            dest_dir = raw_dir / dataset_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            copied = 0
            for src in paths:
                dst = dest_dir / src.name
                # Avoid re-copying if destination already exists with same size
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



