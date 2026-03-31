"""Data management module for MIDI datasets."""

import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger


class DatasetManager:
    """Manager for MIDI datasets."""

    def __init__(self, config: Dict):
        """
        Initialize the dataset manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.raw_data_dir = Path(config["data"]["raw_data_dir"])
        self.processed_data_dir = Path(config["data"]["processed_data_dir"])
        self.embeddings_dir = Path(config["data"]["embeddings_dir"])

        # Create directories if they don't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DatasetManager initialized with raw_data_dir: {self.raw_data_dir}")

    def get_dataset_path(self, dataset_name: str, processed: bool = False) -> Path:
        """
        Get path to a dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'maestro', 'pop909')
            processed: If True, return processed data path, else raw data path

        Returns:
            Path to dataset directory
        """
        base_dir = self.processed_data_dir if processed else self.raw_data_dir
        dataset_path = base_dir / dataset_name
        return dataset_path

    def list_midi_files(
        self, dataset_name: str, processed: bool = False, limit: Optional[int] = None
    ) -> List[Path]:
        """
        List all MIDI files in a dataset.

        Args:
            dataset_name: Name of the dataset
            processed: If True, list from processed data, else raw data
            limit: Maximum number of files to return (None for all)

        Returns:
            List of Path objects pointing to MIDI files
        """
        dataset_path = self.get_dataset_path(dataset_name, processed)

        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return []

        midi_files = list(dataset_path.rglob("*.mid")) + list(dataset_path.rglob("*.midi"))
        midi_files = sorted(midi_files)

        if limit:
            midi_files = midi_files[:limit]

        logger.info(f"Found {len(midi_files)} MIDI files in {dataset_name}")
        return midi_files

    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset from config.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset configuration dictionary
        """
        for dataset in self.config["data"]["datasets"]:
            if dataset["name"].lower() == dataset_name.lower():
                return dataset

        raise ValueError(f"Dataset '{dataset_name}' not found in configuration")

    def ensure_dataset_exists(self, dataset_name: str, download: bool = True) -> bool:
        """
        Ensure dataset exists. Download if necessary and configured.

        Args:
            dataset_name: Name of the dataset
            download: If True, attempt to download missing datasets

        Returns:
            True if dataset exists or was successfully downloaded
        """
        dataset_path = self.get_dataset_path(dataset_name, processed=False)

        if dataset_path.exists() and len(list(dataset_path.glob("*.mid*"))) > 0:
            logger.info(f"Dataset '{dataset_name}' already exists at {dataset_path}")
            return True

        if download:
            info = self.get_dataset_info(dataset_name)
            url = str(info.get("url", "")).strip()
            if not url:
                logger.error(f"Dataset '{dataset_name}' has no URL configured")
                return False

            logger.info(f"Dataset '{dataset_name}' not found. Downloading from: {url}")
            ok = self._download_dataset(url=url, dataset_path=dataset_path)
            if not ok:
                return False

            midi_count = len(list(dataset_path.rglob("*.mid"))) + len(list(dataset_path.rglob("*.midi")))
            logger.info(f"Download completed for '{dataset_name}'. MIDI files found: {midi_count}")
            return midi_count > 0

        return False

    def _download_dataset(self, url: str, dataset_path: Path) -> bool:
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Archive URLs can be downloaded and extracted directly.
        if url.endswith(".zip") or url.endswith(".tar") or url.endswith(".tar.gz") or url.endswith(".tgz"):
            return self._download_and_extract_archive(url, dataset_path)

        # For GitHub repositories (e.g., MidiCaps/POP909), shallow clone is usually enough.
        if "github.com" in url:
            return self._clone_repository(url, dataset_path)

        # Best effort: attempt archive download even if extension is unknown.
        return self._download_and_extract_archive(url, dataset_path)

    def _download_and_extract_archive(self, url: str, dataset_path: Path) -> bool:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                archive_path = tmp_dir_path / "dataset_archive"
                logger.info(f"Downloading archive: {url}")
                urllib.request.urlretrieve(url, str(archive_path))

                if zipfile.is_zipfile(archive_path):
                    with zipfile.ZipFile(archive_path, "r") as zip_ref:
                        zip_ref.extractall(dataset_path)
                    return True

                if tarfile.is_tarfile(archive_path):
                    with tarfile.open(archive_path, "r:*") as tar_ref:
                        tar_ref.extractall(dataset_path)
                    return True

                logger.error(f"Unsupported archive format from URL: {url}")
                return False
        except Exception as exc:
            logger.error(f"Archive download/extract failed for {url}: {exc}")
            return False

    def _clone_repository(self, url: str, dataset_path: Path) -> bool:
        try:
            # Clean empty target to avoid clone failure on existing directory.
            if dataset_path.exists() and not any(dataset_path.iterdir()):
                shutil.rmtree(dataset_path)

            cmd = ["git", "clone", "--depth", "1", url, str(dataset_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Git clone failed: {result.stderr.strip()}")
                return False
            return True
        except Exception as exc:
            logger.error(f"Git clone exception for {url}: {exc}")
            return False


class DataProcessor:
    """Process and standardize MIDI data."""

    def __init__(self, config: Dict):
        """
        Initialize the data processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info("DataProcessor initialized")

    def validate_midi_file(self, midi_path: Path) -> bool:
        """
        Validate if a MIDI file exists and is readable.

        Args:
            midi_path: Path to MIDI file

        Returns:
            True if file is valid, False otherwise
        """
        if not midi_path.exists():
            return False

        if not midi_path.suffix.lower() in [".mid", ".midi"]:
            return False

        try:
            # Try to open and verify the file has some content
            file_size = midi_path.stat().st_size
            return file_size > 0
        except Exception as e:
            logger.error(f"Error validating MIDI file {midi_path}: {e}")
            return False

    def get_file_statistics(self, midi_path: Path) -> Dict:
        """
        Get basic statistics about a MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Dictionary with file statistics
        """
        stats = {
            "path": str(midi_path),
            "exists": midi_path.exists(),
            "size_bytes": midi_path.stat().st_size if midi_path.exists() else 0,
            "valid": self.validate_midi_file(midi_path),
        }
        return stats
