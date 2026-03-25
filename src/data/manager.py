"""Data management module for MIDI datasets."""

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
            logger.warning(f"Dataset '{dataset_name}' not found. Would download from source.")
            logger.info(
                "Download functionality needs to be implemented based on specific dataset requirements."
            )
            # Actual download logic would be implemented here for each dataset
            return False

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
