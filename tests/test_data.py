"""Tests for data management."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.manager import DatasetManager, DataProcessor


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "data": {
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "embeddings_dir": "data/embeddings",
            "datasets": [
                {"name": "maestro", "url": "test_url", "version": "v1"},
                {"name": "pop909", "url": "test_url", "version": "v1"},
            ],
        },
        "preprocessing": {
            "min_note": 21,
            "max_note": 108,
            "quantization_resolution": 480,
            "remove_velocity": False,
            "hard_quantization": False,
        },
    }


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    (tmp_path / "embeddings").mkdir()
    return {
        "raw": tmp_path / "raw",
        "processed": tmp_path / "processed",
        "embeddings": tmp_path / "embeddings",
    }


class TestDatasetManager:
    """Test dataset management."""

    def test_dataset_manager_initialization(self, config):
        """Test DatasetManager initialization."""
        manager = DatasetManager(config)

        assert manager.raw_data_dir == Path("data/raw")
        assert manager.processed_data_dir == Path("data/processed")

    def test_get_dataset_path(self, config):
        """Test getting dataset path."""
        manager = DatasetManager(config)

        raw_path = manager.get_dataset_path("maestro", processed=False)
        processed_path = manager.get_dataset_path("maestro", processed=True)

        assert "maestro" in str(raw_path)
        assert "maestro" in str(processed_path)

    def test_get_dataset_info(self, config):
        """Test getting dataset info from config."""
        manager = DatasetManager(config)

        info = manager.get_dataset_info("maestro")

        assert info["name"] == "maestro"
        assert "url" in info

    def test_list_midi_files(self, config, temp_dirs):
        """Test listing MIDI files."""
        # Create temporary MIDI files
        (temp_dirs["raw"] / "maestro").mkdir()
        (temp_dirs["raw"] / "maestro" / "test1.mid").touch()
        (temp_dirs["raw"] / "maestro" / "test2.midi").touch()

        manager = DatasetManager(config)
        manager.raw_data_dir = temp_dirs["raw"]

        files = manager.list_midi_files("maestro", processed=False)

        assert len(files) == 2


class TestDataProcessor:
    """Test data processing."""

    def test_data_processor_initialization(self, config):
        """Test DataProcessor initialization."""
        processor = DataProcessor(config)

        assert processor.config == config

    def test_validate_midi_file(self, config, temp_dirs):
        """Test MIDI file validation."""
        processor = DataProcessor(config)

        # Create a test MIDI file
        test_file = temp_dirs["raw"] / "test.mid"
        test_file.write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x80")

        assert processor.validate_midi_file(test_file)
        assert not processor.validate_midi_file(Path("nonexistent.mid"))

    def test_get_file_statistics(self, config, temp_dirs):
        """Test getting file statistics."""
        processor = DataProcessor(config)

        test_file = temp_dirs["raw"] / "test.mid"
        test_file.write_bytes(b"test data")

        stats = processor.get_file_statistics(test_file)

        assert "path" in stats
        assert "exists" in stats
        assert stats["exists"]
        assert stats["size_bytes"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
