"""Tests for MIDI preprocessing module."""

import pytest
from pathlib import Path
import pretty_midi
import tempfile
import yaml

from src.preprocessing.processor import MIDIPreprocessor, PreprocessingPipeline


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_midi():
    """Create a simple MIDI file for testing."""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    # Add some notes
    for i, pitch in enumerate([60, 62, 64, 65, 67]):  # C major scale
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=i * 0.5,
            end=(i + 1) * 0.5
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi


@pytest.fixture
def sample_midi_file(sample_midi):
    """Create a temporary MIDI file."""
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        sample_midi.write(str(tmp_path))
        yield tmp_path
        # Cleanup
        if tmp_path.exists():
            tmp_path.unlink()


class TestMIDIPreprocessor:
    """Test MIDI preprocessor."""

    def test_initialization(self, config):
        """Test preprocessor initialization."""
        preprocessor = MIDIPreprocessor(config)
        assert preprocessor.min_note == 21
        assert preprocessor.max_note == 108
        assert preprocessor.quantization_resolution == 480

    def test_load_midi(self, config, sample_midi_file):
        """Test MIDI loading."""
        preprocessor = MIDIPreprocessor(config)
        midi_data = preprocessor.load_midi(sample_midi_file)
        assert midi_data is not None
        assert len(midi_data.instruments) > 0

    def test_load_invalid_midi(self, config):
        """Test loading invalid MIDI file."""
        preprocessor = MIDIPreprocessor(config)
        midi_data = preprocessor.load_midi(Path("nonexistent.mid"))
        assert midi_data is None

    def test_remove_velocity(self, config, sample_midi):
        """Test velocity removal."""
        preprocessor = MIDIPreprocessor(config)

        # Check original velocities are different
        original_velocity = sample_midi.instruments[0].notes[0].velocity
        assert original_velocity != 64

        # Remove velocity
        processed = preprocessor.remove_velocity(sample_midi)

        # Check all velocities are now 64
        for instrument in processed.instruments:
            for note in instrument.notes:
                assert note.velocity == 64

    def test_filter_note_range(self, config):
        """Test note range filtering."""
        preprocessor = MIDIPreprocessor(config)

        # Create MIDI with notes outside range
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        # Add notes: some in range, some out of range
        notes_data = [
            (20, 0.0, 1.0),  # Below range (A0 = 21)
            (60, 1.0, 2.0),  # In range
            (110, 2.0, 3.0),  # Above range (C8 = 108)
        ]

        for pitch, start, end in notes_data:
            note = pretty_midi.Note(velocity=64, pitch=pitch, start=start, end=end)
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        # Filter
        filtered = preprocessor.filter_note_range(midi)

        # Should only have 1 note remaining
        assert len(filtered.instruments[0].notes) == 1
        assert filtered.instruments[0].notes[0].pitch == 60

    def test_quantize_time(self, config, sample_midi):
        """Test time quantization."""
        preprocessor = MIDIPreprocessor(config)

        # Get original times
        original_start = sample_midi.instruments[0].notes[0].start

        # Apply quantization
        quantized = preprocessor.quantize_time(sample_midi, hard_quantize=True)

        # Check that times are quantized
        quantized_start = quantized.instruments[0].notes[0].start
        # Quantized time should be on grid
        assert quantized_start != original_start or original_start == 0.0

    def test_quantize_no_effect_when_disabled(self, config, sample_midi):
        """Test that quantization has no effect when disabled."""
        preprocessor = MIDIPreprocessor(config)

        original_start = sample_midi.instruments[0].notes[0].start
        quantized = preprocessor.quantize_time(sample_midi, hard_quantize=False)
        new_start = quantized.instruments[0].notes[0].start

        assert original_start == new_start

    def test_normalize_instruments(self, config):
        """Test instrument normalization."""
        preprocessor = MIDIPreprocessor(config)

        # Create MIDI with multiple instruments
        midi = pretty_midi.PrettyMIDI()

        # Add 3 non-drum instruments
        for i in range(3):
            instrument = pretty_midi.Instrument(program=i)
            note = pretty_midi.Note(velocity=64, pitch=60 + i, start=0.0, end=1.0)
            instrument.notes.append(note)
            midi.instruments.append(instrument)

        # Normalize
        normalized = preprocessor.normalize_instruments(midi)

        # Should have only 1 non-drum instrument
        non_drum = [inst for inst in normalized.instruments if not inst.is_drum]
        assert len(non_drum) == 1
        # Should have all 3 notes
        assert len(non_drum[0].notes) == 3

    def test_full_preprocess_pipeline(self, config, sample_midi_file):
        """Test full preprocessing pipeline."""
        preprocessor = MIDIPreprocessor(config)

        processed = preprocessor.preprocess(
            sample_midi_file,
            remove_velocity=True,
            hard_quantize=True
        )

        assert processed is not None
        assert len(processed.instruments) > 0

    def test_save_midi(self, config, sample_midi):
        """Test MIDI saving."""
        preprocessor = MIDIPreprocessor(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.mid"
            success = preprocessor.save_midi(sample_midi, output_path)

            assert success
            assert output_path.exists()

            # Verify we can load it back
            loaded = preprocessor.load_midi(output_path)
            assert loaded is not None


class TestPreprocessingPipeline:
    """Test preprocessing pipeline."""

    def test_initialization(self, config):
        """Test pipeline initialization."""
        pipeline = PreprocessingPipeline(config)
        assert pipeline.preprocessor is not None

    def test_process_single_file(self, config, sample_midi_file):
        """Test processing single file."""
        pipeline = PreprocessingPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "processed.mid"
            success = pipeline.process_single_file(
                sample_midi_file,
                output_path,
                remove_velocity=False,
                hard_quantize=False
            )

            assert success
            assert output_path.exists()

    def test_process_dataset(self, config, sample_midi_file):
        """Test batch processing."""
        pipeline = PreprocessingPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "processed"
            midi_files = [sample_midi_file]  # Single file for test

            stats = pipeline.process_dataset(
                midi_files,
                output_dir,
                remove_velocity=False,
                hard_quantize=False
            )

            assert stats["total"] == 1
            assert stats["successful"] == 1
            assert stats["failed"] == 0
            assert output_dir.exists()

    def test_process_dataset_with_invalid_files(self, config):
        """Test batch processing with invalid files."""
        pipeline = PreprocessingPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "processed"
            # List of non-existent files
            midi_files = [Path("nonexistent1.mid"), Path("nonexistent2.mid")]

            stats = pipeline.process_dataset(
                midi_files,
                output_dir,
                remove_velocity=False,
                hard_quantize=False
            )

            assert stats["total"] == 2
            assert stats["successful"] == 0
            assert stats["failed"] == 2
            assert len(stats["failed_files"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
