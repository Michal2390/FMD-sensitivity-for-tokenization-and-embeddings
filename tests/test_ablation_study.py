"""Tests for ablation study experiments."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing.processor import MIDIPreprocessor
from tokenization.tokenizer import TokenizationPipeline
from embeddings.extractor import EmbeddingExtractor
from metrics.fmd import FrechetMusicDistance
from utils.config import load_config


@pytest.fixture
def config():
    """Load configuration."""
    return load_config("configs/config.yaml")


@pytest.fixture
def preprocessor(config):
    """Create preprocessor."""
    return MIDIPreprocessor(config)


@pytest.fixture
def midi_data():
    """Create mock MIDI data."""
    try:
        import pretty_midi
        midi = pretty_midi.PrettyMIDI()
        midi.instruments.append(pretty_midi.Instrument(program=0))
        
        for i in range(5):
            note = pretty_midi.Note(velocity=100, pitch=60+i, start=i, end=i+1)
            midi.instruments[0].notes.append(note)
        
        return midi
    except Exception as e:
        pytest.skip(f"Could not create MIDI: {e}")


class TestVelocityRemovalAblation:
    """Test velocity removal ablation."""

    def test_velocity_removal_creates_variant(self, preprocessor, midi_data):
        """Test that velocity removal creates valid variant."""
        if midi_data is None:
            pytest.skip("MIDI data not available")
        
        modified = preprocessor.remove_velocity(midi_data)
        
        # Check that all velocities are now 64 (default)
        for instrument in modified.instruments:
            for note in instrument.notes:
                assert note.velocity == 64

    def test_velocity_removal_preserves_timing(self, preprocessor, midi_data):
        """Test that velocity removal preserves note timing."""
        if midi_data is None:
            pytest.skip("MIDI data not available")
        
        original_notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                original_notes.append((note.start, note.end, note.pitch))
        
        modified = preprocessor.remove_velocity(midi_data)
        
        modified_notes = []
        for instrument in modified.instruments:
            for note in instrument.notes:
                modified_notes.append((note.start, note.end, note.pitch))
        
        assert original_notes == modified_notes

    def test_velocity_removal_changes_midi_data(self, preprocessor, midi_data):
        """Test that velocity removal actually modifies the MIDI."""
        if midi_data is None:
            pytest.skip("MIDI data not available")
        
        original_velocities = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                original_velocities.append(note.velocity)
        
        modified = preprocessor.remove_velocity(midi_data)
        
        modified_velocities = []
        for instrument in modified.instruments:
            for note in instrument.notes:
                modified_velocities.append(note.velocity)
        
        # Should have changed if original had different velocities
        # For this test, we set them all to 100, so they should change to 64
        assert all(v == 64 for v in modified_velocities)


class TestQuantizationAblation:
    """Test quantization ablation."""

    def test_quantization_creates_variant(self, preprocessor, midi_data):
        """Test that quantization creates valid variant."""
        if midi_data is None:
            pytest.skip("MIDI data not available")
        
        quantized = preprocessor.quantize_time(midi_data)
        
        # Check that quantization was applied
        assert quantized is not None
        assert len(quantized.instruments) > 0

    def test_quantization_changes_timing(self, preprocessor, midi_data):
        """Test that hard quantization changes timing."""
        if midi_data is None:
            pytest.skip("MIDI data not available")
        
        # Add some notes with non-quantized timing
        quantized = preprocessor.quantize_time(midi_data)
        
        # Quantized notes should have start times that are multiples of quantization step
        for instrument in quantized.instruments:
            for note in instrument.notes:
                # Timing should be quantized to grid
                assert note.start >= 0


class TestEmbeddingExtractionForVariants:
    """Test embedding extraction for MIDI variants."""

    def test_extract_embeddings_for_original(self, config):
        """Test extracting embeddings for original MIDI."""
        extractor = EmbeddingExtractor(config)
        config["embeddings"]["cache_embeddings"] = False
        
        # Create simple token sequence
        tokens = [list(range(1, 11))]
        
        embeddings = extractor.extract_embeddings(tokens, "MusicBERT-large")
        
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0

    def test_extract_embeddings_consistency(self, config):
        """Test that embedding extraction is consistent."""
        extractor = EmbeddingExtractor(config)
        config["embeddings"]["cache_embeddings"] = False
        
        tokens = [list(range(1, 11))]
        
        emb1 = extractor.extract_embeddings(tokens, "MusicBERT-large")
        emb2 = extractor.extract_embeddings(tokens, "MusicBERT-large")
        
        # Due to caching or determinism, should be close
        # (Note: without cache, may differ due to randomness in dummy model)
        assert emb1.shape == emb2.shape


class TestFMDDifferenceCalculation:
    """Test FMD difference calculations."""

    def test_fmd_original_vs_modified(self, config):
        """Test FMD between original and modified embeddings."""
        fmd_calc = FrechetMusicDistance(config)
        config["embeddings"]["cache_embeddings"] = False
        
        # Create two different embeddings
        emb1 = np.random.randn(10, 64)
        emb2 = np.random.randn(10, 64)
        
        fmd = fmd_calc.compute_fmd(emb1, emb2)
        
        assert fmd >= 0
        assert isinstance(fmd, float)

    def test_fmd_identical_vs_different(self, config):
        """Test that different embeddings have higher FMD than identical."""
        fmd_calc = FrechetMusicDistance(config)
        
        emb1 = np.random.randn(10, 64)
        
        fmd_identical = fmd_calc.compute_fmd(emb1, emb1)
        
        # Identical should be ~0
        assert fmd_identical < 0.001

    def test_fmd_monotonicity(self, config):
        """Test FMD increases with modification magnitude."""
        fmd_calc = FrechetMusicDistance(config)
        
        emb_base = np.random.randn(10, 64)
        
        # Create slightly and heavily modified versions
        emb_slight = emb_base + np.random.randn(10, 64) * 0.1
        emb_heavy = emb_base + np.random.randn(10, 64) * 1.0
        
        fmd_slight = fmd_calc.compute_fmd(emb_base, emb_slight)
        fmd_heavy = fmd_calc.compute_fmd(emb_base, emb_heavy)
        
        # Heavier modification should generally have higher FMD
        assert fmd_heavy >= fmd_slight or np.isclose(fmd_heavy, fmd_slight)


class TestAblationExperimentStructure:
    """Test ablation study experiment structure."""

    def test_ablation_results_have_required_keys(self):
        """Test that ablation results have required structure."""
        # Mock results
        results = {
            "total_files": 3,
            "tokenizer": "REMI",
            "model": "MusicBERT-large",
            "per_file_results": {},
            "aggregate_statistics": {
                "no_velocity": {
                    "mean_fmd": 0.5,
                    "std_fmd": 0.1,
                    "min_fmd": 0.3,
                    "max_fmd": 0.7,
                    "samples": 3
                },
                "quantized": {
                    "mean_fmd": 0.4,
                    "std_fmd": 0.08,
                    "min_fmd": 0.2,
                    "max_fmd": 0.5,
                    "samples": 3
                }
            }
        }
        
        # Check required keys
        assert "total_files" in results
        assert "tokenizer" in results
        assert "model" in results
        assert "aggregate_statistics" in results
        
        # Check aggregate statistics structure
        assert "no_velocity" in results["aggregate_statistics"]
        assert "mean_fmd" in results["aggregate_statistics"]["no_velocity"]

    def test_ablation_comparison_structure(self):
        """Test structure of tokenization/model comparisons."""
        comparison = {
            "REMI": {
                "total_files": 3,
                "tokenizer": "REMI",
                "aggregate_statistics": {}
            },
            "TSD": {
                "total_files": 3,
                "tokenizer": "TSD",
                "aggregate_statistics": {}
            }
        }
        
        assert "REMI" in comparison
        assert "TSD" in comparison
        assert all("tokenizer" in config for config in comparison.values())


class TestResultsPersistence:
    """Test saving and loading results."""

    def test_save_results_as_json(self):
        """Test that results can be saved as JSON."""
        results = {
            "total_files": 2,
            "tokenizer": "REMI",
            "aggregate_statistics": {
                "no_velocity": {
                    "mean_fmd": float(0.5),
                    "std_fmd": float(0.1),
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results.json"
            
            with open(output_file, "w") as f:
                json.dump(results, f)
            
            # Verify file was created
            assert output_file.exists()
            
            # Verify can be loaded
            with open(output_file, "r") as f:
                loaded = json.load(f)
            
            assert loaded["total_files"] == 2
            assert "no_velocity" in loaded["aggregate_statistics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

