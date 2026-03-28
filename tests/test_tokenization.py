"""Tests for MIDI tokenization module."""

import pytest
from pathlib import Path
import pretty_midi
import tempfile
import yaml

from src.tokenization.tokenizer import (
    REMITokenizer,
    TSDTokenizer,
    OctupleTokenizer,
    MIDILikeTokenizer,
    TokenizationFactory,
    TokenizationPipeline,
)


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_midi_file():
    """Create a simple MIDI file for testing."""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    # Add some notes (C major scale)
    for i, pitch in enumerate([60, 62, 64, 65, 67]):
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=i * 0.5,
            end=(i + 0.4) * 0.5
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        midi.write(str(tmp_path))
        yield tmp_path
        # Cleanup
        if tmp_path.exists():
            tmp_path.unlink()


class TestTokenizers:
    """Test individual tokenizers."""

    def test_remi_tokenizer_initialization(self, config):
        """Test REMI tokenizer initialization."""
        tokenizer = REMITokenizer(config)
        assert tokenizer.tokenizer_type == "REMI"
        assert tokenizer.miditok_tokenizer is not None
        assert tokenizer.get_vocab_size() > 0

    def test_tsd_tokenizer_initialization(self, config):
        """Test TSD tokenizer initialization."""
        tokenizer = TSDTokenizer(config)
        assert tokenizer.tokenizer_type == "TSD"
        assert tokenizer.miditok_tokenizer is not None
        assert tokenizer.get_vocab_size() > 0

    def test_octuple_tokenizer_initialization(self, config):
        """Test Octuple tokenizer initialization."""
        tokenizer = OctupleTokenizer(config)
        assert tokenizer.tokenizer_type == "Octuple"
        assert tokenizer.miditok_tokenizer is not None
        assert tokenizer.get_vocab_size() > 0

    def test_midilike_tokenizer_initialization(self, config):
        """Test MIDI-Like tokenizer initialization."""
        tokenizer = MIDILikeTokenizer(config)
        assert tokenizer.tokenizer_type == "MIDI-Like"
        assert tokenizer.miditok_tokenizer is not None
        assert tokenizer.get_vocab_size() > 0

    def test_remi_encode(self, config, sample_midi_file):
        """Test REMI encoding."""
        tokenizer = REMITokenizer(config)
        tokens = tokenizer.encode(sample_midi_file)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_tsd_encode(self, config, sample_midi_file):
        """Test TSD encoding."""
        tokenizer = TSDTokenizer(config)
        tokens = tokenizer.encode(sample_midi_file)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_octuple_encode(self, config, sample_midi_file):
        """Test Octuple encoding."""
        tokenizer = OctupleTokenizer(config)
        tokens = tokenizer.encode(sample_midi_file)

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_midilike_encode(self, config, sample_midi_file):
        """Test MIDI-Like encoding."""
        tokenizer = MIDILikeTokenizer(config)
        tokens = tokenizer.encode(sample_midi_file)

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_encode_decode_roundtrip(self, config, sample_midi_file):
        """Test that encoding and decoding produces valid MIDI."""
        tokenizer = REMITokenizer(config)

        # Encode
        tokens = tokenizer.encode(sample_midi_file)
        assert len(tokens) > 0

        # Decode
        decoded_midi = tokenizer.decode(tokens)
        assert decoded_midi is not None
        assert isinstance(decoded_midi, pretty_midi.PrettyMIDI)

    def test_get_token_sequence_length(self, config, sample_midi_file):
        """Test getting token sequence length."""
        tokenizer = REMITokenizer(config)
        length = tokenizer.get_token_sequence_length(sample_midi_file)

        assert length > 0
        assert isinstance(length, int)

    def test_different_tokenizers_produce_different_tokens(self, config, sample_midi_file):
        """Test that different tokenizers produce different token sequences."""
        remi_tokenizer = REMITokenizer(config)
        tsd_tokenizer = TSDTokenizer(config)

        remi_tokens = remi_tokenizer.encode(sample_midi_file)
        tsd_tokens = tsd_tokenizer.encode(sample_midi_file)

        # Different tokenizers should produce different sequences
        # (or at least different lengths in most cases)
        assert remi_tokens != tsd_tokens or len(remi_tokens) != len(tsd_tokens)


class TestTokenizationFactory:
    """Test tokenization factory."""

    def test_create_remi_tokenizer(self, config):
        """Test creating REMI tokenizer."""
        tokenizer = TokenizationFactory.create_tokenizer(config, "REMI")
        assert isinstance(tokenizer, REMITokenizer)

    def test_create_tsd_tokenizer(self, config):
        """Test creating TSD tokenizer."""
        tokenizer = TokenizationFactory.create_tokenizer(config, "TSD")
        assert isinstance(tokenizer, TSDTokenizer)

    def test_create_octuple_tokenizer(self, config):
        """Test creating Octuple tokenizer."""
        tokenizer = TokenizationFactory.create_tokenizer(config, "Octuple")
        assert isinstance(tokenizer, OctupleTokenizer)

    def test_create_midilike_tokenizer(self, config):
        """Test creating MIDI-Like tokenizer."""
        tokenizer = TokenizationFactory.create_tokenizer(config, "MIDI-Like")
        assert isinstance(tokenizer, MIDILikeTokenizer)

    def test_create_invalid_tokenizer(self, config):
        """Test creating invalid tokenizer raises error."""
        with pytest.raises(ValueError):
            TokenizationFactory.create_tokenizer(config, "InvalidType")

    def test_get_available_tokenizers(self):
        """Test getting available tokenizers."""
        tokenizers = TokenizationFactory.get_available_tokenizers()
        assert isinstance(tokenizers, list)
        assert "REMI" in tokenizers
        assert "TSD" in tokenizers
        assert "Octuple" in tokenizers
        assert "MIDI-Like" in tokenizers


class TestTokenizationPipeline:
    """Test tokenization pipeline."""

    def test_initialization(self, config):
        """Test pipeline initialization."""
        pipeline = TokenizationPipeline(config)
        assert len(pipeline.tokenizers) == 4
        assert "REMI" in pipeline.tokenizers
        assert "TSD" in pipeline.tokenizers

    def test_tokenize_midi(self, config, sample_midi_file):
        """Test tokenizing MIDI file."""
        pipeline = TokenizationPipeline(config)
        tokens = pipeline.tokenize_midi(sample_midi_file, "REMI")

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_midi_invalid_type(self, config, sample_midi_file):
        """Test tokenizing with invalid tokenizer type."""
        pipeline = TokenizationPipeline(config)

        with pytest.raises(ValueError):
            pipeline.tokenize_midi(sample_midi_file, "InvalidType")

    def test_tokenize_dataset(self, config, sample_midi_file):
        """Test tokenizing multiple files."""
        pipeline = TokenizationPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "tokens"
            midi_files = [sample_midi_file]

            stats = pipeline.tokenize_dataset(
                midi_files,
                output_dir,
                "REMI"
            )

            assert stats["total"] == 1
            assert stats["successful"] == 1
            assert stats["failed"] == 0
            assert stats["vocab_size"] > 0
            assert "avg_token_length" in stats
            assert output_dir.exists()

    def test_load_tokens(self, config, sample_midi_file):
        """Test loading tokens from file."""
        pipeline = TokenizationPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "tokens"
            midi_files = [sample_midi_file]

            # Tokenize and save
            pipeline.tokenize_dataset(midi_files, output_dir, "REMI")

            # Load tokens
            token_files = list(output_dir.glob("*_REMI.json"))
            assert len(token_files) > 0

            loaded = pipeline.load_tokens(token_files[0])
            assert "tokens" in loaded
            assert isinstance(loaded["tokens"], list)

    def test_compare_tokenizers(self, config, sample_midi_file):
        """Test comparing all tokenizers."""
        pipeline = TokenizationPipeline(config)
        results = pipeline.compare_tokenizers(sample_midi_file)

        assert isinstance(results, dict)
        assert "REMI" in results
        assert "TSD" in results
        assert "Octuple" in results
        assert "MIDI-Like" in results

        # Check each result has expected fields
        for tok_type, result in results.items():
            if "error" not in result:
                assert "token_count" in result
                assert "vocab_size" in result
                assert result["token_count"] > 0
                assert result["vocab_size"] > 0

    def test_tokenize_dataset_with_invalid_files(self, config):
        """Test tokenizing with invalid files."""
        pipeline = TokenizationPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "tokens"
            # Non-existent files
            midi_files = [Path("nonexistent.mid")]

            stats = pipeline.tokenize_dataset(
                midi_files,
                output_dir,
                "REMI"
            )

            assert stats["total"] == 1
            assert stats["failed"] == 1
            assert len(stats["failed_files"]) == 1


class TestTokenizerComparison:
    """Test comparison between tokenizers."""

    def test_all_tokenizers_produce_valid_output(self, config, sample_midi_file):
        """Test that all tokenizers produce valid output."""
        tokenizers = {
            "REMI": REMITokenizer(config),
            "TSD": TSDTokenizer(config),
            "Octuple": OctupleTokenizer(config),
            "MIDI-Like": MIDILikeTokenizer(config),
        }

        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.encode(sample_midi_file)
            assert len(tokens) > 0, f"{name} produced empty token sequence"
            assert all(isinstance(t, int) for t in tokens), f"{name} produced non-integer tokens"

    def test_vocab_sizes_are_reasonable(self, config):
        """Test that vocab sizes are within reasonable ranges."""
        tokenizers = {
            "REMI": REMITokenizer(config),
            "TSD": TSDTokenizer(config),
            "Octuple": OctupleTokenizer(config),
            "MIDI-Like": MIDILikeTokenizer(config),
        }

        for name, tokenizer in tokenizers.items():
            vocab_size = tokenizer.get_vocab_size()
            # Vocab size should be reasonable (not 0, not too large)
            assert 100 < vocab_size < 10000, f"{name} has unreasonable vocab size: {vocab_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
