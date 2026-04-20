"""Tests for embedding extraction module."""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings.extractor import (
    EmbeddingFactory,
    MusicBERTModel,
    MusicBERTLargeModel,
    NLPBaselineModel,
    MERTModel,
    EmbeddingExtractor,
    EmbeddingAnalyzer,
)
from utils.config import load_config


class TestEmbeddingFactory:
    """Test EmbeddingFactory class."""

    def test_create_musicbert_model(self):
        """Test creating MusicBERT model."""
        config = load_config("configs/config.yaml")
        model = EmbeddingFactory.create_model(config, "MusicBERT")
        assert isinstance(model, MusicBERTModel)

    def test_create_musicbert_large_model(self):
        """Test creating MusicBERT-large model."""
        config = load_config("configs/config.yaml")
        model = EmbeddingFactory.create_model(config, "MusicBERT-large")
        assert isinstance(model, MusicBERTLargeModel)

    def test_unknown_model(self):
        """Test error handling for unknown model."""
        config = load_config("configs/config.yaml")
        with pytest.raises(ValueError, match="Unknown model"):
            EmbeddingFactory.create_model(config, "UNKNOWN_MODEL")

    def test_get_available_models(self):
        """Test getting list of available models."""
        models = EmbeddingFactory.get_available_models()
        assert "MusicBERT" in models
        assert "MusicBERT-large" in models
        assert "MERT" in models
        assert "NLP-Baseline" in models
        assert len(models) == 4


class TestMusicBERTModel:
    """Test MusicBERT model."""

    def test_model_initialization(self):
        """Test MusicBERT model initialization."""
        config = load_config("configs/config.yaml")
        model = MusicBERTModel(config)
        assert model.model_name == "MusicBERT"
        assert model.embedding_dim > 0

    def test_encode_single_sequence(self):
        """Test encoding single token sequence."""
        config = load_config("configs/config.yaml")
        model = MusicBERTModel(config)

        tokens = [1, 2, 3, 4, 5]
        embedding = model.encode(tokens)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (model.embedding_dim,)
        assert embedding.dtype == np.float32

    def test_encode_empty_sequence(self):
        """Test encoding empty token sequence."""
        config = load_config("configs/config.yaml")
        model = MusicBERTModel(config)

        tokens = []
        embedding = model.encode(tokens)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (model.embedding_dim,)

    def test_encode_batch(self):
        """Test batch encoding."""
        config = load_config("configs/config.yaml")
        model = MusicBERTModel(config)

        token_sequences = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        embeddings = model.encode_batch(token_sequences)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, model.embedding_dim)
        assert embeddings.dtype == np.float32

    def test_get_embedding_dim(self):
        """Test getting embedding dimension."""
        config = load_config("configs/config.yaml")
        model = MusicBERTModel(config)

        dim = model.get_embedding_dim()
        assert dim > 0
        assert isinstance(dim, int)


class TestEmbeddingExtractor:
    """Test EmbeddingExtractor class."""

    def test_extractor_initialization(self):
        """Test initializing extractor."""
        config = load_config("configs/config.yaml")
        extractor = EmbeddingExtractor(config)
        
        assert extractor.models is not None
        assert "MusicBERT" in extractor.models
        assert "MERT" in extractor.models

    def test_extract_embeddings(self):
        """Test extracting embeddings."""
        config = load_config("configs/config.yaml")
        extractor = EmbeddingExtractor(config)
        
        token_sequences = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ]
        
        embeddings = extractor.extract_embeddings(token_sequences, "MusicBERT")

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.dtype == np.float32

    def test_extract_embeddings_multiple_models(self):
        """Test extracting embeddings with multiple models."""
        config = load_config("configs/config.yaml")
        extractor = EmbeddingExtractor(config)
        
        token_sequences = [[1, 2, 3, 4, 5]]
        
        emb1 = extractor.extract_embeddings(token_sequences, "MusicBERT")
        emb2 = extractor.extract_embeddings(token_sequences, "NLP-Baseline")

        # Both should produce valid embeddings
        assert emb1.shape[0] == 1
        assert emb2.shape[0] == 1

    def test_unknown_model_error(self):
        """Test error handling for unknown model."""
        config = load_config("configs/config.yaml")
        extractor = EmbeddingExtractor(config)
        
        with pytest.raises(ValueError, match="Unknown model"):
            extractor.extract_embeddings([[1, 2, 3]], "UNKNOWN_MODEL")

    def test_cache_system(self):
        """Test cache system."""
        config = load_config("configs/config.yaml")
        config["embeddings"]["cache_embeddings"] = True
        extractor = EmbeddingExtractor(config)
        
        token_sequences = [[1, 2, 3, 4, 5]]
        
        # First extraction
        emb1 = extractor.extract_embeddings(token_sequences, "MusicBERT")

        # Second extraction (should use cache)
        emb2 = extractor.extract_embeddings(token_sequences, "MusicBERT")

        # Should be identical due to caching
        np.testing.assert_array_equal(emb1, emb2)

    def test_cache_miss(self):
        """Test cache miss."""
        config = load_config("configs/config.yaml")
        config["embeddings"]["cache_embeddings"] = True
        extractor = EmbeddingExtractor(config)
        
        token_sequences1 = [[1, 2, 3, 4, 5]]
        token_sequences2 = [[6, 7, 8, 9, 10]]
        
        emb1 = extractor.extract_embeddings(token_sequences1, "MusicBERT")
        emb2 = extractor.extract_embeddings(token_sequences2, "MusicBERT")

        # Should be different (cache miss)
        assert not np.allclose(emb1[0], emb2[0])

    def test_load_tokens_from_text_file(self):
        """Test loading tokens from text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1 2 3 4 5")
            temp_path = Path(f.name)
        
        try:
            tokens = EmbeddingExtractor._load_tokens_from_file(temp_path)
            assert tokens == [1, 2, 3, 4, 5]
        finally:
            temp_path.unlink()

    def test_load_tokens_from_json_file(self):
        """Test loading tokens from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([10, 20, 30, 40], f)
            temp_path = Path(f.name)
        
        try:
            tokens = EmbeddingExtractor._load_tokens_from_file(temp_path)
            assert tokens == [10, 20, 30, 40]
        finally:
            temp_path.unlink()

    def test_load_tokens_from_json_with_key(self):
        """Test loading tokens from JSON file with 'tokens' key."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"tokens": [5, 6, 7, 8]}, f)
            temp_path = Path(f.name)
        
        try:
            tokens = EmbeddingExtractor._load_tokens_from_file(temp_path)
            assert tokens == [5, 6, 7, 8]
        finally:
            temp_path.unlink()

    def test_extract_dataset_embeddings(self):
        """Test extracting embeddings for dataset."""
        config = load_config("configs/config.yaml")
        extractor = EmbeddingExtractor(config)
        
        # Create temporary token files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create token files
            token_files = []
            for i in range(3):
                token_file = tmp_path / f"tokens_{i}.txt"
                token_file.write_text(f"{i+1} {i+2} {i+3} {i+4} {i+5}")
                token_files.append(token_file)
            
            # Extract embeddings
            output_dir = tmp_path / "embeddings"
            stats = extractor.extract_dataset_embeddings(token_files, output_dir, "MusicBERT")

            assert stats["successful"] == 3
            assert stats["failed"] == 0
            assert stats["model"] == "MusicBERT"
            assert len(list(output_dir.glob("*.npy"))) == 3


class TestEmbeddingAnalyzer:
    """Test EmbeddingAnalyzer class."""

    def test_compute_statistics(self):
        """Test computing embedding statistics."""
        embeddings = np.random.randn(10, 64)
        stats = EmbeddingAnalyzer.compute_statistics(embeddings)
        
        assert "mean" in stats
        assert "std" in stats
        assert "cov" in stats
        assert "num_samples" in stats
        assert "embedding_dim" in stats
        
        assert stats["mean"].shape == (64,)
        assert stats["std"].shape == (64,)
        assert stats["num_samples"] == 10
        assert stats["embedding_dim"] == 64

    def test_compute_pairwise_distances_euclidean(self):
        """Test computing Euclidean distances."""
        embeddings1 = np.random.randn(5, 64)
        embeddings2 = np.random.randn(3, 64)
        
        distances = EmbeddingAnalyzer.compute_pairwise_distances(
            embeddings1, embeddings2, metric="euclidean"
        )
        
        assert distances.shape == (5, 3)
        assert np.all(distances >= 0)

    def test_compute_pairwise_distances_cosine(self):
        """Test computing cosine distances."""
        embeddings1 = np.random.randn(5, 64)
        embeddings2 = np.random.randn(3, 64)
        
        distances = EmbeddingAnalyzer.compute_pairwise_distances(
            embeddings1, embeddings2, metric="cosine"
        )
        
        assert distances.shape == (5, 3)
        assert np.all(distances >= 0)
        assert np.all(distances <= 2)  # Cosine distance is in [0, 2]

    def test_identical_embeddings_distance_zero(self):
        """Test that identical embeddings have zero distance."""
        embeddings = np.random.randn(5, 64)
        
        distances = EmbeddingAnalyzer.compute_pairwise_distances(
            embeddings, embeddings, metric="euclidean"
        )
        
        # Diagonal should be close to zero
        assert np.allclose(np.diag(distances), 0, atol=1e-6)

    def test_cosine_distance_symmetry(self):
        """Test that cosine distances are symmetric."""
        embeddings1 = np.random.randn(3, 64)
        embeddings2 = np.random.randn(3, 64)
        
        dist12 = EmbeddingAnalyzer.compute_pairwise_distances(
            embeddings1, embeddings2, metric="cosine"
        )
        dist21 = EmbeddingAnalyzer.compute_pairwise_distances(
            embeddings2, embeddings1, metric="cosine"
        )
        
        np.testing.assert_array_almost_equal(dist12, dist21.T)

    def test_unknown_metric_error(self):
        """Test error handling for unknown metric."""
        embeddings1 = np.random.randn(5, 64)
        embeddings2 = np.random.randn(3, 64)
        
        with pytest.raises(ValueError, match="Unknown metric"):
            EmbeddingAnalyzer.compute_pairwise_distances(
                embeddings1, embeddings2, metric="invalid"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

