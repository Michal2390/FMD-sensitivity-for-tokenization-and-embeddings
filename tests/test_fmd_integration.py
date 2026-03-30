"""Tests for FMD integration with embeddings."""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.fmd import FrechetMusicDistance, FMDRanking, FMDComparator
from embeddings.extractor import EmbeddingExtractor, EmbeddingAnalyzer
from utils.config import load_config


class TestFrechetMusicDistanceEnhanced:
    """Test enhanced FMD calculation."""

    @pytest.fixture
    def config(self):
        """Load configuration."""
        return load_config("configs/config.yaml")

    @pytest.fixture
    def fmd_calculator(self, config):
        """Create FMD calculator."""
        return FrechetMusicDistance(config)

    def test_fmd_identical_distributions(self, fmd_calculator):
        """FMD should be 0 for identical distributions."""
        embeddings = np.random.randn(100, 64)
        fmd = fmd_calculator.compute_fmd(embeddings, embeddings)
        assert fmd == pytest.approx(0.0, abs=1e-6)

    def test_fmd_different_distributions(self, fmd_calculator):
        """FMD should be positive for different distributions."""
        embeddings1 = np.random.randn(100, 64)
        embeddings2 = np.random.randn(100, 64) + 2
        fmd = fmd_calculator.compute_fmd(embeddings1, embeddings2)
        assert fmd > 0

    def test_fmd_symmetry(self, fmd_calculator):
        """FMD should be symmetric."""
        embeddings1 = np.random.randn(100, 64)
        embeddings2 = np.random.randn(100, 64)
        
        fmd_1_2 = fmd_calculator.compute_fmd(embeddings1, embeddings2)
        fmd_2_1 = fmd_calculator.compute_fmd(embeddings2, embeddings1)
        
        assert fmd_1_2 == pytest.approx(fmd_2_1)

    def test_fmd_1d_embeddings(self, fmd_calculator):
        """FMD should handle 1D embeddings."""
        embeddings1 = np.random.randn(64)
        embeddings2 = np.random.randn(64)
        
        fmd = fmd_calculator.compute_fmd(embeddings1, embeddings2)
        assert isinstance(fmd, float)
        assert fmd >= 0

    def test_fmd_large_distance(self, fmd_calculator):
        """FMD should be larger for more different distributions."""
        embeddings1 = np.zeros((100, 64))
        embeddings2_small = np.random.randn(100, 64) * 0.1
        embeddings2_large = np.random.randn(100, 64) * 10
        
        fmd_small = fmd_calculator.compute_fmd(embeddings1, embeddings2_small)
        fmd_large = fmd_calculator.compute_fmd(embeddings1, embeddings2_large)
        
        assert fmd_large > fmd_small

    def test_fmd_dimension_mismatch(self, fmd_calculator):
        """FMD should raise error for mismatched dimensions."""
        embeddings1 = np.random.randn(100, 64)
        embeddings2 = np.random.randn(100, 128)
        
        with pytest.raises(ValueError):
            fmd_calculator.compute_fmd(embeddings1, embeddings2)

    def test_fmd_matrix_method(self, fmd_calculator):
        """Test matrix-based FMD calculation."""
        embeddings1 = np.random.randn(50, 64)
        embeddings2 = np.random.randn(50, 64)
        
        fmd = fmd_calculator.compute_fmd_matrix(embeddings1, embeddings2)
        assert isinstance(fmd, float)
        assert fmd >= 0

    def test_batch_fmd_computation(self, fmd_calculator):
        """Test batch FMD computation."""
        embeddings_list = [
            ("dataset1", np.random.randn(100, 64)),
            ("dataset2", np.random.randn(100, 64)),
            ("dataset3", np.random.randn(100, 64)),
            ("dataset4", np.random.randn(100, 64)),
        ]
        
        result = fmd_calculator.compute_batch_fmd(embeddings_list)
        
        assert "fmd_matrix" in result
        assert "names" in result
        assert result["names"] == ["dataset1", "dataset2", "dataset3", "dataset4"]
        assert result["fmd_matrix"].shape == (4, 4)
        assert "mean_fmd" in result
        assert "median_fmd" in result
        assert "std_fmd" in result
        
        # Check diagonal is zero
        assert np.allclose(np.diag(result["fmd_matrix"]), 0)


class TestFMDRankingEnhanced:
    """Test FMD ranking functionality."""

    def test_ranking_by_fmd(self):
        """Test ranking by FMD distances."""
        fmd_matrix = np.array(
            [[0.0, 1.0, 2.0, 0.5], 
             [1.0, 0.0, 1.5, 0.8], 
             [2.0, 1.5, 0.0, 2.5], 
             [0.5, 0.8, 2.5, 0.0]]
        )
        
        ranking = FMDRanking.rank_by_fmd(fmd_matrix, reference_idx=0)
        
        assert "ranking" in ranking
        assert "distances" in ranking
        assert len(ranking["ranking"]) == 4
        # First element should be reference (index 0)
        assert ranking["ranking"][0] == 0

    def test_ranking_stability_identical(self):
        """Test ranking stability for identical rankings."""
        rankings = {
            "config1": np.array([0, 1, 2, 3]),
            "config2": np.array([0, 1, 2, 3]),
            "config3": np.array([0, 1, 2, 3]),
        }
        
        stability = FMDRanking.compute_ranking_stability(rankings)
        assert stability == pytest.approx(1.0)

    def test_ranking_stability_different(self):
        """Test ranking stability for different rankings."""
        rankings = {
            "config1": np.array([0, 1, 2, 3]),
            "config2": np.array([3, 2, 1, 0]),
        }
        
        stability = FMDRanking.compute_ranking_stability(rankings)
        assert 0 <= stability <= 1
        assert stability < 1.0  # Should be less than identical

    def test_ranking_stability_single(self):
        """Test ranking stability with single ranking."""
        rankings = {"config1": np.array([0, 1, 2, 3])}
        
        stability = FMDRanking.compute_ranking_stability(rankings)
        assert stability == 1.0


class TestFMDComparator:
    """Test FMD comparator."""

    @pytest.fixture
    def config(self):
        """Load configuration."""
        return load_config("configs/config.yaml")

    @pytest.fixture
    def comparator(self, config):
        """Create FMD comparator."""
        return FMDComparator(config)

    def test_comparator_initialization(self, comparator):
        """Test comparator initialization."""
        assert comparator.fmd_calc is not None

    def test_compare_models(self, comparator):
        """Test comparing FMD across models."""
        embeddings_by_model = {
            "CLaMP-1": np.random.randn(100, 512),
            "CLaMP-2": np.random.randn(100, 512),
        }
        
        results = comparator.compare_models(embeddings_by_model)
        
        assert "CLaMP-1_vs_CLaMP-2" in results
        assert isinstance(results["CLaMP-1_vs_CLaMP-2"], float)


class TestEmbeddingFMDIntegration:
    """Test integration of embeddings with FMD."""

    @pytest.fixture
    def config(self):
        """Load configuration."""
        config = load_config("configs/config.yaml")
        config["embeddings"]["cache_embeddings"] = False  # Disable cache for tests
        return config

    def test_extract_and_compute_fmd(self, config):
        """Test extracting embeddings and computing FMD."""
        # Create sample token sequences
        token_sequences = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
        ]
        
        # Extract embeddings
        extractor = EmbeddingExtractor(config)
        embeddings = extractor.extract_embeddings(token_sequences, "CLaMP-2")
        
        # Compute FMD
        fmd_calc = FrechetMusicDistance(config)
        fmd = fmd_calc.compute_fmd(embeddings, embeddings)
        
        assert fmd == pytest.approx(0.0, abs=1e-6)

    def test_embedding_statistics_and_fmd(self, config):
        """Test embedding statistics followed by FMD."""
        embeddings = np.random.randn(100, 64)
        
        # Compute statistics
        stats = EmbeddingAnalyzer.compute_statistics(embeddings)
        
        assert "mean" in stats
        assert "std" in stats
        assert "cov" in stats
        
        # Compute FMD with same embeddings
        fmd_calc = FrechetMusicDistance(config)
        fmd = fmd_calc.compute_fmd(embeddings, embeddings)
        
        assert fmd == pytest.approx(0.0, abs=1e-6)

    def test_fmd_pipeline_multiple_datasets(self, config):
        """Test complete FMD pipeline with multiple datasets."""
        extractor = EmbeddingExtractor(config)
        fmd_calc = FrechetMusicDistance(config)
        
        # Create sample datasets
        token_sequences_list = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
        ]
        
        embeddings_list = []
        for i, tokens in enumerate(token_sequences_list):
            emb = extractor.extract_embeddings([tokens], "CLaMP-2")[0]
            embeddings_list.append((f"dataset_{i}", emb))
        
        # Compute pairwise FMD
        result = fmd_calc.compute_batch_fmd(embeddings_list)
        
        assert result["fmd_matrix"].shape == (3, 3)
        assert len(result["names"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

