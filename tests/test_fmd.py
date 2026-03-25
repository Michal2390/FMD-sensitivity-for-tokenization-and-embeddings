"""Tests for FMD calculation."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.fmd import FrechetMusicDistance, FMDRanking


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "fmd_metric": {"use_mean": True, "use_std": True, "metric_name": "Frechet Music Distance"}
    }


@pytest.fixture
def fmd_calculator(config):
    """Create FMD calculator."""
    return FrechetMusicDistance(config)


class TestFrechetMusicDistance:
    """Test Frechet Music Distance calculation."""

    def test_fmd_identical_distributions(self, fmd_calculator):
        """FMD should be 0 for identical distributions."""
        embeddings = np.random.randn(100, 64)
        fmd = fmd_calculator.compute_fmd(embeddings, embeddings)

        assert fmd == 0.0, "FMD should be 0 for identical distributions"

    def test_fmd_different_distributions(self, fmd_calculator):
        """FMD should be positive for different distributions."""
        embeddings1 = np.random.randn(100, 64)
        embeddings2 = np.random.randn(100, 64) + 2  # Shifted distribution

        fmd = fmd_calculator.compute_fmd(embeddings1, embeddings2)

        assert fmd > 0, "FMD should be positive for different distributions"

    def test_fmd_symmetry(self, fmd_calculator):
        """FMD should be symmetric."""
        embeddings1 = np.random.randn(100, 64)
        embeddings2 = np.random.randn(100, 64)

        fmd_1_2 = fmd_calculator.compute_fmd(embeddings1, embeddings2)
        fmd_2_1 = fmd_calculator.compute_fmd(embeddings2, embeddings1)

        assert np.isclose(fmd_1_2, fmd_2_1), "FMD should be symmetric"

    def test_fmd_1d_embeddings(self, fmd_calculator):
        """FMD should handle 1D embeddings."""
        embeddings1 = np.random.randn(64)
        embeddings2 = np.random.randn(64)

        fmd = fmd_calculator.compute_fmd(embeddings1, embeddings2)

        assert isinstance(fmd, float), "FMD should return a float"
        assert fmd >= 0, "FMD should be non-negative"

    def test_fmd_matrix(self, fmd_calculator):
        """Test matrix-based FMD calculation."""
        embeddings1 = np.random.randn(50, 64)
        embeddings2 = np.random.randn(50, 64)

        fmd = fmd_calculator.compute_fmd_matrix(embeddings1, embeddings2)

        assert isinstance(fmd, float), "FMD should return a float"
        assert fmd >= 0, "FMD should be non-negative"


class TestFMDRanking:
    """Test FMD ranking functionality."""

    def test_ranking_by_fmd(self):
        """Test ranking by FMD distances."""
        # Create a mock FMD matrix
        fmd_matrix = np.array(
            [[0.0, 1.0, 2.0, 0.5], [1.0, 0.0, 1.5, 0.8], [2.0, 1.5, 0.0, 2.5], [0.5, 0.8, 2.5, 0.0]]
        )

        ranking = FMDRanking.rank_by_fmd(fmd_matrix, reference_idx=0)

        assert "ranking" in ranking
        assert "distances" in ranking
        assert len(ranking["ranking"]) == 4

    def test_ranking_stability(self):
        """Test ranking stability computation."""
        rankings = {
            "config1": np.array([0, 1, 2, 3]),
            "config2": np.array([0, 1, 2, 3]),
            "config3": np.array([0, 1, 3, 2]),
        }

        stability = FMDRanking.compute_ranking_stability(rankings)

        assert 0 <= stability <= 1, "Stability should be between 0 and 1"
        assert stability > 0.5, "Stability should be high for similar rankings"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
