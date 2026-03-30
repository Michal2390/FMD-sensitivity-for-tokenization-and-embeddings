"""Frechet Music Distance metric implementation."""

from typing import Dict, List, Tuple, Optional
from loguru import logger
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
import json


class FrechetMusicDistance:
    """
    Frechet Music Distance metric for comparing music embeddings.

    Based on the Frechet Distance concept adapted for music analysis,
    comparing the geometric similarity of embedding distributions.
    """

    def __init__(self, config: Dict):
        """
        Initialize Frechet Music Distance calculator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.use_mean = config["fmd_metric"].get("use_mean", True)
        self.use_std = config["fmd_metric"].get("use_std", True)
        logger.info("FrechetMusicDistance initialized")

    def compute_fmd(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """
        Compute Frechet Music Distance between two sets of embeddings.

        This implements the Frechet Distance metric adapted for music analysis:
        FMD = sqrt(max(mean_distance^2 + std_distance^2, covariance_trace_diff))

        Args:
            embeddings1: First set of embeddings (N x D)
            embeddings2: Second set of embeddings (M x D)

        Returns:
            FMD value (non-negative float)
        """
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)

        # Ensure same dimensionality
        if embeddings1.shape[1] != embeddings2.shape[1]:
            raise ValueError(
                f"Embedding dimensions must match: {embeddings1.shape[1]} vs {embeddings2.shape[1]}"
            )

        # Compute statistics for both distributions
        mean1 = np.mean(embeddings1, axis=0)
        mean2 = np.mean(embeddings2, axis=0)
        
        # Compute covariance matrices
        if embeddings1.shape[0] > 1:
            cov1 = np.cov(embeddings1.T)
        else:
            cov1 = np.zeros((embeddings1.shape[1], embeddings1.shape[1]))
        
        if embeddings2.shape[0] > 1:
            cov2 = np.cov(embeddings2.T)
        else:
            cov2 = np.zeros((embeddings2.shape[1], embeddings2.shape[1]))

        # Handle 1D case where cov returns a scalar
        if embeddings1.shape[1] == 1:
            cov1 = np.array([[cov1]])
            cov2 = np.array([[cov2]])

        # Component 1: L2 distance between means (Wasserstein part)
        mean_distance = np.linalg.norm(mean1 - mean2) ** 2

        # Component 2: Difference in covariance matrices (Frechet part)
        # Using Frobenius norm: ||sqrt(cov1) - sqrt(cov2)||_F^2
        try:
            # Compute matrix square roots
            eigvals1, eigvecs1 = np.linalg.eigh(cov1)
            eigvals2, eigvecs2 = np.linalg.eigh(cov2)
            
            # Ensure non-negative eigenvalues
            eigvals1 = np.maximum(eigvals1, 0)
            eigvals2 = np.maximum(eigvals2, 0)
            
            sqrt_cov1 = eigvecs1 @ np.diag(np.sqrt(eigvals1)) @ eigvecs1.T
            sqrt_cov2 = eigvecs2 @ np.diag(np.sqrt(eigvals2)) @ eigvecs2.T
            
            cov_distance = np.linalg.norm(sqrt_cov1 - sqrt_cov2, "fro") ** 2
        except Exception as e:
            logger.warning(f"Failed to compute covariance distance: {e}. Using trace difference.")
            # Fallback: use trace difference
            cov_distance = (np.trace(cov1) - np.trace(cov2)) ** 2

        # Component 3: Optional standard deviation difference
        std_distance = 0.0
        if self.use_std:
            std1 = np.std(embeddings1, axis=0)
            std2 = np.std(embeddings2, axis=0)
            std_distance = np.linalg.norm(std1 - std2) ** 2

        # Combine components: FMD = sqrt(mean_dist^2 + cov_dist + std_dist)
        fmd = np.sqrt(mean_distance + cov_distance + std_distance)

        logger.debug(f"FMD components: mean={np.sqrt(mean_distance):.4f}, cov={np.sqrt(cov_distance):.4f}, std={np.sqrt(std_distance):.4f}, total={fmd:.4f}")

        return float(fmd)

    def compute_fmd_matrix(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """
        Compute FMD using matrix-based approach with optimal transport.

        Args:
            embeddings1: First set of embeddings (N x D)
            embeddings2: Second set of embeddings (M x D)

        Returns:
            FMD value
        """
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)

        # Compute pairwise distances
        distances = cdist(embeddings1, embeddings2, metric="euclidean")

        # Use Hungarian algorithm approximation: minimum cost matching
        # Simplified: use minimum distance for each sample
        fwd_distances = np.min(distances, axis=1)
        bwd_distances = np.min(distances, axis=0)

        fmd = max(np.max(fwd_distances), np.max(bwd_distances))

        return float(fmd)

    def compute_batch_fmd(self, embeddings_list: list) -> Dict:
        """
        Compute pairwise FMD for multiple embedding sets.

        Args:
            embeddings_list: List of (name, embeddings) tuples

        Returns:
            Dictionary with FMD values and statistics
        """
        n = len(embeddings_list)
        fmd_matrix = np.zeros((n, n))
        names = []

        logger.info(f"Computing pairwise FMD for {n} embedding sets")

        for i, (name1, emb1) in enumerate(embeddings_list):
            names.append(name1)
            for j, (name2, emb2) in enumerate(embeddings_list):
                if i <= j:
                    try:
                        fmd = self.compute_fmd(emb1, emb2)
                        fmd_matrix[i, j] = fmd
                        fmd_matrix[j, i] = fmd
                    except Exception as e:
                        logger.error(f"Failed to compute FMD between {name1} and {name2}: {e}")
                        fmd_matrix[i, j] = np.nan
                        fmd_matrix[j, i] = np.nan

        # Get upper triangle values (excluding diagonal)
        upper_triangle = fmd_matrix[np.triu_indices_from(fmd_matrix, k=1)]
        upper_triangle = upper_triangle[~np.isnan(upper_triangle)]

        result = {
            "fmd_matrix": fmd_matrix,
            "names": names,
            "mean_fmd": float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else np.nan,
            "median_fmd": float(np.median(upper_triangle)) if len(upper_triangle) > 0 else np.nan,
            "std_fmd": float(np.std(upper_triangle)) if len(upper_triangle) > 0 else np.nan,
            "max_fmd": float(np.max(upper_triangle)) if len(upper_triangle) > 0 else np.nan,
            "min_fmd": float(np.min(upper_triangle)) if len(upper_triangle) > 0 else np.nan,
        }

        logger.info(f"Batch FMD computation complete: mean={result['mean_fmd']:.4f}, std={result['std_fmd']:.4f}")

        return result


class FMDRanking:
    """Ranking analysis based on FMD values."""

    @staticmethod
    def rank_by_fmd(fmd_matrix: np.ndarray, reference_idx: int) -> Dict:
        """
        Rank datasets by FMD distance from a reference dataset.

        Args:
            fmd_matrix: FMD distance matrix
            reference_idx: Index of reference dataset

        Returns:
            Ranking dictionary
        """
        distances = fmd_matrix[reference_idx, :]
        ranking = np.argsort(distances)

        result = {
            "ranking": ranking,
            "distances": distances[ranking],
        }

        return result

    @staticmethod
    def compute_ranking_stability(rankings_dict: Dict) -> float:
        """
        Compute stability of rankings across different experiments.

        Args:
            rankings_dict: Dictionary with rankings from different experiments

        Returns:
            Stability score (0-1, higher is more stable)
        """
        if len(rankings_dict) < 2:
            return 1.0

        ranking_list = list(rankings_dict.values())
        n_items = len(ranking_list[0])

        # Compute Spearman correlation between all pairs
        correlations = []
        for i in range(len(ranking_list)):
            for j in range(i + 1, len(ranking_list)):
                # Simplified: count how many pairs are in same relative order
                rank1 = ranking_list[i]
                rank2 = ranking_list[j]

                # Compute Kendall tau distance
                disagreements = 0
                for k1 in range(n_items):
                    for k2 in range(k1 + 1, n_items):
                        order1 = rank1[k1] < rank1[k2]
                        order2 = rank2[k1] < rank2[k2]
                        if order1 != order2:
                            disagreements += 1

                # Normalize to 0-1
                max_disagreements = n_items * (n_items - 1) / 2
                correlation = (
                    1 - (disagreements / max_disagreements) if max_disagreements > 0 else 1.0
                )
                correlations.append(correlation)

        stability = np.mean(correlations) if correlations else 1.0
        return float(stability)


class FMDComparator:
    """Compare FMD values across different configurations."""

    def __init__(self, config: Dict):
        """
        Initialize FMD comparator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fmd_calc = FrechetMusicDistance(config)

    def compare_tokenizers(
        self, embeddings_by_tokenizer: Dict[str, np.ndarray], dataset1_name: str, dataset2_name: str
    ) -> Dict:
        """
        Compare FMD values across different tokenizers.

        Args:
            embeddings_by_tokenizer: Dict mapping tokenizer name to embeddings
            dataset1_name: Name of first dataset
            dataset2_name: Name of second dataset

        Returns:
            Dictionary with comparison results
        """
        results = {}

        for tokenizer, embeddings in embeddings_by_tokenizer.items():
            # Assuming embeddings are already computed for both datasets
            # This is a simplified version - actual implementation would need
            # to properly handle dataset-specific embeddings
            fmd = self.fmd_calc.compute_fmd(embeddings, embeddings)
            results[tokenizer] = fmd

        return results

    def compare_models(self, embeddings_by_model: Dict[str, np.ndarray]) -> Dict:
        """
        Compare FMD values across different embedding models.

        Args:
            embeddings_by_model: Dict mapping model name to embeddings

        Returns:
            Dictionary with comparison results
        """
        results = {}
        model_names = list(embeddings_by_model.keys())

        for i, model1 in enumerate(model_names):
            for model2 in model_names[i + 1 :]:
                fmd = self.fmd_calc.compute_fmd(
                    embeddings_by_model[model1], embeddings_by_model[model2]
                )
                results[f"{model1}_vs_{model2}"] = fmd

        return results
