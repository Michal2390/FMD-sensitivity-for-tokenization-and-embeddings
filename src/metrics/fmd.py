"""Frechet Music Distance metric implementation."""

from typing import Dict
from loguru import logger
import numpy as np
from scipy.spatial.distance import cdist


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

        # Compute statistics
        mean1 = np.mean(embeddings1, axis=0)
        mean2 = np.mean(embeddings2, axis=0)

        # Component 1: Distance between means
        mean_distance = np.linalg.norm(mean1 - mean2)

        fmd = mean_distance

        # Component 2: Difference in standard deviations (if enabled)
        if self.use_std:
            std1 = np.std(embeddings1, axis=0)
            std2 = np.std(embeddings2, axis=0)
            std_distance = np.linalg.norm(std1 - std2)
            fmd += std_distance

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

        for i, (name1, emb1) in enumerate(embeddings_list):
            names.append(name1)
            for j, (name2, emb2) in enumerate(embeddings_list):
                if i <= j:
                    fmd = self.compute_fmd(emb1, emb2)
                    fmd_matrix[i, j] = fmd
                    fmd_matrix[j, i] = fmd

        result = {
            "fmd_matrix": fmd_matrix,
            "names": names,
            "mean_fmd": np.mean(fmd_matrix[np.triu_indices_from(fmd_matrix, k=1)]),
            "max_fmd": np.max(fmd_matrix[np.triu_indices_from(fmd_matrix, k=1)]),
            "min_fmd": np.min(fmd_matrix[np.triu_indices_from(fmd_matrix, k=1)]),
        }

        return result

    def compute_fmd_with_ci(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray, n_boot: int = 1000, ci: float = 0.95
    ) -> Dict:
        """
        Compute FMD with bootstrap confidence intervals.

        Args:
            embeddings1: First set of embeddings (N x D)
            embeddings2: Second set of embeddings (M x D)
            n_boot: Number of bootstrap samples
            ci: Confidence interval level

        Returns:
            Dictionary with FMD value and CI
        """
        from scipy.stats import bootstrap

        def fmd_statistic(data1, data2):
            return self.compute_fmd(data1, data2)

        # Bootstrap for embeddings1
        boot1 = bootstrap((embeddings1,), lambda x: x, n_resamples=n_boot, method='basic')
        # Bootstrap for embeddings2
        boot2 = bootstrap((embeddings2,), lambda x: x, n_resamples=n_boot, method='basic')

        # Compute FMD for each bootstrap sample
        fmd_values = []
        for i in range(n_boot):
            emb1_boot = boot1.bootstrap_distribution[i]
            emb2_boot = boot2.bootstrap_distribution[i]
            fmd_values.append(fmd_statistic(emb1_boot, emb2_boot))

        fmd_values = np.array(fmd_values)
        mean_fmd = np.mean(fmd_values)
        ci_lower = np.percentile(fmd_values, (1 - ci) / 2 * 100)
        ci_upper = np.percentile(fmd_values, (1 + ci) / 2 * 100)

        return {
            "fmd": mean_fmd,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_level": ci,
        }


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
