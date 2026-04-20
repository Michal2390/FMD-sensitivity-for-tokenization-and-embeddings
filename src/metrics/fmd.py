"""Frechet Music Distance metric implementation.

Uses the standard Fréchet distance formula:
    FMD = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^½)

Reference: Dowson & Landau (1982), adapted for music embeddings
following Heusel et al. (2017) FID formulation.
"""

from typing import Dict, List, Tuple, Optional
from loguru import logger
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from pathlib import Path
import json


class FrechetMusicDistance:
    """
    Frechet Music Distance metric for comparing music embeddings.

    Implements the standard Fréchet distance between two multivariate
    Gaussians fitted to embedding distributions:

        FMD = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^½)

    This is mathematically equivalent to the Wasserstein-2 distance
    between two Gaussians, and is the same formula used in FID/FAD.
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
        self.epsilon = float(config["fmd_metric"].get("regularization_eps", 1e-4))
        logger.info("FrechetMusicDistance initialized (standard Fréchet formula)")

    def compute_fmd(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """
        Compute Frechet Music Distance between two sets of embeddings.

        Uses the standard Fréchet distance formula:
            FMD = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^½)

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

        d = embeddings1.shape[1]

        # Compute statistics for both distributions
        mean1 = np.mean(embeddings1, axis=0)
        mean2 = np.mean(embeddings2, axis=0)

        # Compute covariance matrices
        if embeddings1.shape[0] > 1:
            cov1 = np.cov(embeddings1.T)
        else:
            cov1 = np.zeros((d, d))

        if embeddings2.shape[0] > 1:
            cov2 = np.cov(embeddings2.T)
        else:
            cov2 = np.zeros((d, d))

        # Handle 1D case where cov returns a scalar
        if d == 1:
            cov1 = np.array([[float(cov1)]])
            cov2 = np.array([[float(cov2)]])

        # Component 1: Squared L2 distance between means
        mean_diff_sq = np.sum((mean1 - mean2) ** 2)

        # Component 2: Covariance trace term
        # Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^½)
        # Add small regularization for numerical stability
        cov1_reg = cov1 + self.epsilon * np.eye(d)
        cov2_reg = cov2 + self.epsilon * np.eye(d)

        try:
            # Compute (Σ₁ · Σ₂)^½ using scipy matrix square root
            product = cov1_reg @ cov2_reg
            sqrt_product = sqrtm(product)

            # sqrtm may return complex values due to numerical issues
            # Take real part if imaginary components are negligible
            if np.iscomplexobj(sqrt_product):
                imag_norm = np.linalg.norm(sqrt_product.imag)
                real_norm = np.linalg.norm(sqrt_product.real)
                if imag_norm / max(real_norm, 1e-10) > 0.01:
                    logger.warning(
                        f"sqrtm has non-negligible imaginary part "
                        f"(imag/real ratio: {imag_norm/max(real_norm, 1e-10):.4f}). "
                        f"Falling back to eigenvalue decomposition."
                    )
                    # Fallback: eigenvalue-based computation
                    cov_trace = self._cov_trace_eigenvalue(cov1_reg, cov2_reg)
                else:
                    sqrt_product = sqrt_product.real
                    cov_trace = float(
                        np.trace(cov1_reg) + np.trace(cov2_reg) - 2.0 * np.trace(sqrt_product)
                    )
            else:
                cov_trace = float(
                    np.trace(cov1_reg) + np.trace(cov2_reg) - 2.0 * np.trace(sqrt_product)
                )
        except Exception as e:
            logger.warning(f"sqrtm failed: {e}. Using eigenvalue fallback.")
            cov_trace = self._cov_trace_eigenvalue(cov1_reg, cov2_reg)

        # Ensure non-negative (can be slightly negative due to numerics)
        cov_trace = max(0.0, cov_trace)

        fmd = float(mean_diff_sq + cov_trace)

        logger.debug(
            f"FMD components: mean_diff²={mean_diff_sq:.6f}, "
            f"cov_trace={cov_trace:.6f}, total={fmd:.6f}"
        )

        return fmd

    def compute_fmd_components(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> Dict[str, float]:
        """Compute FMD and return all intermediate components.

        Returns dict with keys: fmd, mean_diff_sq, cov_trace,
        trace_cov1, trace_cov2, mean_norm1, mean_norm2.
        """
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)

        if embeddings1.shape[1] != embeddings2.shape[1]:
            raise ValueError(
                f"Embedding dimensions must match: {embeddings1.shape[1]} vs {embeddings2.shape[1]}"
            )

        d = embeddings1.shape[1]
        mean1 = np.mean(embeddings1, axis=0)
        mean2 = np.mean(embeddings2, axis=0)

        if embeddings1.shape[0] > 1:
            cov1 = np.cov(embeddings1.T)
        else:
            cov1 = np.zeros((d, d))
        if embeddings2.shape[0] > 1:
            cov2 = np.cov(embeddings2.T)
        else:
            cov2 = np.zeros((d, d))

        if d == 1:
            cov1 = np.array([[float(cov1)]])
            cov2 = np.array([[float(cov2)]])

        mean_diff_sq = float(np.sum((mean1 - mean2) ** 2))
        trace_cov1 = float(np.trace(cov1))
        trace_cov2 = float(np.trace(cov2))
        mean_norm1 = float(np.linalg.norm(mean1))
        mean_norm2 = float(np.linalg.norm(mean2))

        cov1_reg = cov1 + self.epsilon * np.eye(d)
        cov2_reg = cov2 + self.epsilon * np.eye(d)

        try:
            product = cov1_reg @ cov2_reg
            sqrt_product = sqrtm(product)
            if np.iscomplexobj(sqrt_product):
                imag_norm = np.linalg.norm(sqrt_product.imag)
                real_norm = np.linalg.norm(sqrt_product.real)
                if imag_norm / max(real_norm, 1e-10) > 0.01:
                    cov_trace = self._cov_trace_eigenvalue(cov1_reg, cov2_reg)
                else:
                    sqrt_product = sqrt_product.real
                    cov_trace = float(
                        np.trace(cov1_reg) + np.trace(cov2_reg) - 2.0 * np.trace(sqrt_product)
                    )
            else:
                cov_trace = float(
                    np.trace(cov1_reg) + np.trace(cov2_reg) - 2.0 * np.trace(sqrt_product)
                )
        except Exception:
            cov_trace = self._cov_trace_eigenvalue(cov1_reg, cov2_reg)

        cov_trace = max(0.0, cov_trace)
        fmd = float(mean_diff_sq + cov_trace)

        return {
            "fmd": fmd,
            "mean_diff_sq": mean_diff_sq,
            "cov_trace": cov_trace,
            "trace_cov1": trace_cov1,
            "trace_cov2": trace_cov2,
            "mean_norm1": mean_norm1,
            "mean_norm2": mean_norm2,
        }

    @staticmethod
    def _cov_trace_eigenvalue(cov1: np.ndarray, cov2: np.ndarray) -> float:
        """Fallback: compute Tr(Σ₁+Σ₂-2·(Σ₁Σ₂)^½) via eigenvalue decomposition.

        Uses: Tr((Σ₁Σ₂)^½) = Σᵢ √(λᵢ) where λᵢ are eigenvalues of Σ₁Σ₂.
        Handles NaN/Inf by replacing them with 0 before eigendecomposition.
        """
        product = cov1 @ cov2
        # Sanitize NaN/Inf values that can occur with ill-conditioned matrices
        if not np.all(np.isfinite(product)):
            logger.warning("NaN/Inf in covariance product matrix; replacing with zeros")
            product = np.nan_to_num(product, nan=0.0, posinf=0.0, neginf=0.0)
        eigvals = np.linalg.eigvals(product)
        # Eigenvalues should be non-negative for PSD matrices
        eigvals = np.real(eigvals)
        eigvals = np.maximum(eigvals, 0.0)
        trace_sqrt = float(np.sum(np.sqrt(eigvals)))
        return float(np.trace(cov1) + np.trace(cov2) - 2.0 * trace_sqrt)

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


class NormalizedFMD:
    """Normalized Frechet Music Distance — scale-invariant across embedding models.

    Raw FMD is sensitive to the absolute scale of embeddings: models that
    produce higher-norm vectors yield systematically larger FMD values,
    making cross-model comparison meaningless.  NormalizedFMD addresses
    this with three complementary strategies:

    1. **trace** (default):  nFMD = FMD / (Tr(Σ₁) + Tr(Σ₂))
       Normalises by total embedding variance.  Elegant because the
       covariance traces are already computed inside the Fréchet formula.

    2. **norm**:  nFMD = FMD / (‖μ₁‖ + ‖μ₂‖)²
       Normalises by the squared sum of mean norms, compensating for the
       quadratic scaling of mean-difference in the Fréchet distance.

    3. **zscore**:  nFMD = (FMD − μ_baseline) / σ_baseline
       Calibrates each model against a within-genre baseline (same-genre
       split).  Call ``compute_baseline()`` first to establish μ/σ.

    Usage::

        nfmd = NormalizedFMD(config, method="trace")
        value = nfmd.compute(emb_a, emb_b)  # normalised scalar

        # For z-score: first calibrate, then compute
        nfmd_z = NormalizedFMD(config, method="zscore")
        nfmd_z.compute_baseline(emb_same_genre)
        value = nfmd_z.compute(emb_a, emb_b)
    """

    METHODS = ("trace", "norm", "zscore")

    def __init__(self, config: Dict, method: str = "trace"):
        if method not in self.METHODS:
            raise ValueError(f"Unknown normalisation method '{method}'. Choose from {self.METHODS}")
        self.method = method
        self.fmd_calc = FrechetMusicDistance(config)
        # z-score baseline statistics (set via compute_baseline)
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None
        logger.info(f"NormalizedFMD initialised (method={method})")

    # ── baseline calibration (z-score only) ───────────────────────────

    def compute_baseline(
        self,
        embeddings: np.ndarray,
        n_splits: int = 20,
        seed: int = 42,
    ) -> Tuple[float, float]:
        """Compute within-distribution FMD baseline via random splits.

        Splits *embeddings* (a single genre / single condition) into two
        halves repeatedly and records the FMD of each split.  The mean
        and std of these values serve as the z-score reference.

        Args:
            embeddings: (N, D) array — all embeddings from one class.
            n_splits: Number of random 50/50 splits.
            seed: Random seed for reproducibility.

        Returns:
            (baseline_mean, baseline_std)
        """
        rng = np.random.default_rng(seed)
        n = embeddings.shape[0]
        if n < 10:
            logger.warning("Too few samples for baseline; using (0, 1)")
            self._baseline_mean = 0.0
            self._baseline_std = 1.0
            return (0.0, 1.0)

        fmd_values: List[float] = []
        half = n // 2
        for _ in range(n_splits):
            idx = rng.permutation(n)
            a, b = embeddings[idx[:half]], embeddings[idx[half : half * 2]]
            fmd_values.append(self.fmd_calc.compute_fmd(a, b))

        self._baseline_mean = float(np.mean(fmd_values))
        self._baseline_std = float(np.std(fmd_values))
        if self._baseline_std < 1e-12:
            self._baseline_std = 1.0  # avoid division by zero

        logger.info(
            f"Baseline computed: μ={self._baseline_mean:.6f}, "
            f"σ={self._baseline_std:.6f} ({n_splits} splits, n={n})"
        )
        return (self._baseline_mean, self._baseline_std)

    # ── main entry point ──────────────────────────────────────────────

    def compute(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """Compute normalised FMD between two embedding sets.

        Args:
            embeddings1: (N, D) first set.
            embeddings2: (M, D) second set.

        Returns:
            Normalised FMD value (float).
        """
        components = self.fmd_calc.compute_fmd_components(embeddings1, embeddings2)
        raw_fmd = components["fmd"]

        if self.method == "trace":
            denom = components["trace_cov1"] + components["trace_cov2"]
            if denom < 1e-12:
                return 0.0
            return raw_fmd / denom

        elif self.method == "norm":
            sum_norms = components["mean_norm1"] + components["mean_norm2"]
            if sum_norms < 1e-12:
                return 0.0
            return raw_fmd / (sum_norms ** 2)

        elif self.method == "zscore":
            if self._baseline_mean is None:
                raise RuntimeError(
                    "Z-score normalisation requires a baseline. "
                    "Call compute_baseline() first."
                )
            return (raw_fmd - self._baseline_mean) / self._baseline_std

        return raw_fmd  # unreachable

    def compute_all_methods(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> Dict[str, float]:
        """Compute raw FMD and all normalisation variants at once.

        Returns dict with keys: fmd, nfmd_trace, nfmd_norm, nfmd_zscore
        (zscore is None if no baseline has been set).
        """
        components = self.fmd_calc.compute_fmd_components(embeddings1, embeddings2)
        raw = components["fmd"]

        # trace
        denom_trace = components["trace_cov1"] + components["trace_cov2"]
        nfmd_trace = raw / denom_trace if denom_trace > 1e-12 else 0.0

        # norm
        sum_norms = components["mean_norm1"] + components["mean_norm2"]
        nfmd_norm = raw / (sum_norms ** 2) if sum_norms > 1e-12 else 0.0

        # zscore
        nfmd_zscore = None
        if self._baseline_mean is not None:
            nfmd_zscore = (raw - self._baseline_mean) / self._baseline_std

        return {
            "fmd": raw,
            "nfmd_trace": nfmd_trace,
            "nfmd_norm": nfmd_norm,
            "nfmd_zscore": nfmd_zscore,
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
