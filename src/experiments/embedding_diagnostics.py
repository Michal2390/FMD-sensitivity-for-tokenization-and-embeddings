"""Embedding and token-level diagnostics for explaining FMD sensitivity.

Implements:
- PCA / t-SNE visualisation of embeddings per genre × variant
- Intra- vs inter-genre cosine similarity per variant
- Token-level statistics (sequence length, entropy, type distribution)
- Feature importance (per-dimension variance across variants)
- Correlation between token statistics and FMD values
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats

# Optional imports for dimensionality reduction
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    logger.warning("scikit-learn not available – PCA/t-SNE diagnostics disabled")


# ======================================================================
# Token-level statistics
# ======================================================================

def token_sequence_stats(tokens: List[int]) -> Dict[str, float]:
    """Compute descriptive statistics for a single token sequence."""
    n = len(tokens)
    if n == 0:
        return {"length": 0, "entropy": 0.0, "unique_ratio": 0.0}

    counts = Counter(tokens)
    probs = np.array(list(counts.values()), dtype=float) / n
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    unique_ratio = len(counts) / n

    return {
        "length": n,
        "entropy": entropy,
        "unique_ratio": unique_ratio,
        "n_unique": len(counts),
    }


def aggregate_token_stats(
    token_sequences: List[List[int]],
) -> Dict[str, float]:
    """Aggregate token statistics across many sequences."""
    all_stats = [token_sequence_stats(seq) for seq in token_sequences]
    if not all_stats:
        return {}
    df = pd.DataFrame(all_stats)
    result = {}
    for col in df.columns:
        result[f"{col}_mean"] = float(df[col].mean())
        result[f"{col}_std"] = float(df[col].std())
    return result


# ======================================================================
# Cosine similarity analysis
# ======================================================================

def _cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix (N×N)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed = embeddings / norms
    return normed @ normed.T


def intra_inter_cosine(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> Dict[str, float]:
    """Compute mean intra-genre and inter-genre cosine similarities.

    Parameters
    ----------
    emb_a : array (N, D) – embeddings for genre A
    emb_b : array (M, D) – embeddings for genre B

    Returns
    -------
    Dict with ``intra_a``, ``intra_b``, ``inter``, ``separation_gap``.
    """
    sim_a = _cosine_sim_matrix(emb_a)
    sim_b = _cosine_sim_matrix(emb_b)

    # Upper triangle (excluding diagonal) means
    n_a, n_b = sim_a.shape[0], sim_b.shape[0]
    intra_a = float(sim_a[np.triu_indices(n_a, k=1)].mean()) if n_a > 1 else 1.0
    intra_b = float(sim_b[np.triu_indices(n_b, k=1)].mean()) if n_b > 1 else 1.0

    # Inter-genre
    norms_a = np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-12
    norms_b = np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-12
    inter_sim = (emb_a / norms_a) @ (emb_b / norms_b).T
    inter = float(inter_sim.mean())

    avg_intra = (intra_a + intra_b) / 2.0
    separation = avg_intra - inter

    return {
        "intra_a": intra_a,
        "intra_b": intra_b,
        "inter": inter,
        "separation_gap": separation,
    }


# ======================================================================
# Dimensionality reduction
# ======================================================================

def compute_pca(
    embeddings: np.ndarray,
    n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run PCA. Returns (projected_data, explained_variance_ratio)."""
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required for PCA")
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(embeddings)
    return projected, pca.explained_variance_ratio_


def compute_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    seed: int = 42,
) -> np.ndarray:
    """Run t-SNE. Returns projected data."""
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required for t-SNE")
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, max(5.0, embeddings.shape[0] - 1)),
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


# ======================================================================
# Feature importance (per-dimension variance)
# ======================================================================

def per_dimension_variance(
    embeddings_by_variant: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute per-dimension variance across variant means.

    Parameters
    ----------
    embeddings_by_variant : {variant_name: array (N, D)}

    Returns
    -------
    array of shape (D,) – variance of each dimension's mean across variants.
    Only considers variants sharing the most common dimensionality.
    """
    # Group by dimension to avoid inhomogeneous array
    dim_groups: Dict[int, List[np.ndarray]] = {}
    for emb in embeddings_by_variant.values():
        d = emb.shape[1]
        dim_groups.setdefault(d, []).append(emb.mean(axis=0))

    # Pick the dimension group with most variants
    best_dim = max(dim_groups, key=lambda d: len(dim_groups[d]))
    means = np.array(dim_groups[best_dim])
    return means.var(axis=0)


def top_varying_dimensions(
    embeddings_by_variant: Dict[str, np.ndarray],
    top_k: int = 20,
) -> List[Tuple[int, float]]:
    """Return indices and variances of the top-K most varying embedding dims."""
    var = per_dimension_variance(embeddings_by_variant)
    indices = np.argsort(var)[::-1][:top_k]
    return [(int(i), float(var[i])) for i in indices]


# ======================================================================
# FMD ↔ token-stats correlation
# ======================================================================

def fmd_token_correlation(
    fmd_values: np.ndarray,
    token_stat_values: np.ndarray,
) -> Dict[str, float]:
    """Spearman correlation between FMD and a token-level statistic."""
    if len(fmd_values) < 3:
        return {"rho": float("nan"), "p": float("nan")}
    rho, p = sp_stats.spearmanr(fmd_values, token_stat_values)
    return {"rho": float(rho), "p": float(p)}


# ======================================================================
# High-level diagnostic runner
# ======================================================================

def run_embedding_diagnostics(
    embeddings_by_variant: Dict[str, Dict[str, np.ndarray]],
    token_sequences_by_variant: Optional[Dict[str, Dict[str, List[List[int]]]]] = None,
    fmd_per_variant: Optional[Dict[str, float]] = None,
    genre_a: str = "rock",
    genre_b: str = "classical",
    output_dir: Path | str = "results/reports/lakh",
    seed: int = 42,
) -> Dict[str, any]:
    """Run full embedding diagnostics suite.

    Parameters
    ----------
    embeddings_by_variant :
        ``{variant_name: {genre: array (N, D)}}``
    token_sequences_by_variant :
        ``{variant_name: {genre: [[int, …], …]}}`` (optional)
    fmd_per_variant :
        ``{variant_name: float}`` (optional, for correlation)
    genre_a, genre_b : genre labels
    output_dir : where to save CSV/JSON artefacts
    seed : random seed for t-SNE

    Returns
    -------
    Dict with paths to generated artefacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. Cosine similarity intra/inter per variant
    # ------------------------------------------------------------------
    cosine_rows: List[Dict] = []
    for variant_name, genre_embs in embeddings_by_variant.items():
        emb_a = genre_embs.get(genre_a)
        emb_b = genre_embs.get(genre_b)
        if emb_a is None or emb_b is None:
            continue
        cosine_result = intra_inter_cosine(emb_a, emb_b)
        cosine_rows.append({"variant": variant_name, **cosine_result})

    if cosine_rows:
        cosine_df = pd.DataFrame(cosine_rows)
        path = output_dir / "cosine_similarity.csv"
        cosine_df.to_csv(path, index=False)
        outputs["cosine_csv"] = str(path)
        logger.info(f"Cosine similarity: {len(cosine_rows)} variants → {path}")

    # ------------------------------------------------------------------
    # 2. PCA / t-SNE (per variant — safe since emb_a, emb_b share dim)
    # ------------------------------------------------------------------
    if _HAS_SKLEARN:
        tsne_records: List[Dict] = []
        pca_records: List[Dict] = []

        for variant_name, genre_embs in embeddings_by_variant.items():
            emb_a = genre_embs.get(genre_a)
            emb_b = genre_embs.get(genre_b)
            if emb_a is None or emb_b is None:
                continue
            # Skip if dimensions mismatch within a variant (shouldn't happen)
            if emb_a.shape[1] != emb_b.shape[1]:
                logger.warning(f"Dim mismatch in {variant_name}: {emb_a.shape[1]} vs {emb_b.shape[1]}")
                continue

            combined = np.vstack([emb_a, emb_b])
            labels = [genre_a] * len(emb_a) + [genre_b] * len(emb_b)

            # PCA
            try:
                pca_proj, var_ratio = compute_pca(combined, n_components=2)
                for i, (x, y) in enumerate(pca_proj):
                    pca_records.append(
                        {
                            "variant": variant_name,
                            "genre": labels[i],
                            "pc1": float(x),
                            "pc2": float(y),
                        }
                    )
            except Exception as exc:
                logger.warning(f"PCA failed for {variant_name}: {exc}")

            # t-SNE
            if combined.shape[0] >= 10:
                try:
                    tsne_proj = compute_tsne(combined, seed=seed)
                    for i, (x, y) in enumerate(tsne_proj):
                        tsne_records.append(
                            {
                                "variant": variant_name,
                                "genre": labels[i],
                                "tsne1": float(x),
                                "tsne2": float(y),
                            }
                        )
                except Exception as exc:
                    logger.warning(f"t-SNE failed for {variant_name}: {exc}")

        if pca_records:
            pca_df = pd.DataFrame(pca_records)
            path = output_dir / "pca_projections.csv"
            pca_df.to_csv(path, index=False)
            outputs["pca_csv"] = str(path)

        if tsne_records:
            tsne_df = pd.DataFrame(tsne_records)
            path = output_dir / "tsne_projections.csv"
            tsne_df.to_csv(path, index=False)
            outputs["tsne_csv"] = str(path)

    # ------------------------------------------------------------------
    # 3. Feature importance (per-dimension variance across variants)
    #    Group by embedding dimension to avoid inhomogeneous arrays.
    # ------------------------------------------------------------------
    flat_embs: Dict[str, np.ndarray] = {}
    for variant_name, genre_embs in embeddings_by_variant.items():
        parts = [v for v in genre_embs.values() if v is not None]
        if parts:
            # All parts within a variant share the same dim
            flat_embs[variant_name] = np.vstack(parts)

    if len(flat_embs) >= 2:
        # Group variants by dimension
        dim_groups: Dict[int, Dict[str, np.ndarray]] = {}
        for vname, emb in flat_embs.items():
            d = emb.shape[1]
            dim_groups.setdefault(d, {})[vname] = emb

        all_top_dims = []
        for dim, group in dim_groups.items():
            if len(group) >= 2:
                top_dims = top_varying_dimensions(group, top_k=20)
                all_top_dims.extend(
                    [{"dim": d, "variance": v, "embedding_size": dim} for d, v in top_dims]
                )

        if all_top_dims:
            path = output_dir / "top_varying_dimensions.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(all_top_dims, fh, indent=2)
            outputs["top_dims_json"] = str(path)

    # ------------------------------------------------------------------
    # 4. Token statistics
    # ------------------------------------------------------------------
    if token_sequences_by_variant:
        tok_rows: List[Dict] = []
        for variant_name, genre_seqs in token_sequences_by_variant.items():
            for genre, seqs in genre_seqs.items():
                agg = aggregate_token_stats(seqs)
                tok_rows.append({"variant": variant_name, "genre": genre, **agg})

        if tok_rows:
            tok_df = pd.DataFrame(tok_rows)
            path = output_dir / "token_statistics.csv"
            tok_df.to_csv(path, index=False)
            outputs["token_stats_csv"] = str(path)

        # ------------------------------------------------------------------
        # 5. FMD ↔ token entropy/length correlation
        # ------------------------------------------------------------------
        if fmd_per_variant and tok_rows:
            corr_rows: List[Dict] = []
            for stat_key in ("length_mean", "entropy_mean"):
                fmd_vals = []
                stat_vals = []
                for row in tok_rows:
                    vname = row["variant"]
                    if vname in fmd_per_variant and stat_key in row:
                        fmd_vals.append(fmd_per_variant[vname])
                        stat_vals.append(row[stat_key])
                if len(fmd_vals) >= 3:
                    corr = fmd_token_correlation(np.array(fmd_vals), np.array(stat_vals))
                    corr_rows.append({"statistic": stat_key, **corr})

            if corr_rows:
                path = output_dir / "fmd_token_correlations.json"
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(corr_rows, fh, indent=2)
                outputs["fmd_token_corr_json"] = str(path)

    logger.info(f"Embedding diagnostics complete → {len(outputs)} artefacts in {output_dir}")
    return outputs





