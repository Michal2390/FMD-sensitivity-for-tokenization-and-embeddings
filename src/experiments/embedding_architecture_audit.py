"""Embedding architecture audit for cross-model FMD comparability.

This module investigates whether cross-model FMD / nFMD comparisons are
scientifically interpretable. The central hypothesis is that Fréchet-style
metrics require roughly comparable embedding geometry across models; if the
embedding spaces differ strongly in scale, covariance trace, dimensionality,
or output normalisation, raw cross-model comparisons become hard to interpret.

The audit therefore focuses on three questions:
1. How different are the embedding geometries across architectures?
2. How large is the cross-genre FMD signal relative to a within-model
   split-half baseline?
3. Does nFMD reduce cross-model spread enough to be treated as a useful
   hypothesis, rather than a validated replacement for FMD?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from data.manager import DatasetManager
from embeddings.extractor import EmbeddingExtractor
from metrics.fmd import FrechetMusicDistance
from preprocessing.processor import MIDIPreprocessor
from tokenization.tokenizer import TokenizationPipeline

DEFAULT_SEED = 42
DEFAULT_MAX_FILES = 120
DEFAULT_PREPROCESS = (False, False)
DEFAULT_SHARED_ANCHOR_FRACTION = 0.5
MIN_SHARED_MATCHES = 8


@dataclass(frozen=True)
class ArchitectureMetadata:
    """Human-readable metadata for an embedding model."""

    family: str
    input_domain: str
    output_rule: str


@dataclass
class SharedSpaceCalibrator:
    """Map multiple embedding models into an exploratory shared comparison space.

    Procedure:
      1. Centre each model on matched anchor samples.
      2. Reduce to a common dimensionality with PCA/SVD.
      3. Whiten each reduced space.
      4. Align whitened anchors to a reference model with orthogonal Procrustes.

    This is intentionally exploratory: it does not prove semantic equivalence of
    spaces, but it offers a principled way to test whether cross-model spread is
    mostly geometric and whether matched-sample alignment can reduce it.
    """

    reference_model: str
    max_dim: int = 64
    eps: float = 1e-8
    target_dim: int = 0
    params: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    n_anchor_samples: int = 0

    def fit(self, anchors_by_model: Dict[str, np.ndarray]) -> Dict[str, object]:
        if self.reference_model not in anchors_by_model:
            raise ValueError(f"Reference model '{self.reference_model}' missing from anchors")
        if len(anchors_by_model) < 2:
            raise ValueError("Shared-space calibration requires at least two models")

        n_samples = min(arr.shape[0] for arr in anchors_by_model.values())
        min_dim = min(arr.shape[1] for arr in anchors_by_model.values())
        if n_samples < 4:
            raise ValueError("Too few matched samples for shared-space calibration")

        self.target_dim = max(1, min(self.max_dim, min_dim, n_samples - 1))
        self.n_anchor_samples = int(n_samples)
        self.params = {}

        whitened_cache: Dict[str, np.ndarray] = {}
        for model_name, anchor in anchors_by_model.items():
            x = np.asarray(anchor[:n_samples], dtype=float)
            mean = x.mean(axis=0, keepdims=True)
            centered = x - mean
            _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
            components = vt[: self.target_dim].T
            reduced = centered @ components

            variances = (singular_values[: self.target_dim] ** 2) / max(n_samples - 1, 1)
            scales = np.sqrt(np.maximum(variances, self.eps))
            whitened = reduced / scales
            whitened_cache[model_name] = whitened
            self.params[model_name] = {
                "mean": mean.astype(np.float32),
                "components": components.astype(np.float32),
                "scales": scales.astype(np.float32),
                "rotation": np.eye(self.target_dim, dtype=np.float32),
            }

        ref = whitened_cache[self.reference_model]
        for model_name, whitened in whitened_cache.items():
            if model_name == self.reference_model:
                continue
            cross = whitened.T @ ref
            u, _, vt = np.linalg.svd(cross, full_matrices=False)
            rotation = u @ vt
            self.params[model_name]["rotation"] = rotation.astype(np.float32)

        return {
            "reference_model": self.reference_model,
            "target_dim": int(self.target_dim),
            "n_anchor_samples": int(self.n_anchor_samples),
            "models": sorted(anchors_by_model),
        }

    def transform(self, model_name: str, embeddings: np.ndarray) -> np.ndarray:
        if model_name not in self.params:
            raise KeyError(f"Model '{model_name}' has not been fitted in shared space")

        x = np.asarray(embeddings, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        p = self.params[model_name]
        centered = x - p["mean"]
        reduced = centered @ p["components"]
        whitened = reduced / p["scales"]
        aligned = whitened @ p["rotation"]
        return aligned.astype(np.float32)


def get_architecture_metadata(model_name: str, model_config: Optional[Dict] = None) -> ArchitectureMetadata:
    """Infer high-level architectural properties used in the audit report."""
    format_type = (model_config or {}).get("format_type", "unknown")

    if model_name.startswith("CLaMP"):
        return ArchitectureMetadata(
            family="contrastive symbolic",
            input_domain=format_type,
            output_rule="L2-normalized projection",
        )
    if model_name.startswith("MusicBERT"):
        return ArchitectureMetadata(
            family="masked LM symbolic",
            input_domain=format_type,
            output_rule="raw [CLS] hidden state",
        )
    if model_name == "NLP-Baseline":
        return ArchitectureMetadata(
            family="general NLP control",
            input_domain=format_type,
            output_rule="raw [CLS] hidden state",
        )
    if model_name == "MERT":
        return ArchitectureMetadata(
            family="audio self-supervised",
            input_domain=format_type,
            output_rule="raw mean-pooled audio hidden state",
        )
    return ArchitectureMetadata(
        family="unknown",
        input_domain=format_type,
        output_rule="unknown",
    )


def is_symbolic_shared_space_candidate(
    model_name: str,
    model_config: Optional[Dict] = None,
    diagnostics: Optional[Dict[str, object]] = None,
) -> bool:
    """Models eligible for exploratory shared-space calibration.

    We currently restrict calibration to symbolic/text token-based encoders and
    exclude the audio-domain MERT model from the aligned comparison space.
    """
    metadata = get_architecture_metadata(model_name, model_config)
    if metadata.family == "audio self-supervised":
        return False
    if diagnostics and diagnostics.get("status") == "diagnostics_failed":
        return False
    return True


def safe_spread(values: Iterable[float], eps: float = 1e-12) -> float:
    """Return max/min spread for positive finite values."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > eps]
    if arr.size < 2:
        return 1.0
    return float(arr.max() / max(arr.min(), eps))


def split_shared_space_ids(
    common_ids: List[str],
    seed: int = DEFAULT_SEED,
    anchor_fraction: float = DEFAULT_SHARED_ANCHOR_FRACTION,
    min_matches: int = MIN_SHARED_MATCHES,
) -> Tuple[List[str], List[str], str]:
    """Split matched IDs into anchor and evaluation subsets for shared-space audit.

    Prefer a true holdout evaluation; if the matched set is too small, fall back to
    an in-sample protocol and mark it explicitly in the summary/report.
    """
    ids = list(common_ids)
    if len(ids) < min_matches:
        return [], [], "insufficient_matches"

    if len(ids) >= 2 * min_matches:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(ids))
        n_anchor = int(round(len(ids) * anchor_fraction))
        n_anchor = min(max(n_anchor, min_matches), len(ids) - min_matches)
        anchor_ids = [ids[i] for i in perm[:n_anchor]]
        eval_ids = [ids[i] for i in perm[n_anchor:]]
        return sorted(anchor_ids), sorted(eval_ids), "holdout"

    return sorted(ids), sorted(ids), "in_sample_fallback"


def effective_dimensionality(embeddings: np.ndarray, threshold: float = 0.90) -> int:
    """Number of principal directions needed to explain a variance threshold."""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.shape[0] < 2:
        return min(1, embeddings.shape[1])

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    power = singular_values ** 2
    total_power = float(power.sum())
    if total_power <= 1e-12:
        return 1

    cumulative = np.cumsum(power / total_power)
    return int(np.searchsorted(cumulative, threshold) + 1)


def mean_pairwise_cosine(embeddings: np.ndarray, max_samples: int = 500, seed: int = DEFAULT_SEED) -> float:
    """Average upper-triangular cosine similarity inside one embedding cloud."""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.shape[0] < 2:
        return 1.0

    rng = np.random.default_rng(seed)
    if embeddings.shape[0] > max_samples:
        idx = rng.choice(embeddings.shape[0], size=max_samples, replace=False)
        embeddings = embeddings[idx]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normalized = embeddings / norms
    sim = normalized @ normalized.T
    tri = np.triu_indices_from(sim, k=1)
    return float(sim[tri].mean()) if tri[0].size else 1.0


def summarize_embedding_distribution(
    embeddings: np.ndarray,
    fmd_calc: FrechetMusicDistance,
    token_lengths: Optional[List[int]] = None,
    baseline_splits: int = 12,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """Describe one embedding distribution and its intrinsic split-half baseline."""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    cov = fmd_calc._estimate_covariance(embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    baseline = fmd_calc.compute_split_half_baseline(
        embeddings,
        n_splits=baseline_splits,
        seed=seed,
    )

    token_lengths = token_lengths or []
    return {
        "n_samples": int(embeddings.shape[0]),
        "emb_dim": int(embeddings.shape[1]),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "trace_cov": float(np.trace(cov)),
        "eff_dim_90": int(effective_dimensionality(embeddings, threshold=0.90)),
        "eff_dim_95": int(effective_dimensionality(embeddings, threshold=0.95)),
        "mean_cosine_sim": float(mean_pairwise_cosine(embeddings, seed=seed)),
        "mean_token_length": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "std_token_length": float(np.std(token_lengths)) if token_lengths else 0.0,
        **baseline,
    }


def audit_distribution_pair(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    fmd_calc: FrechetMusicDistance,
    baseline_a: Optional[Dict[str, float]] = None,
    baseline_b: Optional[Dict[str, float]] = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """Audit one between-distribution comparison against split-half baselines."""
    components = fmd_calc.compute_nfmd(embeddings_a, embeddings_b)
    baseline_a = baseline_a or fmd_calc.compute_split_half_baseline(embeddings_a, seed=seed)
    baseline_b = baseline_b or fmd_calc.compute_split_half_baseline(embeddings_b, seed=seed + 1)

    baseline_fmd = float(np.mean([baseline_a["fmd_mean"], baseline_b["fmd_mean"]]))
    baseline_nfmd_trace = float(np.mean([baseline_a["nfmd_trace_mean"], baseline_b["nfmd_trace_mean"]]))
    baseline_nfmd_norm = float(np.mean([baseline_a["nfmd_norm_mean"], baseline_b["nfmd_norm_mean"]]))

    fmd_value = float(components["fmd"])
    mean_share = float(components["mean_diff_sq"] / fmd_value) if fmd_value > 1e-12 else 0.0
    cov_share = float(components["cov_trace"] / fmd_value) if fmd_value > 1e-12 else 0.0

    return {
        **components,
        "baseline_fmd_mean": baseline_fmd,
        "baseline_nfmd_trace_mean": baseline_nfmd_trace,
        "baseline_nfmd_norm_mean": baseline_nfmd_norm,
        "signal_ratio_fmd": float(fmd_value / baseline_fmd) if baseline_fmd > 1e-12 else np.nan,
        "signal_ratio_nfmd_trace": (
            float(components["nfmd_trace"] / baseline_nfmd_trace)
            if baseline_nfmd_trace > 1e-12
            else np.nan
        ),
        "signal_ratio_nfmd_norm": (
            float(components["nfmd_norm"] / baseline_nfmd_norm)
            if baseline_nfmd_norm > 1e-12
            else np.nan
        ),
        "mean_component_share": mean_share,
        "cov_component_share": cov_share,
        "norm_ratio": safe_spread([components["mean_norm1"], components["mean_norm2"]]),
        "trace_ratio": safe_spread([components["trace_cov1"], components["trace_cov2"]]),
    }


def compute_cross_model_spread_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Quantify how much the same music comparison spreads across models."""
    metrics = ["fmd", "nfmd_trace", "signal_ratio_fmd", "signal_ratio_nfmd_trace"]
    rows: List[Dict[str, float | str]] = []

    for metric in metrics:
        for (tokenizer, pair), group in pair_df.groupby(["tokenizer", "pair"]):
            spread = safe_spread(group[metric].values)
            rows.append(
                {
                    "metric": metric,
                    "tokenizer": tokenizer,
                    "pair": pair,
                    "spread": spread,
                    "n_models": int(group["model"].nunique()),
                }
            )

    spread_df = pd.DataFrame(rows)
    if spread_df.empty:
        return spread_df

    summary = (
        spread_df.groupby("metric")["spread"]
        .agg(["mean", "median", "max", "min"])
        .reset_index()
        .rename(columns={"mean": "mean_spread", "median": "median_spread", "max": "max_spread", "min": "min_spread"})
    )
    return summary.sort_values("median_spread")


class EmbeddingArchitectureAudit:
    """Run embedding-geometry audit for default preprocessing."""

    def __init__(self, config: Dict, max_files: int = DEFAULT_MAX_FILES, seed: int = DEFAULT_SEED):
        self.config = config
        self.max_files = max_files
        self.seed = seed

        self.dataset_manager = DatasetManager(config)
        self.preprocessor = MIDIPreprocessor(config)
        self.tokenization = TokenizationPipeline(config)
        self.emb_extractor = EmbeddingExtractor(config)
        self.fmd_calc = FrechetMusicDistance(config)

        self.genres = config["lakh"]["genres"]
        self.tokenizers = [t["type"] for t in config["tokenization"]["tokenizers"]]
        self.model_configs = {m["name"]: m for m in config["embeddings"]["models"]}
        self.models = list(self.model_configs)
        self.model_diagnostics = self.emb_extractor.get_model_diagnostics()

        reports_root = Path(config.get("results", {}).get("reports_dir", "results/reports"))
        plots_root = Path(config.get("results", {}).get("plots_dir", "results/plots"))
        self.output_dir = reports_root / "architecture_audit"
        self.plots_dir = plots_root / "paper"

        self.emb_cache: Dict[Tuple[str, str, str], np.ndarray] = {}
        self.token_length_cache: Dict[Tuple[str, str, str], List[int]] = {}
        self.sample_id_cache: Dict[Tuple[str, str, str], List[str]] = {}

    def extract_all(self) -> None:
        """Extract embeddings for all genre × tokenizer × model cells."""
        remove_velocity, hard_quant = DEFAULT_PREPROCESS
        combos = list(product(self.genres, self.tokenizers, self.models))

        for idx, (genre, tokenizer_name, model_name) in enumerate(combos, start=1):
            key = (genre, tokenizer_name, model_name)
            if key in self.emb_cache:
                continue

            logger.info(
                f"[{idx}/{len(combos)}] Architecture audit extract: {genre} × {tokenizer_name} × {model_name}"
            )
            ds_name = f"lakh_{genre}"
            midi_files = self.dataset_manager.list_midi_files(ds_name, processed=False, limit=self.max_files)
            if not midi_files:
                logger.warning(f"No MIDI files for {ds_name}")
                continue

            vectors: List[np.ndarray] = []
            token_lengths: List[int] = []
            sample_ids: List[str] = []
            tokenizer = self.tokenization.tokenizers[tokenizer_name]

            for midi_path in midi_files:
                try:
                    midi_data = self.preprocessor.load_midi(midi_path)
                    if midi_data is None:
                        continue
                    if remove_velocity:
                        midi_data = self.preprocessor.remove_velocity(midi_data)
                    if hard_quant:
                        midi_data = self.preprocessor.quantize_time(midi_data)
                    midi_data = self.preprocessor.filter_note_range(midi_data)
                    midi_data = self.preprocessor.normalize_instruments(midi_data)

                    tokens = tokenizer.encode_midi_object(midi_data)
                    if not tokens:
                        continue

                    vector = self.emb_extractor.extract_embeddings(
                        [tokens],
                        model_name,
                        midi_data_list=[midi_data],
                    )[0]
                    vectors.append(vector)
                    token_lengths.append(len(tokens))
                    sample_ids.append(midi_path.name)
                except Exception as exc:  # pragma: no cover - defensive for messy MIDI files
                    logger.debug(f"Skip {midi_path.name}: {exc}")

            if vectors:
                self.emb_cache[key] = np.vstack(vectors)
                self.token_length_cache[key] = token_lengths
                self.sample_id_cache[key] = sample_ids
                logger.info(f"  → {len(vectors)} embeddings, dim={vectors[0].shape[0]}")

    def _compute_cell_stats_for_cache(self, emb_cache: Dict[Tuple[str, str, str], np.ndarray]) -> pd.DataFrame:
        """Per-cell geometry statistics and split-half baselines for a given cache."""
        columns = [
            "genre", "tokenizer", "model", "architecture_family", "input_domain", "output_rule",
            "model_status", "is_real_model", "uses_fallback", "n_samples", "emb_dim", "mean_norm",
            "std_norm", "trace_cov", "eff_dim_90", "eff_dim_95", "mean_cosine_sim",
            "mean_token_length", "std_token_length", "fmd_mean", "fmd_std", "nfmd_trace_mean",
            "nfmd_trace_std", "nfmd_norm_mean", "nfmd_norm_std", "n_splits",
        ]
        rows: List[Dict[str, float | int | str]] = []
        for genre, tokenizer_name, model_name in product(self.genres, self.tokenizers, self.models):
            key = (genre, tokenizer_name, model_name)
            embeddings = emb_cache.get(key)
            if embeddings is None or embeddings.shape[0] < 4:
                continue

            metadata = get_architecture_metadata(model_name, self.model_configs.get(model_name))
            diagnostics = self.model_diagnostics.get(model_name, {})
            stats = summarize_embedding_distribution(
                embeddings,
                self.fmd_calc,
                token_lengths=self.token_length_cache.get(key),
                seed=self.seed,
            )
            rows.append(
                {
                    "genre": genre,
                    "tokenizer": tokenizer_name,
                    "model": model_name,
                    "architecture_family": metadata.family,
                    "input_domain": metadata.input_domain,
                    "output_rule": metadata.output_rule,
                    "model_status": diagnostics.get("status", "unknown"),
                    "is_real_model": bool(diagnostics.get("is_real_model", False)),
                    "uses_fallback": bool(diagnostics.get("uses_fallback", False)),
                    **stats,
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def compute_cell_stats(self) -> pd.DataFrame:
        """Per-cell geometry statistics and split-half baselines."""
        return self._compute_cell_stats_for_cache(self.emb_cache)

    def _compute_pair_audit_for_cache(
        self,
        cell_stats: pd.DataFrame,
        emb_cache: Dict[Tuple[str, str, str], np.ndarray],
        models_subset: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Cross-genre FMD audit with signal-to-baseline ratios for a given cache."""
        baseline_lookup = {
            (row["genre"], row["tokenizer"], row["model"]): {
                "fmd_mean": row["fmd_mean"],
                "nfmd_trace_mean": row["nfmd_trace_mean"],
                "nfmd_norm_mean": row["nfmd_norm_mean"],
            }
            for _, row in cell_stats.iterrows()
        }

        columns = [
            "tokenizer", "model", "genre_a", "genre_b", "pair", "fmd", "mean_diff_sq", "cov_trace",
            "trace_cov1", "trace_cov2", "mean_norm1", "mean_norm2", "nfmd_trace", "nfmd_norm",
            "baseline_fmd_mean", "baseline_nfmd_trace_mean", "baseline_nfmd_norm_mean",
            "signal_ratio_fmd", "signal_ratio_nfmd_trace", "signal_ratio_nfmd_norm",
            "mean_component_share", "cov_component_share", "norm_ratio", "trace_ratio",
        ]
        rows: List[Dict[str, float | str]] = []
        active_models = models_subset or self.models
        for tokenizer_name, model_name in product(self.tokenizers, active_models):
            for genre_a, genre_b in combinations(self.genres, 2):
                key_a = (genre_a, tokenizer_name, model_name)
                key_b = (genre_b, tokenizer_name, model_name)
                emb_a = emb_cache.get(key_a)
                emb_b = emb_cache.get(key_b)
                if emb_a is None or emb_b is None:
                    continue

                audit = audit_distribution_pair(
                    emb_a,
                    emb_b,
                    self.fmd_calc,
                    baseline_a=baseline_lookup.get(key_a),
                    baseline_b=baseline_lookup.get(key_b),
                    seed=self.seed,
                )
                rows.append(
                    {
                        "tokenizer": tokenizer_name,
                        "model": model_name,
                        "genre_a": genre_a,
                        "genre_b": genre_b,
                        "pair": f"{genre_a}_vs_{genre_b}",
                        **audit,
                    }
                )
        return pd.DataFrame(rows, columns=columns)

    def compute_pair_audit(self, cell_stats: pd.DataFrame) -> pd.DataFrame:
        """Cross-genre FMD audit with signal-to-baseline ratios."""
        return self._compute_pair_audit_for_cache(cell_stats, self.emb_cache)

    def _collect_symbolic_anchor_sets(
        self,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, List[str]], List[str], Dict[str, object]]:
        """Collect matched anchor embeddings per tokenizer for symbolic models."""
        eligible_models = [
            model_name
            for model_name in self.models
            if is_symbolic_shared_space_candidate(
                model_name,
                self.model_configs.get(model_name),
                self.model_diagnostics.get(model_name),
            )
        ]
        token_anchors: Dict[str, Dict[str, np.ndarray]] = {}
        token_eval_ids: Dict[str, List[str]] = {}
        summary: Dict[str, object] = {
            "eligible_models": eligible_models,
            "per_tokenizer": {},
        }

        for tokenizer_name in self.tokenizers:
            model_maps: Dict[str, Dict[str, np.ndarray]] = {}
            for model_name in eligible_models:
                id_to_vec: Dict[str, np.ndarray] = {}
                for genre in self.genres:
                    key = (genre, tokenizer_name, model_name)
                    embeddings = self.emb_cache.get(key)
                    sample_ids = self.sample_id_cache.get(key, [])
                    if embeddings is None or not sample_ids:
                        continue
                    for sample_id, vector in zip(sample_ids, embeddings):
                        id_to_vec[f"{genre}/{sample_id}"] = vector
                if id_to_vec:
                    model_maps[model_name] = id_to_vec

            if len(model_maps) < 2:
                summary["per_tokenizer"][tokenizer_name] = {
                    "status": "insufficient_models",
                    "n_models": int(len(model_maps)),
                }
                continue

            common_ids = sorted(set.intersection(*(set(m.keys()) for m in model_maps.values())))
            anchor_ids, eval_ids, protocol = split_shared_space_ids(common_ids, seed=self.seed)
            if protocol == "insufficient_matches":
                summary["per_tokenizer"][tokenizer_name] = {
                    "status": "insufficient_matches",
                    "n_models": int(len(model_maps)),
                    "n_matches": int(len(common_ids)),
                }
                continue

            token_anchors[tokenizer_name] = {
                model_name: np.vstack([model_maps[model_name][sample_id] for sample_id in anchor_ids])
                for model_name in model_maps
            }
            token_eval_ids[tokenizer_name] = eval_ids
            summary["per_tokenizer"][tokenizer_name] = {
                "status": "ok",
                "n_models": int(len(model_maps)),
                "n_matches": int(len(common_ids)),
                "n_anchor_matches": int(len(anchor_ids)),
                "n_eval_matches": int(len(eval_ids)),
                "evaluation_protocol": protocol,
            }

        return token_anchors, token_eval_ids, eligible_models, summary

    def compute_shared_space_audit(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, object]]:
        """Exploratory cross-model calibration via shared-space alignment."""
        token_anchors, token_eval_ids, eligible_models, anchor_summary = self._collect_symbolic_anchor_sets()
        if not token_anchors:
            return None, None, {
                "status": "skipped",
                "reason": "no_tokenizer_with_sufficient_matched_symbolic_anchors",
                "eligible_models": eligible_models,
                **anchor_summary,
            }

        reference_model = "CLaMP-2" if "CLaMP-2" in eligible_models else sorted(eligible_models)[0]
        calibrators: Dict[str, SharedSpaceCalibrator] = {}
        calibration_rows: List[Dict[str, object]] = []
        shared_cache: Dict[Tuple[str, str, str], np.ndarray] = {}

        for tokenizer_name, anchors_by_model in token_anchors.items():
            calibrator = SharedSpaceCalibrator(reference_model=reference_model)
            fit_info = calibrator.fit(anchors_by_model)
            calibrators[tokenizer_name] = calibrator
            token_protocol = anchor_summary.get("per_tokenizer", {}).get(tokenizer_name, {}).get(
                "evaluation_protocol",
                "unknown",
            )
            calibration_rows.append(
                {
                    "tokenizer": tokenizer_name,
                    "evaluation_protocol": token_protocol,
                    "n_eval_samples": int(len(token_eval_ids.get(tokenizer_name, []))),
                    **fit_info,
                }
            )

        for genre, tokenizer_name, model_name in product(self.genres, self.tokenizers, eligible_models):
            key = (genre, tokenizer_name, model_name)
            embeddings = self.emb_cache.get(key)
            calibrator = calibrators.get(tokenizer_name)
            if embeddings is None or calibrator is None:
                continue
            sample_ids = self.sample_id_cache.get(key, [])
            allowed_ids = set(token_eval_ids.get(tokenizer_name, []))
            if sample_ids and allowed_ids:
                keep_idx = [
                    idx for idx, sample_id in enumerate(sample_ids) if f"{genre}/{sample_id}" in allowed_ids
                ]
                if len(keep_idx) < 4:
                    continue
                embeddings = embeddings[keep_idx]
            shared_cache[key] = calibrator.transform(model_name, embeddings)

        shared_cell_stats = self._compute_cell_stats_for_cache(shared_cache)
        shared_pair_df = self._compute_pair_audit_for_cache(
            shared_cell_stats,
            shared_cache,
            models_subset=eligible_models,
        )
        if not shared_pair_df.empty:
            shared_pair_df["comparison_space"] = "shared_space"

        calibration_df = pd.DataFrame(calibration_rows)
        summary = {
            "status": "ok",
            "reference_model": reference_model,
            "eligible_models": eligible_models,
            "n_tokenizers_calibrated": int(len(calibrators)),
            "evaluation_protocols": {
                tokenizer: info.get("evaluation_protocol", "unknown")
                for tokenizer, info in anchor_summary.get("per_tokenizer", {}).items()
                if info.get("status") == "ok"
            },
            "anchor_summary": anchor_summary,
            "calibration_rows": calibration_df.to_dict(orient="records"),
        }
        return shared_pair_df, calibration_df, summary

    def build_summary(
        self,
        cell_stats: pd.DataFrame,
        pair_df: pd.DataFrame,
        shared_pair_df: Optional[pd.DataFrame] = None,
        shared_space_summary: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Aggregate audit findings into a compact JSON-friendly summary."""
        spread_summary = compute_cross_model_spread_summary(pair_df)
        shared_spread_summary = (
            compute_cross_model_spread_summary(shared_pair_df)
            if shared_pair_df is not None and not shared_pair_df.empty
            else pd.DataFrame()
        )
        model_geo = (
            cell_stats.groupby("model")
            .agg(
                mean_norm=("mean_norm", "mean"),
                mean_trace_cov=("trace_cov", "mean"),
                mean_eff_dim_90=("eff_dim_90", "mean"),
                mean_fmd_baseline=("fmd_mean", "mean"),
                mean_nfmd_trace_baseline=("nfmd_trace_mean", "mean"),
                model_status=("model_status", "first"),
                uses_fallback=("uses_fallback", "first"),
            )
            .reset_index()
        ) if not cell_stats.empty else pd.DataFrame()

        spread_delta: List[Dict[str, object]] = []
        if not spread_summary.empty and not shared_spread_summary.empty:
            merged = spread_summary.merge(
                shared_spread_summary,
                on="metric",
                suffixes=("_raw", "_shared"),
            )
            for _, row in merged.iterrows():
                spread_delta.append(
                    {
                        "metric": row["metric"],
                        "median_spread_raw": float(row["median_spread_raw"]),
                        "median_spread_shared": float(row["median_spread_shared"]),
                        "reduction_ratio": float(
                            row["median_spread_shared"] / row["median_spread_raw"]
                        ) if row["median_spread_raw"] > 1e-12 else np.nan,
                    }
                )

        claim = (
            "Cross-model FMD appears to be architecture-dependent; nFMD should be treated as a "
            "working calibration hypothesis until embedding geometry is better understood."
        )
        if spread_delta:
            delta_map = {row["metric"]: row["reduction_ratio"] for row in spread_delta}
            raw_ratio = float(delta_map.get("fmd", np.nan))
            nfmd_ratio = float(delta_map.get("nfmd_trace", np.nan))
            signal_ratio = float(delta_map.get("signal_ratio_nfmd_trace", np.nan))
            if np.isfinite(raw_ratio) and raw_ratio < 1 and (
                (np.isfinite(nfmd_ratio) and nfmd_ratio > 1) or (np.isfinite(signal_ratio) and signal_ratio > 1)
            ):
                claim = (
                    "Cross-model FMD appears to be architecture-dependent. Holdout shared-space alignment can "
                    "reduce raw FMD spread, but current evidence is mixed because normalized and baseline-aware "
                    "spreads may worsen; shared-space calibration therefore remains exploratory."
                )

        summary: Dict[str, object] = {
            "n_cells": int(len(cell_stats)),
            "n_pairs": int(len(pair_df)),
            "nfmd_status": "hypothesis_only",
            "claim": claim,
            "cross_model_spread": spread_summary.to_dict(orient="records"),
            "shared_space_spread": shared_spread_summary.to_dict(orient="records"),
            "shared_space_delta": spread_delta,
            "model_geometry": model_geo.to_dict(orient="records"),
            "model_diagnostics": self.model_diagnostics,
            "shared_space": shared_space_summary or {"status": "not_run"},
        }
        return summary

    def generate_plots(
        self,
        cell_stats: pd.DataFrame,
        pair_df: pd.DataFrame,
        spread_summary: pd.DataFrame,
        shared_spread_summary: Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
        """Generate architecture-audit plots."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", font_scale=1.0)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        outputs: Dict[str, str] = {}

        def save(fig, name: str) -> None:
            path = self.plots_dir / f"{name}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            outputs[name] = str(path)

        if not cell_stats.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=cell_stats,
                x="mean_norm",
                y="trace_cov",
                hue="model",
                style="tokenizer",
                ax=ax,
                s=85,
            )
            ax.set_title("Embedding geometry by tokenizer × model × genre")
            ax.set_xlabel("Mean embedding norm")
            ax.set_ylabel("Covariance trace")
            save(fig, "architecture_norm_trace_scatter")

        if not pair_df.empty:
            signal_df = pair_df.melt(
                id_vars=["tokenizer", "model", "pair"],
                value_vars=["signal_ratio_fmd", "signal_ratio_nfmd_trace"],
                var_name="metric",
                value_name="signal_ratio",
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=signal_df, x="model", y="signal_ratio", hue="metric", ax=ax)
            ax.set_title("Cross-genre signal relative to split-half baseline")
            ax.set_xlabel("Embedding model")
            ax.set_ylabel("Signal / baseline")
            ax.tick_params(axis="x", rotation=25)
            save(fig, "architecture_signal_to_baseline")

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=spread_summary, x="metric", y="median_spread", ax=ax, palette="deep")
            ax.set_title("Median cross-model spread for the same music comparison")
            ax.set_xlabel("Metric")
            ax.set_ylabel("Median max/min spread across models")
            ax.tick_params(axis="x", rotation=20)
            save(fig, "architecture_cross_model_spread")

            if shared_spread_summary is not None and not shared_spread_summary.empty:
                compare_df = pd.concat(
                    [
                        spread_summary.assign(space="raw_or_normalized"),
                        shared_spread_summary.assign(space="shared_space"),
                    ],
                    ignore_index=True,
                )
                compare_df = compare_df[compare_df["metric"].isin(["fmd", "nfmd_trace"])]
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(
                    data=compare_df,
                    x="metric",
                    y="median_spread",
                    hue="space",
                    ax=ax,
                    palette="deep",
                )
                ax.set_title("Median cross-model spread before vs after shared-space calibration")
                ax.set_xlabel("Metric")
                ax.set_ylabel("Median max/min spread across models")
                save(fig, "architecture_shared_space_spread")

        return outputs

    def generate_report(
        self,
        cell_stats: pd.DataFrame,
        pair_df: pd.DataFrame,
        spread_summary: pd.DataFrame,
        summary: Dict[str, object],
        plot_paths: Dict[str, str],
        shared_spread_summary: Optional[pd.DataFrame] = None,
    ) -> Path:
        """Write markdown report with cautious conclusions."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "ARCHITECTURE_AUDIT_REPORT.md"

        lines: List[str] = [
            "# Embedding Architecture Audit",
            "",
            "**Goal:** assess whether cross-model FMD / nFMD comparisons are interpretable, and treat nFMD as a hypothesis rather than a validated replacement.",
            "",
            "## Working Hypothesis",
            "",
            "Cross-model Fréchet comparisons only make sense when embedding spaces have roughly comparable geometry. If architectures emit vectors with different normalisation rules, norm scales, covariance traces, or effective dimensionalities, FMD may be dominated by architecture rather than musical content.",
            "",
            "## Architecture facts from the current pipeline",
            "",
            "| Model | Family | Output rule |",
            "|-------|--------|-------------|",
        ]

        seen_models = set()
        for _, row in cell_stats[["model", "architecture_family", "output_rule"]].drop_duplicates().iterrows():
            if row["model"] in seen_models:
                continue
            seen_models.add(row["model"])
            lines.append(f"| {row['model']} | {row['architecture_family']} | {row['output_rule']} |")

        lines.extend([
            "",
            "## Runtime model integrity",
            "",
            "These flags help separate genuine architectural effects from proxy/fallback artefacts.",
            "",
            "| Model | Status | Real model? | Uses fallback? |",
            "|-------|--------|-------------|----------------|",
        ])
        for model_name in self.models:
            diag = self.model_diagnostics.get(model_name, {})
            lines.append(
                f"| {model_name} | {diag.get('status', 'unknown')} | {bool(diag.get('is_real_model', False))} | {bool(diag.get('uses_fallback', False))} |"
            )

        lines.extend([
            "",
            "## Geometry summary by model",
            "",
            "| Model | Mean norm | Mean trace(Σ) | Mean eff. dim (90%) | Split-half FMD | Split-half nFMD_trace |",
            "|------|-----------|---------------|---------------------|----------------|-----------------------|",
        ])

        if not cell_stats.empty:
            geo = (
                cell_stats.groupby("model")
                .agg(
                    mean_norm=("mean_norm", "mean"),
                    mean_trace_cov=("trace_cov", "mean"),
                    mean_eff_dim_90=("eff_dim_90", "mean"),
                    mean_fmd_mean=("fmd_mean", "mean"),
                    mean_nfmd_trace_mean=("nfmd_trace_mean", "mean"),
                )
                .reset_index()
            )
            for _, row in geo.iterrows():
                lines.append(
                    f"| {row['model']} | {row['mean_norm']:.3f} | {row['mean_trace_cov']:.3f} | "
                    f"{row['mean_eff_dim_90']:.1f} | {row['mean_fmd_mean']:.4f} | {row['mean_nfmd_trace_mean']:.4f} |"
                )

        lines.extend([
            "",
            "## Cross-model spread for the same music comparison",
            "",
            "If the same tokenizer + genre pair yields very different values depending only on the embedding model, then cross-model comparability is weak.",
            "",
            "| Metric | Mean spread | Median spread | Max spread | Min spread |",
            "|--------|-------------|---------------|------------|------------|",
        ])
        if not spread_summary.empty:
            for _, row in spread_summary.iterrows():
                lines.append(
                    f"| {row['metric']} | {row['mean_spread']:.3f} | {row['median_spread']:.3f} | {row['max_spread']:.3f} | {row['min_spread']:.3f} |"
                )

        if shared_spread_summary is not None and not shared_spread_summary.empty:
            lines.extend([
                "",
                "## Exploratory shared-space calibration",
                "",
                "We additionally fit a matched-sample shared comparison space for symbolic models only (PCA whitening + orthogonal Procrustes to a reference model). Whenever enough matched files are available, alignment is fitted on an anchor subset and evaluated on disjoint holdout samples; otherwise the report explicitly flags an in-sample fallback.",
                "",
                "| Metric | Mean spread | Median spread | Max spread | Min spread |",
                "|--------|-------------|---------------|------------|------------|",
            ])
            for _, row in shared_spread_summary.iterrows():
                lines.append(
                    f"| {row['metric']} | {row['mean_spread']:.3f} | {row['median_spread']:.3f} | {row['max_spread']:.3f} | {row['min_spread']:.3f} |"
                )

            deltas = summary.get("shared_space_delta", []) or []
            if deltas:
                lines.extend([
                    "",
                    "### Calibration protocol by tokenizer",
                    "",
                    "| Tokenizer | Protocol | # anchor matches | # eval matches |",
                    "|-----------|----------|------------------|----------------|",
                ])
                for tokenizer, info in (summary.get("shared_space", {}) or {}).get("anchor_summary", {}).get("per_tokenizer", {}).items():
                    if info.get("status") != "ok":
                        continue
                    lines.append(
                        f"| {tokenizer} | {info.get('evaluation_protocol', 'unknown')} | {info.get('n_anchor_matches', 0)} | {info.get('n_eval_matches', 0)} |"
                    )

                lines.extend([
                    "",
                    "### Spread reduction relative to raw space",
                    "",
                    "| Metric | Median spread raw | Median spread shared | Shared/raw ratio |",
                    "|--------|-------------------|----------------------|------------------|",
                ])
                for row in deltas:
                    lines.append(
                        f"| {row['metric']} | {row['median_spread_raw']:.3f} | {row['median_spread_shared']:.3f} | {row['reduction_ratio']:.3f} |"
                    )

        if not pair_df.empty:
            model_signal = (
                pair_df.groupby("model")
                .agg(
                    fmd=("fmd", "mean"),
                    nfmd_trace=("nfmd_trace", "mean"),
                    signal_ratio_fmd=("signal_ratio_fmd", "mean"),
                    signal_ratio_nfmd_trace=("signal_ratio_nfmd_trace", "mean"),
                    mean_component_share=("mean_component_share", "mean"),
                    cov_component_share=("cov_component_share", "mean"),
                )
                .reset_index()
            )
            lines.extend([
                "",
                "## Signal relative to model-specific baseline",
                "",
                "| Model | Mean raw FMD | Mean nFMD_trace | Mean raw signal/baseline | Mean nFMD_trace signal/baseline | Mean-share in FMD | Cov-share in FMD |",
                "|------|--------------|-----------------|--------------------------|----------------------------------|------------------|------------------|",
            ])
            for _, row in model_signal.iterrows():
                lines.append(
                    f"| {row['model']} | {row['fmd']:.4f} | {row['nfmd_trace']:.4f} | "
                    f"{row['signal_ratio_fmd']:.3f} | {row['signal_ratio_nfmd_trace']:.3f} | "
                    f"{row['mean_component_share']:.3f} | {row['cov_component_share']:.3f} |"
                )

        lines.extend([
            "",
            "## Interpretation",
            "",
            "1. **Certain architectural differences are built into the pipeline itself.** In particular, CLaMP outputs L2-normalized projections, whereas MusicBERT / NLP-Baseline use raw [CLS] states and MERT uses raw mean-pooled audio states.",
            "2. **Therefore raw FMD is not guaranteed to be cross-model comparable.** Differences may reflect embedding geometry rather than only musical structure.",
            "3. **nFMD should be treated as a hypothesis.** If it reduces cross-model spread and preserves signal-over-baseline, it is promising; if not, it should not be presented as a validated solution.",
            "4. **Shared-space calibration is an exploratory research direction, not yet a validated fix.** Holdout evaluation is substantially more informative than in-sample alignment, but even holdout results only show that part of the spread may be geometric --- not that semantics are perfectly aligned across models.",
            "5. **Mixed holdout outcomes matter.** A reduction in raw FMD spread is not sufficient if normalized or signal-to-baseline spreads worsen after alignment; this would indicate partial geometric debiasing without stable comparative semantics.",
            "6. **Safer current recommendation:** compare rankings within a fixed embedding model, and treat cross-model comparisons as exploratory unless further calibration is introduced.",
            "",
            "## Output artefacts",
            "",
            f"- `architecture_cell_stats.csv` — per-genre geometry and split-half baselines",
            f"- `architecture_pair_audit.csv` — pairwise FMD/nFMD plus signal-to-baseline ratios",
            f"- `architecture_shared_space_pair_audit.csv` — exploratory pair audit after shared-space alignment",
            f"- `architecture_shared_space_calibration.csv` — tokenizer-level calibration metadata",
            f"- `architecture_summary.json` — compact machine-readable summary",
        ])

        for name, path in plot_paths.items():
            lines.append(f"- `{Path(path).name}` — plot: {name}")

        lines.extend([
            "",
            "## Status",
            "",
            f"- `nFMD`: **{summary.get('nfmd_status', 'hypothesis_only')}**",
            f"- Main claim: {summary.get('claim', '')}",
            "",
        ])

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def run(self) -> Dict[str, str]:
        """Execute the full audit pipeline."""
        logger.info("=" * 70)
        logger.info("EMBEDDING ARCHITECTURE AUDIT")
        logger.info("=" * 70)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.extract_all()
        cell_stats = self.compute_cell_stats()
        pair_df = self.compute_pair_audit(cell_stats)
        spread_summary = compute_cross_model_spread_summary(pair_df)
        shared_pair_df, calibration_df, shared_space_summary = self.compute_shared_space_audit()
        shared_spread_summary = (
            compute_cross_model_spread_summary(shared_pair_df)
            if shared_pair_df is not None and not shared_pair_df.empty
            else pd.DataFrame()
        )
        summary = self.build_summary(
            cell_stats,
            pair_df,
            shared_pair_df=shared_pair_df,
            shared_space_summary=shared_space_summary,
        )
        plot_paths = self.generate_plots(
            cell_stats,
            pair_df,
            spread_summary,
            shared_spread_summary=shared_spread_summary,
        )
        report_path = self.generate_report(
            cell_stats,
            pair_df,
            spread_summary,
            summary,
            plot_paths,
            shared_spread_summary=shared_spread_summary,
        )

        cell_stats_path = self.output_dir / "architecture_cell_stats.csv"
        pair_path = self.output_dir / "architecture_pair_audit.csv"
        spread_path = self.output_dir / "architecture_cross_model_spread.csv"
        shared_pair_path = self.output_dir / "architecture_shared_space_pair_audit.csv"
        shared_spread_path = self.output_dir / "architecture_shared_space_spread.csv"
        calibration_path = self.output_dir / "architecture_shared_space_calibration.csv"
        summary_path = self.output_dir / "architecture_summary.json"

        cell_stats.to_csv(cell_stats_path, index=False)
        pair_df.to_csv(pair_path, index=False)
        spread_summary.to_csv(spread_path, index=False)
        if shared_pair_df is not None and not shared_pair_df.empty:
            shared_pair_df.to_csv(shared_pair_path, index=False)
        if not shared_spread_summary.empty:
            shared_spread_summary.to_csv(shared_spread_path, index=False)
        if calibration_df is not None and not calibration_df.empty:
            calibration_df.to_csv(calibration_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        logger.info(f"Report: {report_path}")
        logger.info(f"Cell stats: {cell_stats_path}")
        logger.info(f"Pair audit: {pair_path}")
        logger.info(f"Spread summary: {spread_path}")
        if shared_pair_df is not None and not shared_pair_df.empty:
            logger.info(f"Shared-space pair audit: {shared_pair_path}")
        if not shared_spread_summary.empty:
            logger.info(f"Shared-space spread: {shared_spread_path}")
        if calibration_df is not None and not calibration_df.empty:
            logger.info(f"Shared-space calibration: {calibration_path}")
        logger.info(f"Summary JSON: {summary_path}")

        return {
            "report": str(report_path),
            "cell_stats": str(cell_stats_path),
            "pair_audit": str(pair_path),
            "spread_summary": str(spread_path),
            "shared_pair_audit": str(shared_pair_path),
            "shared_spread_summary": str(shared_spread_path),
            "shared_calibration": str(calibration_path),
            "summary_json": str(summary_path),
            **plot_paths,
        }

