"""Sensitivity Profiler - paper-grade FMD sensitivity study.

Steps:
  1. Define honest model/input configurations (MidiTok models, CLaMP MTF/ABC)
  2. Use at least 4 datasets with clear stylistic relations
  3. Self-similarity sanity check (split-half FMD ≈ 0)
  4. Cross-dataset ranking with Spearman τ between configs
  5. Perturbation sensitivity profiling
  6. Bootstrap stability analysis
  7. Synthesis (tables + plots)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from utils.config import load_config
from experiments.study_config import (
    MIDITOK_FORMAT,
    validate_embedding_input,
)


@dataclass
class SensitivityConfig:
    """A single configuration (model + input representation) to evaluate."""
    name: str
    model: str
    input_format: str  # MIDITOK, MTF, ABC
    tokenizer: Optional[str] = None
    description: str = ""


@dataclass
class PerturbationSpec:
    """Specification for one perturbation variant."""
    name: str
    description: str
    remove_velocity: bool = False
    quantize_time: bool = False
    constant_tempo: bool = False


@dataclass
class SensitivityResult:
    """Container for all pivot results."""
    self_similarity: Dict = field(default_factory=dict)
    cross_dataset_fmd: Dict = field(default_factory=dict)
    spearman_matrix: Dict = field(default_factory=dict)
    perturbation_profiles: Dict = field(default_factory=dict)
    bootstrap_stability: Dict = field(default_factory=dict)


class SensitivityProfiler:
    """Main class implementing the 7-step sensitivity pivot."""

    def __init__(self, main_config: Dict, pivot_config_path: str = "configs/sensitivity_pivot.yaml"):
        """Initialize the profiler.

        Args:
            main_config: Main project config (for infrastructure: embeddings, tokenization, etc.)
            pivot_config_path: Path to the pivot-specific config YAML.
        """
        self.config = main_config
        self.pivot_cfg = load_config(pivot_config_path)

        # Infrastructure
        print("[Progress] importing sensitivity infrastructure", flush=True)
        from data.manager import DatasetManager
        from embeddings.extractor import EmbeddingExtractor
        from metrics.fmd import FrechetMusicDistance
        from preprocessing.processor import MIDIPreprocessor
        from tokenization.tokenizer import TokenizationPipeline

        print("[Progress] constructing DatasetManager", flush=True)
        self.dataset_manager = DatasetManager(main_config)
        print("[Progress] constructing MIDIPreprocessor", flush=True)
        self.preprocessor = MIDIPreprocessor(main_config)
        print("[Progress] constructing TokenizationPipeline", flush=True)
        self.tokenization = TokenizationPipeline(main_config)
        print("[Progress] constructing EmbeddingExtractor (models may download/load now)", flush=True)
        self.embeddings = EmbeddingExtractor(main_config)
        print("[Progress] constructing FrechetMusicDistance", flush=True)
        self.fmd = FrechetMusicDistance(main_config)

        # Pivot parameters
        self.seed = int(self.pivot_cfg.get("seed", 42))
        self.max_files = int(self.pivot_cfg.get("max_files_per_dataset", 200))
        self.output_dir = Path(self.pivot_cfg.get("output_dir", "results/reports/sensitivity_pivot"))
        self.plots_dir = Path(self.pivot_cfg.get("plots_dir", "results/plots/sensitivity_pivot"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Parse and validate configurations. This deliberately rejects the old
        # invalid CLaMP2-REMI setup: CLaMP-2 consumes MTF/M3, not MidiTok REMI.
        self.configurations = []
        for c in self.pivot_cfg["configurations"]:
            spec = validate_embedding_input(
                model=c["model"],
                input_format=c.get("input_format"),
                tokenizer=c.get("tokenizer"),
            )
            self.configurations.append(
                SensitivityConfig(
                    name=c["name"],
                    model=spec.model,
                    input_format=spec.input_format,
                    tokenizer=spec.tokenizer,
                    description=c.get("description", ""),
                )
            )

        # Parse datasets (required + optional that exist on disk)
        self.datasets = [d["name"] for d in self.pivot_cfg["datasets"] if not d.get("optional", False)]
        for d in self.pivot_cfg["datasets"]:
            if d.get("optional", False):
                ds_path = Path(self.config["data"]["raw_data_dir"]) / d["name"]
                if ds_path.exists() and list(ds_path.glob("**/*.mid*")):
                    self.datasets.append(d["name"])
        self.optional_datasets = [d["name"] for d in self.pivot_cfg["datasets"] if d.get("optional", False)]

        self.min_pairs_for_spearman = int(self.pivot_cfg.get("min_pairs_for_spearman", 6))

        # Parse perturbations
        self.perturbations = [
            PerturbationSpec(
                name=p["name"],
                description=p["description"],
                remove_velocity=p.get("remove_velocity", False),
                quantize_time=p.get("quantize_time", False),
                constant_tempo=p.get("constant_tempo", False),
            )
            for p in self.pivot_cfg["perturbations"]
        ]

        # Bootstrap config
        self.bootstrap_cfg = self.pivot_cfg.get("bootstrap", {})

        logger.info(
            f"SensitivityProfiler initialized: {len(self.configurations)} configs, "
            f"{len(self.datasets)} datasets, {len(self.perturbations)} perturbations"
        )

    # ─── Dataset Loading ────────────────────────────────────────────────

    def _load_dataset_midi(self, dataset_name: str) -> List[Path]:
        """Load MIDI file paths for a dataset, up to max_files."""
        dataset_dir = Path(self.config["data"]["raw_data_dir"]) / dataset_name
        if not dataset_dir.exists():
            # Try to ensure it exists via dataset manager
            self.dataset_manager.ensure_dataset_exists(dataset_name, download=False)

        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return []

        midi_files = sorted(dataset_dir.glob("**/*.mid")) + sorted(dataset_dir.glob("**/*.midi"))
        rng = np.random.default_rng(self.seed)
        if len(midi_files) > self.max_files:
            indices = rng.choice(len(midi_files), size=self.max_files, replace=False)
            midi_files = [midi_files[i] for i in sorted(indices)]

        logger.info(f"Dataset '{dataset_name}': {len(midi_files)} MIDI files loaded")
        return midi_files

    # ─── Embedding Extraction ───────────────────────────────────────────

    def _extract_embeddings(
        self,
        midi_files: List[Path],
        config: SensitivityConfig,
        perturbation: PerturbationSpec | None = None,
    ) -> np.ndarray:
        """Extract embeddings for a list of MIDI files using a specific config and perturbation.

        Args:
            midi_files: List of MIDI file paths
            config: Configuration (model + tokenizer)
            perturbation: Optional perturbation to apply

        Returns:
            (N, D) numpy array of embeddings
        """
        token_sequences: List[List[int]] = []
        midi_data_list: List = []
        failed = 0

        for midi_path in midi_files:
            try:
                # Load and preprocess
                midi_data = self.preprocessor.load_midi(midi_path)
                if midi_data is None:
                    failed += 1
                    continue

                # Basic preprocessing
                midi_data = self.preprocessor.filter_note_range(midi_data)
                midi_data = self.preprocessor.normalize_instruments(midi_data)

                # Apply perturbations
                if perturbation:
                    if perturbation.remove_velocity:
                        midi_data = self.preprocessor.remove_velocity(midi_data)
                    if perturbation.quantize_time:
                        midi_data = self.preprocessor.quantize_time(midi_data, hard_quantize=True)
                    if perturbation.constant_tempo:
                        midi_data = self.preprocessor.normalize_tempo(midi_data, target_bpm=120.0)

                # Tokenize only when a MidiTok-based model is used; CLaMP MTF/ABC
                # use midi_data directly and must not be labelled as REMI.
                tokens: List[int] = []
                if config.input_format == MIDITOK_FORMAT:
                    if not config.tokenizer:
                        raise ValueError(f"Config {config.name} requires a MidiTok tokenizer")
                    tokenizer = self.tokenization.tokenizers[config.tokenizer]
                    tokens = tokenizer.encode_midi_object(midi_data)
                    if not tokens:
                        failed += 1
                        continue

                token_sequences.append(tokens)
                midi_data_list.append(midi_data)

            except Exception as e:
                logger.debug(f"Failed to process {midi_path.name}: {e}")
                failed += 1

        if failed > 0:
            logger.warning(f"Config={config.name}: {failed}/{len(midi_files)} files failed")

        if not token_sequences:
            return np.array([])

        miditok_tokenizer = None
        if config.input_format == MIDITOK_FORMAT and config.tokenizer:
            miditok_tokenizer = self.tokenization.tokenizers[config.tokenizer].miditok_tokenizer

        input_formats = [config.input_format] * len(token_sequences)
        embeddings = self.embeddings.extract_embeddings(
            token_sequences=token_sequences,
            model_name=config.model,
            midi_data_list=midi_data_list,
            input_formats=input_formats,
            miditok_tokenizer=miditok_tokenizer,
            use_cache=False,
        )
        return embeddings

    # ─── Bootstrap helpers (perturbation significance) ──────────────────

    def _noise_floor_distribution(
        self, emb: np.ndarray, n_resamples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Null distribution of within-dataset FMD (split-half bootstrap).

        Repeatedly splits a single embedding set into two random halves and
        measures their FMD. This is the sampling noise a perturbation must
        exceed to count as detected; its mean is the per-configuration
        noise floor used for the SNR.
        """
        n = emb.shape[0]
        half = n // 2
        vals = []
        for _ in range(n_resamples):
            idx = rng.permutation(n)
            a = emb[idx[:half]]
            b = emb[idx[half:2 * half]]
            vals.append(float(self.fmd.compute_fmd(a, b)))
        return np.array(vals)

    def _perturbation_bootstrap(
        self,
        emb_orig: np.ndarray,
        emb_pert: np.ndarray,
        n_resamples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Bootstrap distribution of FMD(original, perturbed).

        Resamples the two embedding sets independently (FMD compares
        distributions, not paired points, so unpaired resampling is the
        correct bootstrap and avoids any mis-alignment when a file fails
        under one condition but not the other).
        """
        n_o, n_p = emb_orig.shape[0], emb_pert.shape[0]
        vals = []
        for _ in range(n_resamples):
            io = rng.choice(n_o, n_o, replace=True)
            ip = rng.choice(n_p, n_p, replace=True)
            vals.append(float(self.fmd.compute_fmd(emb_orig[io], emb_pert[ip])))
        return np.array(vals)

    def _permutation_pvalue(
        self,
        emb_orig: np.ndarray,
        emb_pert: np.ndarray,
        observed_fmd: float,
        n_perm: int,
        rng: np.random.Generator,
    ) -> float:
        """Two-sample permutation test for FMD(original, perturbed).

        Under H0 (the perturbation has no distributional effect) the
        'original'/'perturbed' labels are exchangeable. We pool the two sets,
        randomly relabel into two groups of the original sizes, and recompute
        FMD; the p-value is the fraction of permuted FMDs at least as large as
        the observed one. This is matched-sample-size and free of the
        bootstrap tie-inflation, so it is the appropriate significance test
        (unlike the split-half floor, which is only an effect-size reference).
        """
        pooled = np.vstack([emb_orig, emb_pert])
        n_a = emb_orig.shape[0]
        total = pooled.shape[0]
        count = 0
        for _ in range(n_perm):
            idx = rng.permutation(total)
            a = pooled[idx[:n_a]]
            b = pooled[idx[n_a:]]
            if self.fmd.compute_fmd(a, b) >= observed_fmd:
                count += 1
        # add-one estimator: never reports an impossible p == 0
        return (count + 1) / (n_perm + 1)

    # ─── Step 3: Self-Similarity ────────────────────────────────────────

    def run_self_similarity(self) -> pd.DataFrame:
        """Step 3: Split-half FMD within each dataset for each configuration.

        Expected: FMD ≈ 0 for each split. If high → configuration is unstable.
        """
        from tqdm import tqdm

        logger.info("=" * 60)
        logger.info("STEP 3: Self-Similarity Sanity Check")
        logger.info("=" * 60)

        rows = []
        rng = np.random.default_rng(self.seed)

        for dataset_name in tqdm(self.datasets, desc="Self-similarity"):
            midi_files = self._load_dataset_midi(dataset_name)
            if len(midi_files) < 10:
                logger.warning(f"Skipping {dataset_name}: too few files ({len(midi_files)})")
                continue

            # Random split into two halves
            indices = rng.permutation(len(midi_files))
            half = len(indices) // 2
            split_a = [midi_files[i] for i in indices[:half]]
            split_b = [midi_files[i] for i in indices[half:2 * half]]

            for cfg in self.configurations:
                emb_a = self._extract_embeddings(split_a, cfg)
                emb_b = self._extract_embeddings(split_b, cfg)

                if emb_a.size == 0 or emb_b.size == 0:
                    logger.warning(f"Empty embeddings for {cfg.name}/{dataset_name}")
                    continue

                fmd_val = float(self.fmd.compute_fmd(emb_a, emb_b))
                rows.append({
                    "dataset": dataset_name,
                    "configuration": cfg.name,
                    "n_samples_per_half": half,
                    "split_half_fmd": fmd_val,
                })
                logger.info(f"  {cfg.name} / {dataset_name}: split-half FMD = {fmd_val:.4f}")

        df = pd.DataFrame(rows)
        out_path = self.output_dir / "self_similarity.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Self-similarity results saved to {out_path}")
        return df

    # ─── Step 4: Cross-Dataset Ranking ──────────────────────────────────

    def run_cross_dataset_ranking(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Step 4: Compute FMD for all dataset pairs × configurations.

        Returns ranking table + Spearman τ agreement matrix.
        """
        from scipy.stats import spearmanr
        from tqdm import tqdm

        logger.info("=" * 60)
        logger.info("STEP 4: Cross-Dataset Ranking")
        logger.info("=" * 60)

        # Pre-extract embeddings for each (dataset, config) pair
        embeddings_cache: Dict[Tuple[str, str], np.ndarray] = {}

        for dataset_name in tqdm(self.datasets, desc="Extracting embeddings"):
            midi_files = self._load_dataset_midi(dataset_name)
            if not midi_files:
                continue
            for cfg in self.configurations:
                emb = self._extract_embeddings(midi_files, cfg)
                if emb.size > 0:
                    embeddings_cache[(dataset_name, cfg.name)] = emb
                    logger.info(f"  {cfg.name}/{dataset_name}: {emb.shape[0]} embeddings ({emb.shape[1]}d)")

        # Compute FMD for all pairs
        dataset_pairs = list(combinations(self.datasets, 2))
        rows = []

        for ds_a, ds_b in dataset_pairs:
            for cfg in self.configurations:
                key_a = (ds_a, cfg.name)
                key_b = (ds_b, cfg.name)
                if key_a not in embeddings_cache or key_b not in embeddings_cache:
                    continue

                fmd_val = float(self.fmd.compute_fmd(
                    embeddings_cache[key_a], embeddings_cache[key_b]
                ))
                rows.append({
                    "dataset_a": ds_a,
                    "dataset_b": ds_b,
                    "pair": f"{ds_a}_vs_{ds_b}",
                    "configuration": cfg.name,
                    "fmd": fmd_val,
                })
                logger.info(f"  {cfg.name}: {ds_a} vs {ds_b} → FMD = {fmd_val:.4f}")

        df = pd.DataFrame(rows)
        out_path = self.output_dir / "cross_dataset_fmd.csv"
        df.to_csv(out_path, index=False)

        # Compute Spearman τ between configurations
        spearman_rows = []
        config_names = [c.name for c in self.configurations]

        for cfg_a, cfg_b in combinations(config_names, 2):
            # Get FMD vectors aligned by pair
            merged = df[df["configuration"].isin([cfg_a, cfg_b])].pivot_table(
                index="pair", columns="configuration", values="fmd"
            )
            n_pairs = int(merged.shape[0])
            if n_pairs < self.min_pairs_for_spearman:
                logger.warning(
                    f"Skipping Spearman {cfg_a} vs {cfg_b}: only {n_pairs} pairs "
                    f"(need >= {self.min_pairs_for_spearman})"
                )
                continue
            if cfg_a in merged.columns and cfg_b in merged.columns:
                tau, p_value = spearmanr(merged[cfg_a], merged[cfg_b])
                spearman_rows.append({
                    "config_a": cfg_a,
                    "config_b": cfg_b,
                    "spearman_tau": float(tau),
                    "p_value": float(p_value) if n_pairs >= 10 else float("nan"),
                    "n_pairs": n_pairs,
                    "interpretable": n_pairs >= 10,
                })
                logger.info(f"  Spearman τ({cfg_a}, {cfg_b}) = {tau:.4f} (n={n_pairs})")

        spearman_df = pd.DataFrame(spearman_rows)
        spearman_path = self.output_dir / "spearman_ranking_agreement.csv"
        spearman_df.to_csv(spearman_path, index=False)

        logger.info(f"Cross-dataset FMD saved to {out_path}")
        logger.info(f"Spearman agreement saved to {spearman_path}")
        return df, spearman_df

    # ─── Step 5: Perturbation Sensitivity ───────────────────────────────

    def run_perturbation_sensitivity(self, dataset_name: str = "maestro") -> pd.DataFrame:
        """Step 5: Compute FMD(original, perturbed) for each perturbation × config.

        This reveals the 'sensitivity profile' of each configuration.
        Key expected insight: MTF preserves velocity; ABC/REMI may not.

        Args:
            dataset_name: Dataset to use for perturbation analysis (default: maestro)
        """
        from tqdm import tqdm

        logger.info("=" * 60)
        logger.info("STEP 5: Perturbation Sensitivity Profiling")
        logger.info(f"  Dataset: {dataset_name}")
        logger.info("=" * 60)

        midi_files = self._load_dataset_midi(dataset_name)
        if not midi_files:
            logger.error(f"No MIDI files for dataset '{dataset_name}'")
            return pd.DataFrame()

        # Get the 'original' perturbation (no changes)
        original_pert = self.perturbations[0]  # Should be "original"
        assert original_pert.name == "original", f"First perturbation should be 'original', got '{original_pert.name}'"

        pb_cfg = self.pivot_cfg.get("perturbation_bootstrap", {})
        n_boot = int(pb_cfg.get("n_resamples", 100))
        n_perm = int(pb_cfg.get("n_permutations", 200))
        rng = np.random.default_rng(int(pb_cfg.get("seed", self.seed)))

        # Incremental checkpointing: a multi-hour run must not lose every
        # completed cell to a crash, so the CSV is rewritten after each config.
        if dataset_name == "maestro":
            out_path = self.output_dir / "perturbation_sensitivity.csv"
        else:
            out_path = self.output_dir / f"perturbation_sensitivity_{dataset_name}.csv"

        rows = []

        for cfg in tqdm(self.configurations, desc="Perturbation sensitivity"):
            # Extract original embeddings
            emb_original = self._extract_embeddings(midi_files, cfg, original_pert)
            if emb_original.size == 0:
                logger.warning(f"No original embeddings for {cfg.name}")
                continue

            # Per-configuration noise floor (null distribution) — computed once.
            null_dist = self._noise_floor_distribution(emb_original, n_boot, rng)
            noise_floor_mean = float(null_dist.mean())
            noise_floor_ci_upper = float(np.percentile(null_dist, 97.5))

            for pert in self.perturbations[1:]:  # Skip "original"
                emb_perturbed = self._extract_embeddings(midi_files, cfg, pert)
                if emb_perturbed.size == 0:
                    logger.warning(f"No perturbed embeddings for {cfg.name}/{pert.name}")
                    continue

                fmd_val = float(self.fmd.compute_fmd(emb_original, emb_perturbed))

                # Bootstrap CI of the perturbation FMD and significance test
                # against the noise floor (does the perturbation move the
                # distribution more than within-dataset resampling does?).
                boot_dist = self._perturbation_bootstrap(
                    emb_original, emb_perturbed, n_boot, rng
                )
                snr = fmd_val / noise_floor_mean if noise_floor_mean > 1e-12 else float("nan")
                # Significance via a proper two-sample permutation test; the
                # split-half floor is kept only as the SNR effect-size scale.
                perm_p = self._permutation_pvalue(
                    emb_original, emb_perturbed, fmd_val, n_perm, rng
                )
                # "Detected" = systematically different (permutation-significant)
                # AND larger than within-dataset sampling noise (SNR >= 1).
                detected = bool(perm_p < 0.05 and snr >= 1.0)

                rows.append({
                    "dataset": dataset_name,
                    "configuration": cfg.name,
                    "perturbation": pert.name,
                    "description": pert.description,
                    "fmd_vs_original": fmd_val,
                    "fmd_boot_mean": float(boot_dist.mean()),
                    "fmd_ci_lower": float(np.percentile(boot_dist, 2.5)),
                    "fmd_ci_upper": float(np.percentile(boot_dist, 97.5)),
                    "noise_floor_mean": noise_floor_mean,
                    "noise_floor_ci_upper": noise_floor_ci_upper,
                    "snr": snr,
                    "perm_p_value": perm_p,
                    "significant": detected,
                    "n_original": emb_original.shape[0],
                    "n_perturbed": emb_perturbed.shape[0],
                    "n_bootstrap": n_boot,
                    "n_permutations": n_perm,
                })
                logger.info(
                    f"  {cfg.name} | {pert.name}: FMD={fmd_val:.4f} "
                    f"SNR={snr:.2f} perm_p={perm_p:.3f} {'DET' if detected else 'ns'}"
                )

            # checkpoint after each completed configuration
            pd.DataFrame(rows).to_csv(out_path, index=False)
            logger.info(f"  [checkpoint] {len(rows)} rows -> {out_path}")

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        logger.info(f"Perturbation sensitivity saved to {out_path}")
        return df

    # ─── Step 5b: Per-file paired analysis ──────────────────────────────

    @staticmethod
    def _holm(p_values: List[float]) -> List[float]:
        """Holm step-down correction; returns adjusted p-values in input order."""
        m = len(p_values)
        order = np.argsort(p_values)
        adjusted = [0.0] * m
        running_max = 0.0
        for rank, idx in enumerate(order):
            adj = min(1.0, (m - rank) * p_values[idx])
            running_max = max(running_max, adj)
            adjusted[idx] = running_max
        return adjusted

    def run_paired_file_analysis(self, dataset_name: str = "maestro") -> pd.DataFrame:
        """Per-file paired perturbation analysis (cosine shifts + Wilcoxon).

        Complements the distributional FMD study with a sample-level view:
        for every file we measure the cosine distance between its original
        and perturbed embedding, giving N paired observations per
        configuration x perturbation. This sidesteps the n << d covariance
        estimation entirely and shows the signal is not an artifact of a few
        outlier files.

        Crucially, each file is also encoded a SECOND time without any
        perturbation (the "retest" condition). The retest distance is the
        per-file encoding-noise floor: any nondeterministic input
        conversion shows up as a non-zero retest distance (this is how the
        broken music21-based ABC export was caught), and a perturbation
        only counts as a file-level effect if it moves the embedding MORE
        than re-encoding the same unperturbed file does. All contrasts are Wilcoxon signed-rank tests, Holm
        corrected per dataset:
          - per configuration x perturbation: shift > retest (primary);
          - per configuration: velocity shift > tempo shift (secondary);
          - the control contrast: within CLaMP-2 (same model, same space),
            retest-corrected velocity shifts under MTF vs under ABC.

        Files are aligned across conditions per configuration: a file that
        fails under any condition is dropped from all of them.
        """
        from scipy.stats import wilcoxon

        def _safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
            try:
                stat, p = wilcoxon(a, b, alternative="greater")
                return float(stat), float(p)
            except ValueError:  # all differences zero
                return float("nan"), 1.0

        logger.info("=" * 60)
        logger.info("STEP 5b: Per-file Paired Analysis (with retest noise floor)")
        logger.info(f"  Dataset: {dataset_name}")
        logger.info("=" * 60)

        midi_files = self._load_dataset_midi(dataset_name)
        if not midi_files:
            logger.error(f"No MIDI files for dataset '{dataset_name}'")
            return pd.DataFrame()

        assert self.perturbations[0].name == "original"
        original_pert = self.perturbations[0]
        # original encoded twice (retest) + the four perturbations
        conditions = [("original", original_pert), ("__retest__", original_pert)] + [
            (p.name, p) for p in self.perturbations[1:]
        ]
        suffix = "" if dataset_name == "maestro" else f"_{dataset_name}"
        shifts_path = self.output_dir / f"paired_file_shifts{suffix}.csv"
        tests_path = self.output_dir / f"paired_file_tests{suffix}.csv"

        shift_rows: List[Dict] = []
        # per config: {condition: {file_name: shift}}, for the contrasts
        shifts_by_cfg: Dict[str, Dict[str, Dict[str, float]]] = {}

        for cfg in self.configurations:
            per_cond_embs: Dict[str, List[np.ndarray]] = {name: [] for name, _ in conditions}
            kept_files: List[str] = []

            for midi_path in midi_files:
                embs_this_file: Dict[str, np.ndarray] = {}
                ok = True
                for cond_name, pert in conditions:
                    emb = self._extract_embeddings([midi_path], cfg, pert)
                    if emb.size == 0:
                        ok = False
                        break
                    embs_this_file[cond_name] = emb[0]
                if ok:
                    kept_files.append(midi_path.name)
                    for name, vec in embs_this_file.items():
                        per_cond_embs[name].append(vec)

            n_kept = len(kept_files)
            if n_kept < 10:
                logger.warning(f"{cfg.name}: only {n_kept} aligned files, skipping")
                continue

            orig = np.stack(per_cond_embs["original"])
            orig_unit = orig / (np.linalg.norm(orig, axis=1, keepdims=True) + 1e-12)
            # corpus angular scale: median cosine distance between distinct files
            sim = orig_unit @ orig_unit.T
            iu = np.triu_indices(n_kept, k=1)
            corpus_scale = float(np.median(1.0 - sim[iu]))

            shifts_by_cfg[cfg.name] = {}
            for cond_name, _ in conditions[1:]:  # retest + perturbations
                arr = np.stack(per_cond_embs[cond_name])
                arr_unit = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
                shifts = 1.0 - np.sum(orig_unit * arr_unit, axis=1)
                shifts_by_cfg[cfg.name][cond_name] = dict(zip(kept_files, shifts))
                for fname, s in zip(kept_files, shifts):
                    shift_rows.append({
                        "dataset": dataset_name,
                        "configuration": cfg.name,
                        "perturbation": cond_name,
                        "file": fname,
                        "cosine_shift": float(s),
                        "corpus_scale": corpus_scale,
                        "normalized_shift": float(s / corpus_scale) if corpus_scale > 1e-12 else float("nan"),
                    })
                logger.info(
                    f"  {cfg.name} | {cond_name}: median shift = {np.median(shifts):.5f} "
                    f"({np.median(shifts) / corpus_scale:.2f}x corpus scale, n={n_kept})"
                )

            # checkpoint after each configuration
            pd.DataFrame(shift_rows).to_csv(shifts_path, index=False)
            logger.info(f"  [checkpoint] {len(shift_rows)} rows -> {shifts_path}")

        # ── paired contrasts ────────────────────────────────────────────
        test_rows: List[Dict] = []

        def _paired(by_cond: Dict[str, Dict[str, float]], cond_a: str, cond_b: str):
            common = sorted(set(by_cond[cond_a]) & set(by_cond[cond_b]))
            a = np.array([by_cond[cond_a][f] for f in common])
            b = np.array([by_cond[cond_b][f] for f in common])
            return common, a, b

        for cfg_name, by_cond in shifts_by_cfg.items():
            if "__retest__" not in by_cond:
                continue
            # primary: each perturbation vs the retest noise floor
            for pert in self.perturbations[1:]:
                if pert.name not in by_cond:
                    continue
                common, a, b = _paired(by_cond, pert.name, "__retest__")
                stat, p = _safe_wilcoxon(a, b)
                test_rows.append({
                    "dataset": dataset_name,
                    "contrast": "exceeds_retest_noise",
                    "configuration": cfg_name,
                    "perturbation": pert.name,
                    "n_files": len(common),
                    "median_shift": float(np.median(a)),
                    "median_ref": float(np.median(b)),
                    "wilcoxon_stat": stat,
                    "p_raw": p,
                })
            # secondary: velocity vs tempo
            if "no_velocity" in by_cond and "constant_tempo" in by_cond:
                common, a, b = _paired(by_cond, "no_velocity", "constant_tempo")
                stat, p = _safe_wilcoxon(a, b)
                test_rows.append({
                    "dataset": dataset_name,
                    "contrast": "velocity_gt_tempo",
                    "configuration": cfg_name,
                    "perturbation": "no_velocity",
                    "n_files": len(common),
                    "median_shift": float(np.median(a)),
                    "median_ref": float(np.median(b)),
                    "wilcoxon_stat": stat,
                    "p_raw": p,
                })

        # control contrast on retest-corrected (excess) shifts, same model space
        if "CLaMP2-MTF" in shifts_by_cfg and "CLaMP2-ABC" in shifts_by_cfg:
            excess = {}
            for name in ("CLaMP2-MTF", "CLaMP2-ABC"):
                by_cond = shifts_by_cfg[name]
                if "no_velocity" in by_cond and "__retest__" in by_cond:
                    common = set(by_cond["no_velocity"]) & set(by_cond["__retest__"])
                    excess[name] = {
                        f: by_cond["no_velocity"][f] - by_cond["__retest__"][f] for f in common
                    }
            if len(excess) == 2:
                common = sorted(set(excess["CLaMP2-MTF"]) & set(excess["CLaMP2-ABC"]))
                if len(common) >= 10:
                    a = np.array([excess["CLaMP2-MTF"][f] for f in common])
                    b = np.array([excess["CLaMP2-ABC"][f] for f in common])
                    stat, p = _safe_wilcoxon(a, b)
                    test_rows.append({
                        "dataset": dataset_name,
                        "contrast": "control_mtf_gt_abc_velocity_excess",
                        "configuration": "CLaMP2-MTF_vs_CLaMP2-ABC",
                        "perturbation": "no_velocity",
                        "n_files": len(common),
                        "median_shift": float(np.median(a)),
                        "median_ref": float(np.median(b)),
                        "wilcoxon_stat": stat,
                        "p_raw": p,
                    })

        if test_rows:
            adjusted = self._holm([r["p_raw"] for r in test_rows])
            for row, p_holm in zip(test_rows, adjusted):
                row["p_holm"] = p_holm
                row["significant"] = bool(p_holm < 0.05)
                logger.info(
                    f"  TEST {row['contrast']} | {row['configuration']}: "
                    f"p_raw={row['p_raw']:.2e} p_holm={row['p_holm']:.2e} "
                    f"{'SIG' if row['significant'] else 'ns'}"
                )

        tests_df = pd.DataFrame(test_rows)
        tests_df.to_csv(tests_path, index=False)
        logger.info(f"Paired tests saved to {tests_path}")
        return tests_df

    # ─── Step 6: Bootstrap Stability ────────────────────────────────────

    def run_bootstrap_stability(self) -> pd.DataFrame:
        """Step 6: Bootstrap CI for one dataset pair across configurations.

        Resample N times, compute FMD each time, report mean ± std.
        Shows which configuration is more stable with limited data.
        """
        from tqdm import tqdm

        logger.info("=" * 60)
        logger.info("STEP 6: Bootstrap Stability Analysis")
        logger.info("=" * 60)

        bs_cfg = self.bootstrap_cfg
        n_resamples = int(bs_cfg.get("n_resamples", 10))
        sample_size = int(bs_cfg.get("sample_size", 500))
        bs_seed = int(bs_cfg.get("seed", self.seed))
        pair = bs_cfg.get("dataset_pair", ["maestro", "pop909"])

        ds_a_name, ds_b_name = pair[0], pair[1]
        logger.info(f"  Pair: {ds_a_name} vs {ds_b_name}, N={sample_size}, resamples={n_resamples}")

        midi_a = self._load_dataset_midi(ds_a_name)
        midi_b = self._load_dataset_midi(ds_b_name)

        if not midi_a or not midi_b:
            logger.error("Cannot load datasets for bootstrap")
            return pd.DataFrame()

        rows = []
        rng = np.random.default_rng(bs_seed)

        for cfg in tqdm(self.configurations, desc="Bootstrap stability"):
            # Extract all embeddings first
            emb_a_full = self._extract_embeddings(midi_a, cfg)
            emb_b_full = self._extract_embeddings(midi_b, cfg)

            if emb_a_full.size == 0 or emb_b_full.size == 0:
                logger.warning(f"Empty embeddings for bootstrap: {cfg.name}")
                continue

            actual_sample_a = min(sample_size, emb_a_full.shape[0])
            actual_sample_b = min(sample_size, emb_b_full.shape[0])

            fmd_values = []
            for i in range(n_resamples):
                idx_a = rng.choice(emb_a_full.shape[0], size=actual_sample_a, replace=True)
                idx_b = rng.choice(emb_b_full.shape[0], size=actual_sample_b, replace=True)
                fmd_val = float(self.fmd.compute_fmd(emb_a_full[idx_a], emb_b_full[idx_b]))
                fmd_values.append(fmd_val)

            fmd_arr = np.array(fmd_values)
            ci_lower = float(np.percentile(fmd_arr, 2.5))
            ci_upper = float(np.percentile(fmd_arr, 97.5))

            rows.append({
                "configuration": cfg.name,
                "dataset_a": ds_a_name,
                "dataset_b": ds_b_name,
                "n_resamples": n_resamples,
                "sample_size_a": actual_sample_a,
                "sample_size_b": actual_sample_b,
                "fmd_mean": float(fmd_arr.mean()),
                "fmd_std": float(fmd_arr.std()),
                "fmd_ci_lower": ci_lower,
                "fmd_ci_upper": ci_upper,
                "cv": float(fmd_arr.std() / max(fmd_arr.mean(), 1e-8)),
            })
            logger.info(
                f"  {cfg.name}: FMD = {fmd_arr.mean():.4f} ± {fmd_arr.std():.4f} "
                f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            )

        df = pd.DataFrame(rows)
        out_path = self.output_dir / "bootstrap_stability.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Bootstrap stability saved to {out_path}")
        return df

    # ─── Step 7: Synthesis — Generate Summary Tables/Plots ──────────────

    def generate_summary(self, results: SensitivityResult) -> Dict:
        """Step 7: Generate summary JSON with all key findings."""
        summary = {
            "configurations": [c.name for c in self.configurations],
            "datasets": self.datasets,
            "perturbations": [p.name for p in self.perturbations],
        }

        # Add self-similarity summary
        if results.self_similarity:
            summary["self_similarity"] = results.self_similarity

        # Add cross-dataset summary
        if results.cross_dataset_fmd:
            summary["cross_dataset_fmd"] = results.cross_dataset_fmd

        # Add spearman summary
        if results.spearman_matrix:
            summary["spearman_agreement"] = results.spearman_matrix

        # Add perturbation profile summary
        if results.perturbation_profiles:
            summary["perturbation_sensitivity"] = results.perturbation_profiles

        # Add bootstrap summary
        if results.bootstrap_stability:
            summary["bootstrap_stability"] = results.bootstrap_stability

        out_path = self.output_dir / "sensitivity_pivot_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved to {out_path}")
        return summary

    def generate_plots(self):
        """Generate publication-quality plots from saved CSV results."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping plots")
            return

        # Plot 1: Perturbation sensitivity heatmap
        pert_path = self.output_dir / "perturbation_sensitivity.csv"
        if pert_path.exists():
            df = pd.read_csv(pert_path)
            pivot = df.pivot_table(index="perturbation", columns="configuration", values="fmd_vs_original")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
            ax.set_title("Perturbation Sensitivity Profile\n(FMD: original vs perturbed)")
            ax.set_ylabel("Perturbation")
            ax.set_xlabel("Configuration")
            plt.tight_layout()
            fig.savefig(self.plots_dir / "perturbation_heatmap.png", dpi=150)
            plt.close(fig)
            logger.info("Plot saved: perturbation_heatmap.png")

        # Plot 2: Cross-dataset FMD bar chart
        cross_path = self.output_dir / "cross_dataset_fmd.csv"
        if cross_path.exists():
            df = pd.read_csv(cross_path)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=df, x="pair", y="fmd", hue="configuration", ax=ax)
            ax.set_title("Cross-Dataset FMD by Configuration")
            ax.set_xlabel("Dataset Pair")
            ax.set_ylabel("FMD")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig.savefig(self.plots_dir / "cross_dataset_bar.png", dpi=150)
            plt.close(fig)
            logger.info("Plot saved: cross_dataset_bar.png")

        # Plot 3: Bootstrap stability
        bs_path = self.output_dir / "bootstrap_stability.csv"
        if bs_path.exists():
            df = pd.read_csv(bs_path)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(df["configuration"], df["fmd_mean"], yerr=df["fmd_std"], capsize=5, color="steelblue")
            ax.set_title("Bootstrap Stability (FMD mean ± std)")
            ax.set_ylabel("FMD")
            ax.set_xlabel("Configuration")
            plt.tight_layout()
            fig.savefig(self.plots_dir / "bootstrap_stability.png", dpi=150)
            plt.close(fig)
            logger.info("Plot saved: bootstrap_stability.png")

        # Plot 4: Self-similarity
        ss_path = self.output_dir / "self_similarity.csv"
        if ss_path.exists():
            df = pd.read_csv(ss_path)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=df, x="dataset", y="split_half_fmd", hue="configuration", ax=ax)
            ax.set_title("Self-Similarity: Split-Half FMD (should be ≈ 0)")
            ax.set_ylabel("Split-Half FMD")
            ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            plt.tight_layout()
            fig.savefig(self.plots_dir / "self_similarity.png", dpi=150)
            plt.close(fig)
            logger.info("Plot saved: self_similarity.png")

    # ─── Main Runner ────────────────────────────────────────────────────

    def run_all(self) -> SensitivityResult:
        """Run the complete 7-step sensitivity pivot pipeline."""
        logger.info("=" * 60)
        logger.info("SENSITIVITY PIVOT - Full Pipeline")
        logger.info("=" * 60)

        result = SensitivityResult()

        # Step 3: Self-similarity
        try:
            ss_df = self.run_self_similarity()
            result.self_similarity = ss_df.to_dict(orient="records") if not ss_df.empty else {}
        except Exception as e:
            logger.error(f"Step 3 (self-similarity) failed: {e}")

        # Step 4: Cross-dataset ranking
        try:
            cross_df, spearman_df = self.run_cross_dataset_ranking()
            result.cross_dataset_fmd = cross_df.to_dict(orient="records") if not cross_df.empty else {}
            result.spearman_matrix = spearman_df.to_dict(orient="records") if not spearman_df.empty else {}
        except Exception as e:
            logger.error(f"Step 4 (cross-dataset ranking) failed: {e}")

        # Step 5: Perturbation sensitivity
        try:
            pert_df = self.run_perturbation_sensitivity(dataset_name="maestro")
            result.perturbation_profiles = pert_df.to_dict(orient="records") if not pert_df.empty else {}
        except Exception as e:
            logger.error(f"Step 5 (perturbation sensitivity) failed: {e}")

        # Step 6: Bootstrap stability
        try:
            bs_df = self.run_bootstrap_stability()
            result.bootstrap_stability = bs_df.to_dict(orient="records") if not bs_df.empty else {}
        except Exception as e:
            logger.error(f"Step 6 (bootstrap stability) failed: {e}")

        # Step 7: Synthesis
        try:
            self.generate_summary(result)
            self.generate_plots()
        except Exception as e:
            logger.error(f"Step 7 (synthesis) failed: {e}")

        logger.info("=" * 60)
        logger.info("SENSITIVITY PIVOT PIPELINE COMPLETE")
        logger.info(f"  Results: {self.output_dir}")
        logger.info(f"  Plots:   {self.plots_dir}")
        logger.info("=" * 60)

        return result
