"""Paper-oriented experiment pipeline for FMD sensitivity analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import md5
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import csv
import json

import numpy as np
import yaml
from loguru import logger
from scipy.stats import kendalltau, spearmanr

from data.manager import DatasetManager
from embeddings.extractor import EmbeddingExtractor
from metrics.fmd import FMDRanking, FrechetMusicDistance
from preprocessing.processor import MIDIPreprocessor
from tokenization.tokenizer import TokenizationPipeline


@dataclass(frozen=True)
class PipelineVariant:
    """Single pipeline configuration used in experiments."""

    tokenizer: str
    model: str
    remove_velocity: bool
    hard_quantization: bool

    @property
    def name(self) -> str:
        """Compact string label for reports."""
        return (
            f"tok={self.tokenizer}|model={self.model}|"
            f"vel={'off' if self.remove_velocity else 'on'}|"
            f"quant={'on' if self.hard_quantization else 'off'}"
        )


class PaperExperimentRunner:
    """Runs comparisons needed for an article draft."""

    def __init__(self, config: Dict):
        self.config = config
        self.dataset_manager = DatasetManager(config)
        self.preprocessor = MIDIPreprocessor(config)
        self.tokenization = TokenizationPipeline(config)
        self.embeddings = EmbeddingExtractor(config)
        self.fmd = FrechetMusicDistance(config)

        paper_cfg = config.get("paper", {})
        self.max_files = int(paper_cfg.get("max_files_per_dataset", 8))
        self.synthetic_fallback_samples = int(paper_cfg.get("synthetic_fallback_samples", 12))
        self.seed = int(paper_cfg.get("seed", 42))
        self.dataset_names = self._dataset_names()
        self.compare_all_pairs = bool(paper_cfg.get("compare_all_pairs", False))
        self.fallback_mode = str(paper_cfg.get("fallback_mode", "synthetic")).strip().lower()
        self.bootstrap_cfg = paper_cfg.get("bootstrap_ci", {})
        self.bootstrap_enabled = bool(self.bootstrap_cfg.get("enabled", True))
        self.bootstrap_resamples = int(self.bootstrap_cfg.get("n_resamples", 50))
        self.bootstrap_ci = float(self.bootstrap_cfg.get("confidence", 0.95))
        self.bootstrap_seed = int(self.bootstrap_cfg.get("seed", self.seed))

        self.genre_aliases = self._load_genre_aliases(paper_cfg)
        self.top_variants_per_pair = int(paper_cfg.get("top_variants_per_pair", 3))

        reports_dir = config.get("results", {}).get("reports_dir", "results/reports")
        self.output_dir = Path(reports_dir) / "paper"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_variants(
        self,
        tokenizers: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        preprocessing_grid: Sequence[Tuple[bool, bool]] | None = None,
    ) -> List[PipelineVariant]:
        """Build experiment grid tokenizer x model x preprocessing."""
        if tokenizers is None:
            tokenizers = [t["type"] for t in self.config["tokenization"]["tokenizers"]]
        if models is None:
            models = [m["name"] for m in self.config["embeddings"]["models"]]
        if preprocessing_grid is None:
            preprocessing_grid = [(False, False), (True, False), (False, True), (True, True)]

        variants = [
            PipelineVariant(tok, model, remove_vel, hard_quant)
            for tok, model, (remove_vel, hard_quant) in product(tokenizers, models, preprocessing_grid)
        ]
        logger.info(f"Built {len(variants)} variants")
        return variants

    @staticmethod
    def _parse_pairs(raw_pairs: Iterable) -> List[Tuple[str, str]]:
        """Parse YAML pairs that can be tuple-like strings or arrays."""
        pairs: List[Tuple[str, str]] = []
        for entry in raw_pairs:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                pairs.append((str(entry[0]), str(entry[1])))
                continue

            text = str(entry).strip().strip("()")
            if "," not in text:
                continue
            left, right = [p.strip().strip("'\"") for p in text.split(",", 1)]
            if left and right:
                pairs.append((left, right))
        return pairs

    def _dataset_names(self) -> List[str]:
        return [d["name"] for d in self.config["data"]["datasets"]]

    def _load_genre_aliases(self, paper_cfg: Dict) -> Dict[str, str]:
        """Load alias mapping from config and optional standalone YAML file."""
        aliases: Dict[str, str] = {}

        inline_aliases = paper_cfg.get("genre_aliases", {})
        if isinstance(inline_aliases, dict):
            aliases.update(
                {
                    str(alias).strip().lower(): str(dataset).strip()
                    for alias, dataset in inline_aliases.items()
                    if str(alias).strip() and str(dataset).strip()
                }
            )

        mapping_file = paper_cfg.get("genre_mapping_file")
        if not mapping_file:
            return aliases

        mapping_path = Path(mapping_file)
        if not mapping_path.exists():
            logger.warning(f"Genre mapping file not found: {mapping_path}")
            return aliases

        with open(mapping_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        file_aliases = payload.get("aliases", payload)
        if isinstance(file_aliases, dict):
            for alias, dataset in file_aliases.items():
                alias_text = str(alias).strip().lower()
                dataset_text = str(dataset).strip()
                if alias_text and dataset_text:
                    aliases[alias_text] = dataset_text

        return aliases

    def _resolve_dataset_name(self, name_or_alias: str) -> str | None:
        label = str(name_or_alias).strip()
        if not label:
            return None

        label_lower = label.lower()
        if label in self.dataset_names:
            return label
        if label_lower in self.genre_aliases:
            resolved = self.genre_aliases[label_lower]
            return resolved if resolved in self.dataset_names else None
        if label_lower in [x.lower() for x in self.dataset_names]:
            by_lower = {x.lower(): x for x in self.dataset_names}
            return by_lower[label_lower]
        return None

    def _resolve_special_pairs(self) -> List[Dict]:
        paper_cfg = self.config.get("paper", {})
        parsed = self._parse_pairs(paper_cfg.get("special_pairs", []))

        resolved: List[Dict] = []
        for left, right in parsed:
            ds_left = self._resolve_dataset_name(left)
            ds_right = self._resolve_dataset_name(right)
            if not ds_left or not ds_right:
                logger.warning(f"Skipping special pair ({left}, {right}) - unresolved dataset alias")
                continue
            resolved.append(
                {
                    "genre_a": left,
                    "genre_b": right,
                    "dataset_a": ds_left,
                    "dataset_b": ds_right,
                }
            )

        return resolved

    def _list_dataset_midis(self, dataset_name: str) -> List[Path]:
        files = self.dataset_manager.list_midi_files(dataset_name, processed=False, limit=self.max_files)
        if files:
            return files

        # Fallback: some repos keep MIDI files directly in data/raw.
        raw_root = Path(self.config["data"]["raw_data_dir"])
        flat = sorted(list(raw_root.glob("*.mid")) + list(raw_root.glob("*.midi")))
        return flat[: self.max_files]

    def _preprocess_midi_file(self, midi_path: Path, variant: PipelineVariant):
        midi_data = self.preprocessor.load_midi(midi_path)
        if midi_data is None:
            return None
        if variant.remove_velocity:
            midi_data = self.preprocessor.remove_velocity(midi_data)
        if variant.hard_quantization:
            midi_data = self.preprocessor.quantize_time(midi_data)
        midi_data = self.preprocessor.filter_note_range(midi_data)
        midi_data = self.preprocessor.normalize_instruments(midi_data)
        return midi_data

    def _synthetic_embeddings(self, dataset_name: str, variant: PipelineVariant, dim: int = 512) -> np.ndarray:
        key = f"{dataset_name}|{variant.name}|{self.seed}"
        seed = int(md5(key.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1.0, size=(self.synthetic_fallback_samples, dim)).astype(np.float32)

    @staticmethod
    def _normalize_embedding_payload(payload) -> Dict:
        if isinstance(payload, dict):
            return {
                "embeddings": payload.get("embeddings"),
                "source": str(payload.get("source", "real")),
                "real_files": int(payload.get("real_files", 0)),
                "total_files": int(payload.get("total_files", 0)),
            }
        return {
            "embeddings": payload,
            "source": "real",
            "real_files": int(payload.shape[0]) if payload is not None else 0,
            "total_files": int(payload.shape[0]) if payload is not None else 0,
        }

    def _bootstrap_ci_for_pair(self, emb_a: np.ndarray, emb_b: np.ndarray, key: str) -> Dict:
        if not self.bootstrap_enabled or emb_a.size == 0 or emb_b.size == 0:
            return {"mean": None, "std": None, "ci_lower": None, "ci_upper": None}

        local_seed = int(md5(f"{key}|{self.bootstrap_seed}".encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(local_seed)
        n_a = emb_a.shape[0]
        n_b = emb_b.shape[0]
        sample_n_a = max(2, min(n_a, self.synthetic_fallback_samples))
        sample_n_b = max(2, min(n_b, self.synthetic_fallback_samples))
        values = []

        for _ in range(self.bootstrap_resamples):
            idx_a = rng.integers(0, n_a, size=sample_n_a)
            idx_b = rng.integers(0, n_b, size=sample_n_b)
            values.append(float(self.fmd.compute_fmd(emb_a[idx_a], emb_b[idx_b])))

        arr = np.array(values, dtype=float)
        alpha = max(0.0, min(0.5, (1.0 - self.bootstrap_ci) / 2.0))
        lower = float(np.quantile(arr, alpha))
        upper = float(np.quantile(arr, 1.0 - alpha))
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": lower,
            "ci_upper": upper,
        }

    def _extract_dataset_embeddings(self, dataset_name: str, variant: PipelineVariant) -> Dict:
        midi_files = self._list_dataset_midis(dataset_name)
        vectors: List[np.ndarray] = []
        real_count = 0

        for midi_path in midi_files:
            try:
                midi_data = self._preprocess_midi_file(midi_path, variant)
                if midi_data is None:
                    continue
                tokenizer = self.tokenization.tokenizers[variant.tokenizer]
                tokens = tokenizer.encode_midi_object(midi_data)
                if not tokens:
                    continue
                vec = self.embeddings.extract_embeddings([tokens], variant.model)[0]
                vectors.append(vec)
                real_count += 1
            except Exception as exc:
                logger.warning(f"Skipping {midi_path} for {variant.name}: {exc}")

        if not vectors:
            if self.fallback_mode == "strict":
                return {
                    "embeddings": None,
                    "source": "missing",
                    "real_files": 0,
                    "total_files": len(midi_files),
                }

            logger.warning(
                f"No embeddings from files for dataset={dataset_name}, variant={variant.name}; "
                "using deterministic synthetic fallback"
            )
            return {
                "embeddings": self._synthetic_embeddings(dataset_name, variant),
                "source": "synthetic",
                "real_files": 0,
                "total_files": len(midi_files),
            }

        return {
            "embeddings": np.vstack(vectors),
            "source": "real",
            "real_files": real_count,
            "total_files": len(midi_files),
        }

    def run_pairwise_benchmark(self, variants: Sequence[PipelineVariant]) -> List[Dict]:
        """Compute FMD for configured dataset pairs across all variants."""
        names = self._dataset_names()
        if self.compare_all_pairs:
            pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
        else:
            exp5 = self.config.get("experiments", {}).get("exp5_cross_genre", {})
            pairs = self._parse_pairs(exp5.get("pairs", []))
            if not pairs:
                pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

        rows: List[Dict] = []
        cache: Dict[Tuple[str, str], Dict] = {}

        for variant in variants:
            for ds_a, ds_b in pairs:
                key_a = (ds_a, variant.name)
                key_b = (ds_b, variant.name)
                if key_a not in cache:
                    payload_a = self._extract_dataset_embeddings(ds_a, variant)
                    cache[key_a] = self._normalize_embedding_payload(payload_a)
                if key_b not in cache:
                    payload_b = self._extract_dataset_embeddings(ds_b, variant)
                    cache[key_b] = self._normalize_embedding_payload(payload_b)

                emb_a = cache[key_a]["embeddings"]
                emb_b = cache[key_b]["embeddings"]
                is_valid = emb_a is not None and emb_b is not None
                real_pair = cache[key_a]["source"] == "real" and cache[key_b]["source"] == "real"
                bootstrap = {
                    "mean": None,
                    "std": None,
                    "ci_lower": None,
                    "ci_upper": None,
                }
                fmd_value = None

                if is_valid:
                    fmd_value = float(self.fmd.compute_fmd(emb_a, emb_b))
                    bootstrap = self._bootstrap_ci_for_pair(
                        emb_a,
                        emb_b,
                        key=f"{variant.name}|{ds_a}|{ds_b}",
                    )

                rows.append(
                    {
                        "variant": variant.name,
                        "tokenizer": variant.tokenizer,
                        "model": variant.model,
                        "remove_velocity": variant.remove_velocity,
                        "hard_quantization": variant.hard_quantization,
                        "dataset_a": ds_a,
                        "dataset_b": ds_b,
                        "fmd": fmd_value,
                        "valid": is_valid,
                        "real_pair": real_pair,
                        "source_a": cache[key_a]["source"],
                        "source_b": cache[key_b]["source"],
                        "real_files_a": cache[key_a]["real_files"],
                        "real_files_b": cache[key_b]["real_files"],
                        "bootstrap_mean": bootstrap["mean"],
                        "bootstrap_std": bootstrap["std"],
                        "bootstrap_ci_lower": bootstrap["ci_lower"],
                        "bootstrap_ci_upper": bootstrap["ci_upper"],
                    }
                )

        return rows

    @staticmethod
    def _split_pairwise_rows(pairwise_rows: List[Dict]) -> Dict[str, List[Dict]]:
        valid_rows = [row for row in pairwise_rows if row.get("valid") and row.get("fmd") is not None]
        real_only_rows = [row for row in valid_rows if row.get("real_pair")]
        return {
            "all": valid_rows,
            "real_only": real_only_rows,
        }

    def compute_variant_effects(self, pairwise_rows: List[Dict]) -> Dict:
        """Compute per-variant deltas for tokenizer/model under fixed controls."""
        filtered = [row for row in pairwise_rows if row.get("valid") and row.get("fmd") is not None]
        if not filtered:
            return {"tokenizer_deltas": [], "model_deltas": []}

        by_cell: Dict[Tuple[str, str, bool, bool], List[float]] = defaultdict(list)
        for row in filtered:
            key = (
                str(row["tokenizer"]),
                str(row["model"]),
                bool(row["remove_velocity"]),
                bool(row["hard_quantization"]),
            )
            by_cell[key].append(float(row["fmd"]))

        cell_mean = {key: float(np.mean(values)) for key, values in by_cell.items()}

        tokenizer_deltas: List[Dict] = []
        model_deltas: List[Dict] = []

        tokenizers = sorted({key[0] for key in cell_mean})
        models = sorted({key[1] for key in cell_mean})
        preprocess = sorted({(key[2], key[3]) for key in cell_mean})

        for model in models:
            for remove_velocity, hard_quantization in preprocess:
                for i in range(len(tokenizers)):
                    for j in range(i + 1, len(tokenizers)):
                        t1 = tokenizers[i]
                        t2 = tokenizers[j]
                        k1 = (t1, model, remove_velocity, hard_quantization)
                        k2 = (t2, model, remove_velocity, hard_quantization)
                        if k1 in cell_mean and k2 in cell_mean:
                            tokenizer_deltas.append(
                                {
                                    "model": model,
                                    "remove_velocity": remove_velocity,
                                    "hard_quantization": hard_quantization,
                                    "tokenizer_a": t1,
                                    "tokenizer_b": t2,
                                    "mean_fmd_a": cell_mean[k1],
                                    "mean_fmd_b": cell_mean[k2],
                                    "delta_fmd": float(cell_mean[k1] - cell_mean[k2]),
                                }
                            )

        for tokenizer in tokenizers:
            for remove_velocity, hard_quantization in preprocess:
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        m1 = models[i]
                        m2 = models[j]
                        k1 = (tokenizer, m1, remove_velocity, hard_quantization)
                        k2 = (tokenizer, m2, remove_velocity, hard_quantization)
                        if k1 in cell_mean and k2 in cell_mean:
                            model_deltas.append(
                                {
                                    "tokenizer": tokenizer,
                                    "remove_velocity": remove_velocity,
                                    "hard_quantization": hard_quantization,
                                    "model_a": m1,
                                    "model_b": m2,
                                    "mean_fmd_a": cell_mean[k1],
                                    "mean_fmd_b": cell_mean[k2],
                                    "delta_fmd": float(cell_mean[k1] - cell_mean[k2]),
                                }
                            )

        return {
            "tokenizer_deltas": tokenizer_deltas,
            "model_deltas": model_deltas,
        }

    def run_ranking_benchmark(self, variants: Sequence[PipelineVariant]) -> Dict:
        """Build rankings and stability across configurations."""
        dataset_names = self._dataset_names()
        ranking_by_variant: Dict[str, Dict[str, List[int]]] = {}

        for variant in variants:
            emb_sets = []
            for ds_name in dataset_names:
                payload = self._normalize_embedding_payload(self._extract_dataset_embeddings(ds_name, variant))
                emb = payload["embeddings"]
                if emb is None:
                    if self.fallback_mode == "strict":
                        raise RuntimeError(
                            f"Missing real embeddings for dataset={ds_name}, variant={variant.name} in strict mode"
                        )
                    emb = self._synthetic_embeddings(ds_name, variant)
                emb_sets.append((ds_name, emb))

            matrix_result = self.fmd.compute_batch_fmd(emb_sets)
            matrix = matrix_result["fmd_matrix"]
            per_ref: Dict[str, List[int]] = {}
            for ref_idx, ref_name in enumerate(dataset_names):
                ranking = FMDRanking.rank_by_fmd(matrix, ref_idx)["ranking"]
                per_ref[ref_name] = [int(x) for x in ranking]
            ranking_by_variant[variant.name] = per_ref

        # Stability by reference dataset over all variants.
        stability: Dict[str, float] = {}
        for ref_name in dataset_names:
            rankings_dict = {
                variant_name: np.array(ref_rankings[ref_name], dtype=int)
                for variant_name, ref_rankings in ranking_by_variant.items()
            }
            stability[ref_name] = float(FMDRanking.compute_ranking_stability(rankings_dict))

        return {"rankings": ranking_by_variant, "stability": stability}

    def evaluate_expected_order(self, ranking_results: Dict) -> Dict:
        """Compare obtained rankings against expected similarity order from config."""
        expected_cfg = self.config.get("paper", {}).get("expected_orders", [])
        if not expected_cfg:
            return {"available": False, "details": []}

        dataset_names = self._dataset_names()
        idx_by_name = {name: i for i, name in enumerate(dataset_names)}
        details = []

        for variant_name, ranking_by_ref in ranking_results["rankings"].items():
            for expected in expected_cfg:
                ref = expected.get("reference")
                order = expected.get("order", [])
                if ref not in ranking_by_ref:
                    continue

                predicted_indices = ranking_by_ref[ref]
                predicted_names = [dataset_names[i] for i in predicted_indices]

                expected_rank = {name: rank for rank, name in enumerate(order)}
                pred_scores = []
                exp_scores = []
                for name in predicted_names:
                    if name in expected_rank:
                        pred_scores.append(len(pred_scores))
                        exp_scores.append(expected_rank[name])

                if len(pred_scores) >= 2:
                    sp = spearmanr(pred_scores, exp_scores).correlation
                    kd = kendalltau(pred_scores, exp_scores).correlation
                else:
                    sp = np.nan
                    kd = np.nan

                details.append(
                    {
                        "variant": variant_name,
                        "reference": ref,
                        "predicted_order": predicted_names,
                        "expected_order": order,
                        "spearman": float(sp) if sp == sp else None,
                        "kendall": float(kd) if kd == kd else None,
                    }
                )

        return {"available": True, "details": details}

    def compute_special_pair_metrics(self, pairwise_rows: List[Dict]) -> Dict:
        """Aggregate configured special genre pairs for publication-ready analysis."""
        special_pairs = self._resolve_special_pairs()
        if not special_pairs:
            return {"available": False, "rows": [], "summary": [], "top_variants": []}

        usable_rows = [row for row in pairwise_rows if row.get("valid") and row.get("fmd") is not None]
        if not usable_rows:
            return {"available": False, "rows": [], "summary": [], "top_variants": []}

        lookup: Dict[Tuple[str, str, str], float] = {}
        for row in usable_rows:
            key_direct = (row["variant"], row["dataset_a"], row["dataset_b"])
            key_reverse = (row["variant"], row["dataset_b"], row["dataset_a"])
            lookup[key_direct] = float(row["fmd"])
            lookup[key_reverse] = float(row["fmd"])

        rows: List[Dict] = []
        for pair in special_pairs:
            for variant_name in sorted({row["variant"] for row in usable_rows}):
                value = lookup.get((variant_name, pair["dataset_a"], pair["dataset_b"]))
                if value is None:
                    continue
                rows.append(
                    {
                        "variant": variant_name,
                        "genre_a": pair["genre_a"],
                        "genre_b": pair["genre_b"],
                        "pair": f"{pair['genre_a']} vs {pair['genre_b']}",
                        "dataset_a": pair["dataset_a"],
                        "dataset_b": pair["dataset_b"],
                        "fmd": float(value),
                    }
                )

        if not rows:
            return {"available": False, "rows": [], "summary": [], "top_variants": []}

        grouped: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            grouped[row["pair"]].append(float(row["fmd"]))

        global_mean = float(np.mean([row["fmd"] for row in rows]))
        summary: List[Dict] = []
        for pair_name, values in grouped.items():
            arr = np.array(values, dtype=float)
            summary.append(
                {
                    "pair": pair_name,
                    "count": int(arr.size),
                    "mean_fmd": float(np.mean(arr)),
                    "std_fmd": float(np.std(arr)),
                    "min_fmd": float(np.min(arr)),
                    "max_fmd": float(np.max(arr)),
                    # >1.0 means this pair is more separable than average configured pair.
                    "distinguishability_ratio": float(np.mean(arr) / max(global_mean, 1e-9)),
                }
            )

        summary.sort(key=lambda item: item["mean_fmd"], reverse=True)

        top_rows: List[Dict] = []
        for pair_name in sorted(grouped.keys()):
            pair_rows = sorted(
                [row for row in rows if row["pair"] == pair_name],
                key=lambda item: float(item["fmd"]),
                reverse=True,
            )
            for rank, row in enumerate(pair_rows[: self.top_variants_per_pair], start=1):
                top_rows.append(
                    {
                        "pair": pair_name,
                        "rank": rank,
                        "variant": row["variant"],
                        "fmd": float(row["fmd"]),
                        "dataset_a": row["dataset_a"],
                        "dataset_b": row["dataset_b"],
                    }
                )

        return {"available": True, "rows": rows, "summary": summary, "top_variants": top_rows}

    def save_outputs(
        self,
        pairwise_rows: List[Dict],
        ranking_results: Dict,
        expected_eval: Dict,
        special_metrics: Dict | None = None,
        variant_effects: Dict | None = None,
    ) -> Dict:
        """Save JSON, CSV and markdown summary for paper draft."""
        json_path = self.output_dir / "paper_results.json"
        csv_path = self.output_dir / "pairwise_fmd.csv"
        csv_all_path = self.output_dir / "pairwise_fmd_all.csv"
        csv_real_only_path = self.output_dir / "pairwise_fmd_real_only.csv"
        md_path = self.output_dir / "paper_summary.md"
        special_csv_path = self.output_dir / "special_pair_fmd.csv"
        special_summary_csv_path = self.output_dir / "special_pair_summary.csv"
        special_top_csv_path = self.output_dir / "special_pair_top_variants.csv"
        tokenizer_delta_csv_path = self.output_dir / "variant_delta_tokenizer.csv"
        model_delta_csv_path = self.output_dir / "variant_delta_model.csv"

        if special_metrics is None:
            special_metrics = {"available": False, "rows": [], "summary": [], "top_variants": []}
        if variant_effects is None:
            variant_effects = {
                "all": {"tokenizer_deltas": [], "model_deltas": []},
                "real_only": {"tokenizer_deltas": [], "model_deltas": []},
            }

        split_rows = self._split_pairwise_rows(pairwise_rows)
        all_rows = split_rows["all"]
        real_only_rows = split_rows["real_only"]

        payload = {
            "pairwise_all": all_rows,
            "pairwise_real_only": real_only_rows,
            "ranking": ranking_results,
            "expected_eval": expected_eval,
            "special_pairs": special_metrics,
            "variant_effects": variant_effects,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        pairwise_fields = [
            "variant",
            "tokenizer",
            "model",
            "remove_velocity",
            "hard_quantization",
            "dataset_a",
            "dataset_b",
            "fmd",
            "valid",
            "real_pair",
            "source_a",
            "source_b",
            "real_files_a",
            "real_files_b",
            "bootstrap_mean",
            "bootstrap_std",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=pairwise_fields)
            writer.writeheader()
            writer.writerows(all_rows)

        with open(csv_all_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=pairwise_fields)
            writer.writeheader()
            writer.writerows(all_rows)

        with open(csv_real_only_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=pairwise_fields)
            writer.writeheader()
            writer.writerows(real_only_rows)

        with open(special_csv_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = ["variant", "genre_a", "genre_b", "pair", "dataset_a", "dataset_b", "fmd"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(special_metrics.get("rows", []))

        with open(special_summary_csv_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "pair",
                "count",
                "mean_fmd",
                "std_fmd",
                "min_fmd",
                "max_fmd",
                "distinguishability_ratio",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(special_metrics.get("summary", []))

        with open(special_top_csv_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = ["pair", "rank", "variant", "fmd", "dataset_a", "dataset_b"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(special_metrics.get("top_variants", []))

        with open(tokenizer_delta_csv_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "model",
                "remove_velocity",
                "hard_quantization",
                "tokenizer_a",
                "tokenizer_b",
                "mean_fmd_a",
                "mean_fmd_b",
                "delta_fmd",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(variant_effects.get("all", {}).get("tokenizer_deltas", []))

        with open(model_delta_csv_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "tokenizer",
                "remove_velocity",
                "hard_quantization",
                "model_a",
                "model_b",
                "mean_fmd_a",
                "mean_fmd_b",
                "delta_fmd",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(variant_effects.get("all", {}).get("model_deltas", []))

        with open(self.output_dir / "variant_delta_tokenizer_real_only.csv", "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "model",
                "remove_velocity",
                "hard_quantization",
                "tokenizer_a",
                "tokenizer_b",
                "mean_fmd_a",
                "mean_fmd_b",
                "delta_fmd",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(variant_effects.get("real_only", {}).get("tokenizer_deltas", []))

        with open(self.output_dir / "variant_delta_model_real_only.csv", "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "tokenizer",
                "remove_velocity",
                "hard_quantization",
                "model_a",
                "model_b",
                "mean_fmd_a",
                "mean_fmd_b",
                "delta_fmd",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(variant_effects.get("real_only", {}).get("model_deltas", []))

        # Simple markdown summary for direct paper drafting.
        lines = [
            "# Paper-Oriented FMD Benchmark Summary",
            "",
            "## Pairwise comparisons",
            "",
            f"All valid rows: **{len(all_rows)}**",
            f"Real-only rows: **{len(real_only_rows)}**",
            "",
            "## Ranking stability by reference dataset",
            "",
        ]
        for ref_name, score in ranking_results["stability"].items():
            lines.append(f"- `{ref_name}`: {score:.4f}")

        lines.extend(["", "## Expected-order agreement", ""])
        if expected_eval.get("available"):
            details = expected_eval.get("details", [])
            if not details:
                lines.append("No comparable expected-order entries were found.")
            else:
                for row in details[:20]:
                    lines.append(
                        "- "
                        f"variant `{row['variant']}`, ref `{row['reference']}` -> "
                        f"spearman={row['spearman']}, kendall={row['kendall']}"
                    )
        else:
            lines.append("Not configured. Add `paper.expected_orders` to config for this section.")

        lines.extend(["", "## Special genre-pair separability", ""])
        if special_metrics.get("available"):
            for row in special_metrics.get("summary", [])[:10]:
                lines.append(
                    "- "
                    f"`{row['pair']}` -> mean FMD={row['mean_fmd']:.4f}, "
                    f"std={row['std_fmd']:.4f}, ratio={row['distinguishability_ratio']:.3f}"
                )
        else:
            lines.append("Not configured. Add `paper.special_pairs` and optional `paper.genre_aliases`.")

        lines.extend(["", "## Top separating variants per special pair", ""])
        if special_metrics.get("available") and special_metrics.get("top_variants"):
            for row in special_metrics.get("top_variants", [])[:30]:
                lines.append(
                    "- "
                    f"`{row['pair']}` rank {row['rank']}: `{row['variant']}` (FMD={row['fmd']:.4f})"
                )
        else:
            lines.append("No top-variant entries available.")

        lines.extend(["", "## Variant effects (delta FMD)", ""])
        tok_rows = variant_effects.get("all", {}).get("tokenizer_deltas", [])
        mod_rows = variant_effects.get("all", {}).get("model_deltas", [])
        tok_real_rows = variant_effects.get("real_only", {}).get("tokenizer_deltas", [])
        mod_real_rows = variant_effects.get("real_only", {}).get("model_deltas", [])
        lines.append(f"Tokenizer deltas rows (all): **{len(tok_rows)}**")
        lines.append(f"Model deltas rows (all): **{len(mod_rows)}**")
        lines.append(f"Tokenizer deltas rows (real-only): **{len(tok_real_rows)}**")
        lines.append(f"Model deltas rows (real-only): **{len(mod_real_rows)}**")
        for row in tok_rows[:8]:
            lines.append(
                "- "
                f"model `{row['model']}` ({row['remove_velocity']}/{row['hard_quantization']}): "
                f"`{row['tokenizer_a']}` - `{row['tokenizer_b']}` = {row['delta_fmd']:.4f}"
            )
        for row in mod_rows[:8]:
            lines.append(
                "- "
                f"tokenizer `{row['tokenizer']}` ({row['remove_velocity']}/{row['hard_quantization']}): "
                f"`{row['model_a']}` - `{row['model_b']}` = {row['delta_fmd']:.4f}"
            )

        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "csv_all": str(csv_all_path),
            "csv_real_only": str(csv_real_only_path),
            "markdown": str(md_path),
            "special_csv": str(special_csv_path),
            "special_summary_csv": str(special_summary_csv_path),
            "special_top_variants_csv": str(special_top_csv_path),
            "variant_delta_tokenizer_csv": str(tokenizer_delta_csv_path),
            "variant_delta_model_csv": str(model_delta_csv_path),
            "variant_delta_tokenizer_real_only_csv": str(self.output_dir / "variant_delta_tokenizer_real_only.csv"),
            "variant_delta_model_real_only_csv": str(self.output_dir / "variant_delta_model_real_only.csv"),
        }

    def run_full(self) -> Dict:
        """Run full benchmark suited for research reporting."""
        paper_cfg = self.config.get("paper", {})
        tokenizers = paper_cfg.get("tokenizers") or None
        models = paper_cfg.get("models") or None

        variants = self.build_variants(tokenizers=tokenizers, models=models)
        pairwise_rows = self.run_pairwise_benchmark(variants)
        ranking_results = self.run_ranking_benchmark(variants)
        expected_eval = self.evaluate_expected_order(ranking_results)
        split_rows = self._split_pairwise_rows(pairwise_rows)
        special_metrics = self.compute_special_pair_metrics(split_rows["all"])
        variant_effects = {
            "all": self.compute_variant_effects(split_rows["all"]),
            "real_only": self.compute_variant_effects(split_rows["real_only"]),
        }
        files = self.save_outputs(pairwise_rows, ranking_results, expected_eval, special_metrics, variant_effects)

        logger.info(f"Paper benchmark completed. Outputs: {files}")
        return {
            "variants": [v.name for v in variants],
            "pairwise_rows": len(pairwise_rows),
            "stability": ranking_results["stability"],
            "outputs": files,
        }

    def run_quick(self) -> Dict:
        """Fast smoke benchmark for one-click run in IDE."""
        tokenizers = [self.config["tokenization"]["tokenizers"][0]["type"]]
        models = [self.config["embeddings"]["models"][0]["name"]]
        variants = self.build_variants(
            tokenizers=tokenizers,
            models=models,
            preprocessing_grid=[(False, False)],
        )
        pairwise_rows = self.run_pairwise_benchmark(variants)
        ranking_results = self.run_ranking_benchmark(variants)
        expected_eval = self.evaluate_expected_order(ranking_results)
        split_rows = self._split_pairwise_rows(pairwise_rows)
        special_metrics = self.compute_special_pair_metrics(split_rows["all"])
        variant_effects = {
            "all": self.compute_variant_effects(split_rows["all"]),
            "real_only": self.compute_variant_effects(split_rows["real_only"]),
        }
        files = self.save_outputs(pairwise_rows, ranking_results, expected_eval, special_metrics, variant_effects)
        return {
            "variants": [v.name for v in variants],
            "pairwise_rows": len(pairwise_rows),
            "outputs": files,
        }

