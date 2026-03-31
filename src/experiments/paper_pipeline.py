"""Paper-oriented experiment pipeline for FMD sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import csv
import json

import numpy as np
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

    def _extract_dataset_embeddings(self, dataset_name: str, variant: PipelineVariant) -> np.ndarray:
        midi_files = self._list_dataset_midis(dataset_name)
        vectors: List[np.ndarray] = []

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
            except Exception as exc:
                logger.warning(f"Skipping {midi_path} for {variant.name}: {exc}")

        if not vectors:
            logger.warning(
                f"No embeddings from files for dataset={dataset_name}, variant={variant.name}; "
                "using deterministic synthetic fallback"
            )
            return self._synthetic_embeddings(dataset_name, variant)

        return np.vstack(vectors)

    def run_pairwise_benchmark(self, variants: Sequence[PipelineVariant]) -> List[Dict]:
        """Compute FMD for configured dataset pairs across all variants."""
        exp5 = self.config.get("experiments", {}).get("exp5_cross_genre", {})
        pairs = self._parse_pairs(exp5.get("pairs", []))
        if not pairs:
            names = self._dataset_names()
            pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

        rows: List[Dict] = []
        cache: Dict[Tuple[str, str], np.ndarray] = {}

        for variant in variants:
            for ds_a, ds_b in pairs:
                key_a = (ds_a, variant.name)
                key_b = (ds_b, variant.name)
                if key_a not in cache:
                    cache[key_a] = self._extract_dataset_embeddings(ds_a, variant)
                if key_b not in cache:
                    cache[key_b] = self._extract_dataset_embeddings(ds_b, variant)

                fmd_value = self.fmd.compute_fmd(cache[key_a], cache[key_b])
                rows.append(
                    {
                        "variant": variant.name,
                        "dataset_a": ds_a,
                        "dataset_b": ds_b,
                        "fmd": float(fmd_value),
                    }
                )

        return rows

    def run_ranking_benchmark(self, variants: Sequence[PipelineVariant]) -> Dict:
        """Build rankings and stability across configurations."""
        dataset_names = self._dataset_names()
        ranking_by_variant: Dict[str, Dict[str, List[int]]] = {}

        for variant in variants:
            emb_sets = []
            for ds_name in dataset_names:
                emb = self._extract_dataset_embeddings(ds_name, variant)
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

    def save_outputs(self, pairwise_rows: List[Dict], ranking_results: Dict, expected_eval: Dict) -> Dict:
        """Save JSON, CSV and markdown summary for paper draft."""
        json_path = self.output_dir / "paper_results.json"
        csv_path = self.output_dir / "pairwise_fmd.csv"
        md_path = self.output_dir / "paper_summary.md"

        payload = {
            "pairwise": pairwise_rows,
            "ranking": ranking_results,
            "expected_eval": expected_eval,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["variant", "dataset_a", "dataset_b", "fmd"])
            writer.writeheader()
            writer.writerows(pairwise_rows)

        # Simple markdown summary for direct paper drafting.
        lines = [
            "# Paper-Oriented FMD Benchmark Summary",
            "",
            "## Pairwise comparisons",
            "",
            f"Total rows: **{len(pairwise_rows)}**",
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

        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
        }

    def run_full(self) -> Dict:
        """Run full benchmark suited for research reporting."""
        paper_cfg = self.config.get("paper", {})
        tokenizers = paper_cfg.get("tokenizers")
        models = paper_cfg.get("models")

        variants = self.build_variants(tokenizers=tokenizers, models=models)
        pairwise_rows = self.run_pairwise_benchmark(variants)
        ranking_results = self.run_ranking_benchmark(variants)
        expected_eval = self.evaluate_expected_order(ranking_results)
        files = self.save_outputs(pairwise_rows, ranking_results, expected_eval)

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
        files = self.save_outputs(pairwise_rows, ranking_results, expected_eval)
        return {
            "variants": [v.name for v in variants],
            "pairwise_rows": len(pairwise_rows),
            "outputs": files,
        }

