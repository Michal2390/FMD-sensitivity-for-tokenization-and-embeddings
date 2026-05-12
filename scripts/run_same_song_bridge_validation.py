#!/usr/bin/env python3
"""Run same-song cross-tokenizer bridge validation on real Lakh data.

This script focuses specifically on diagnostics 8A / 8B / 8C:
- 8A: per-model PCA / t-SNE panels for the same songs across tokenizers,
- 8B: cross-tokenizer nearest-neighbor accuracy,
- 8C: same-song vs different-song cosine gap.

It uses real Lakh MIDI files but avoids the full 96-variant FMD benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.embedding_diagnostics import run_embedding_diagnostics
from experiments.lakh_plots import generate_lakh_plots
from experiments.paper_pipeline import PaperExperimentRunner, PipelineVariant
from utils.config import load_config, setup_logging, get_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-song cross-tokenizer bridge validation on Lakh")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--genre-a", default="rock")
    parser.add_argument("--genre-b", default="jazz")
    parser.add_argument("--max-files-per-genre", type=int, default=12)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "reports" / "lakh"))
    parser.add_argument("--plots-dir", default=str(ROOT / "results" / "plots" / "paper"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tokenizers", nargs="*", default=None)
    return parser.parse_args()


def _prepare_sequences(
    runner: PaperExperimentRunner,
    dataset_name: str,
    tokenizer_name: str,
    variant: PipelineVariant,
    max_files: int,
    logger,
) -> Dict[str, List]:
    midi_files = runner._list_dataset_midis(dataset_name)[:max_files]
    tokenizer = runner.tokenization.tokenizers[tokenizer_name]
    records = {"tokens": [], "midi_data": [], "sample_ids": []}
    for midi_path in midi_files:
        try:
            midi_data = runner._preprocess_midi_file(midi_path, variant)
            if midi_data is None:
                continue
            tokens = tokenizer.encode_midi_object(midi_data)
            if not tokens:
                continue
            records["tokens"].append(tokens)
            records["midi_data"].append(midi_data)
            records["sample_ids"].append(midi_path.name)
        except Exception as exc:  # pragma: no cover - defensive for messy MIDI files
            logger.warning(f"Skip {midi_path.name} ({tokenizer_name}): {exc}")
    return records


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    setup_logging(
        config["logging"].get("level", "INFO"),
        config["logging"].get("log_file", "logs/experiment.log"),
    )
    logger = get_logger(__name__)

    config_for_runner = copy.deepcopy(config)
    config_for_runner.setdefault("paper", {})["max_files_per_dataset"] = int(args.max_files_per_genre)
    runner = PaperExperimentRunner(config_for_runner)

    tokenizers = args.tokenizers or [t["type"] for t in config["tokenization"]["tokenizers"]]
    models = args.models or [m["name"] for m in config["embeddings"]["models"]]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Running same-song bridge validation on real Lakh data: genres=({args.genre_a}, {args.genre_b}), "
        f"files/genre={args.max_files_per_genre}, tokenizers={tokenizers}, models={models}"
    )

    base_variant = PipelineVariant(tokenizer=tokenizers[0], model=models[0], remove_velocity=False, hard_quantization=False)
    datasets = [(f"lakh_{args.genre_a}", args.genre_a), (f"lakh_{args.genre_b}", args.genre_b)]

    prepared_by_tokenizer: Dict[str, Dict[str, Dict[str, List]]] = {}
    for tokenizer_name in tokenizers:
        prepared_by_tokenizer[tokenizer_name] = {}
        variant = PipelineVariant(tokenizer=tokenizer_name, model=models[0], remove_velocity=False, hard_quantization=False)
        for dataset_name, genre_label in datasets:
            prepared_by_tokenizer[tokenizer_name][genre_label] = _prepare_sequences(
                runner=runner,
                dataset_name=dataset_name,
                tokenizer_name=tokenizer_name,
                variant=variant,
                max_files=int(args.max_files_per_genre),
                logger=logger,
            )
            logger.info(
                f"Prepared {len(prepared_by_tokenizer[tokenizer_name][genre_label]['tokens'])} files for "
                f"genre={genre_label}, tokenizer={tokenizer_name}"
            )

    embeddings_by_variant: Dict[str, Dict[str, np.ndarray]] = {}
    token_sequences_by_variant: Dict[str, Dict[str, List[List[int]]]] = {}
    sample_ids_by_variant: Dict[str, Dict[str, List[str]]] = {}

    for model_name in models:
        for tokenizer_name in tokenizers:
            variant = PipelineVariant(tokenizer=tokenizer_name, model=model_name, remove_velocity=False, hard_quantization=False)
            variant_embs: Dict[str, np.ndarray] = {}
            variant_tokens: Dict[str, List[List[int]]] = {}
            variant_ids: Dict[str, List[str]] = {}
            for _, genre_label in datasets:
                prepared = prepared_by_tokenizer[tokenizer_name][genre_label]
                if not prepared["tokens"]:
                    continue
                embeddings = runner.embeddings.extract_embeddings(
                    prepared["tokens"],
                    model_name,
                    midi_data_list=prepared["midi_data"],
                )
                variant_embs[genre_label] = embeddings
                variant_tokens[genre_label] = prepared["tokens"]
                variant_ids[genre_label] = prepared["sample_ids"]
                logger.info(
                    f"Embeddings ready: model={model_name}, tokenizer={tokenizer_name}, genre={genre_label}, n={len(prepared['tokens'])}"
                )
            embeddings_by_variant[variant.name] = variant_embs
            token_sequences_by_variant[variant.name] = variant_tokens
            sample_ids_by_variant[variant.name] = variant_ids

    diag_outputs = run_embedding_diagnostics(
        embeddings_by_variant=embeddings_by_variant,
        token_sequences_by_variant=token_sequences_by_variant,
        sample_ids_by_variant=sample_ids_by_variant,
        genre_a=args.genre_a,
        genre_b=args.genre_b,
        output_dir=output_dir,
        seed=int(config.get("paper", {}).get("seed", 42)),
    )

    plot_config = copy.deepcopy(config)
    plot_config.setdefault("lakh", {})["output_dir"] = str(output_dir)
    plot_config.setdefault("lakh", {})["plots_dir"] = str(args.plots_dir)
    plot_outputs = generate_lakh_plots(plot_config)

    summary = {
        "genres": [args.genre_a, args.genre_b],
        "max_files_per_genre": int(args.max_files_per_genre),
        "models": models,
        "tokenizers": tokenizers,
        "diagnostics_outputs": diag_outputs,
        "plot_outputs": plot_outputs,
    }
    summary_path = output_dir / "same_song_bridge_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved same-song bridge summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


