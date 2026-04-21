#!/usr/bin/env python3
"""Generative model evaluation — FMD/nFMD ranking stability across models.

Validates that nFMD gives more stable rankings than raw FMD when comparing
generative baselines (Markov, random) against real music.

Expected ranking: real << Markov << random
Key question: Is this ranking consistent across embedding models for nFMD
but not for raw FMD?
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.manager import DatasetManager
from data.generative_loader import MarkovMIDIGenerator, RandomMIDIGenerator
from embeddings.extractor import EmbeddingExtractor
from metrics.fmd import FrechetMusicDistance
from preprocessing.processor import MIDIPreprocessor
from tokenization.tokenizer import TokenizationPipeline

OUTPUT_DIR = Path("results/reports/generative_eval")
PLOTS_DIR = Path("results/plots/paper")
SEED = 42
N_GENERATED = 100


def load_config() -> Dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def extract_embeddings_from_files(
    midi_files: List[Path],
    tok_name: str,
    model_name: str,
    config: Dict,
    preprocessor: MIDIPreprocessor,
    tokenization: TokenizationPipeline,
    embeddings: EmbeddingExtractor,
) -> np.ndarray | None:
    """Extract embeddings for a list of MIDI files."""
    vectors = []
    for midi_path in midi_files:
        try:
            midi_data = preprocessor.load_midi(midi_path)
            if midi_data is None:
                continue
            midi_data = preprocessor.filter_note_range(midi_data)
            midi_data = preprocessor.normalize_instruments(midi_data)
            tokenizer = tokenization.tokenizers[tok_name]
            tokens = tokenizer.encode_midi_object(midi_data)
            if not tokens:
                continue
            vec = embeddings.extract_embeddings(
                [tokens], model_name, midi_data_list=[midi_data]
            )[0]
            vectors.append(vec)
        except Exception as exc:
            logger.debug(f"Skip {midi_path.name}: {exc}")
    return np.vstack(vectors) if vectors else None


def main():
    logger.info("=" * 70)
    logger.info("GENERATIVE MODEL EVALUATION — FMD/nFMD RANKING STABILITY")
    logger.info("=" * 70)

    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_manager = DatasetManager(config)
    preprocessor = MIDIPreprocessor(config)
    tokenization = TokenizationPipeline(config)
    emb_extractor = EmbeddingExtractor(config)
    fmd_calc = FrechetMusicDistance(config)

    tokenizers = [t["type"] for t in config["tokenization"]["tokenizers"]]
    models = [m["name"] for m in config["embeddings"]["models"]]

    # ── Step 1: Get real MIDI files (use rock from Lakh) ──────────────
    logger.info("\n=== Step 1: Loading real MIDI files ===")
    real_files = dataset_manager.list_midi_files("lakh_rock", processed=False, limit=120)
    if not real_files:
        logger.error("No real MIDI files found. Run multi-genre analysis first.")
        return
    logger.info(f"Real files: {len(real_files)}")

    # ── Step 2: Generate baseline MIDI files ──────────────────────────
    logger.info("\n=== Step 2: Generating baselines ===")
    gen_dir = Path("data/raw/generated")

    # Markov baseline
    markov_dir = gen_dir / "markov"
    markov_gen = MarkovMIDIGenerator(seed=SEED)
    markov_gen.fit(real_files, max_files=120)
    markov_files = markov_gen.generate(n_files=N_GENERATED, output_dir=markov_dir)
    logger.info(f"Markov files: {len(markov_files)}")

    # Random baseline
    random_dir = gen_dir / "random"
    random_gen = RandomMIDIGenerator(seed=SEED)
    random_files = random_gen.generate(n_files=N_GENERATED, output_dir=random_dir)
    logger.info(f"Random files: {len(random_files)}")

    # ── Step 3: Extract embeddings & compute FMD/nFMD ─────────────────
    logger.info("\n=== Step 3: Computing FMD/nFMD across all variants ===")
    rows: List[Dict] = []
    sources = {
        "real_split": None,  # will use split-half of real
        "markov": markov_files,
        "random": random_files,
    }

    total = len(tokenizers) * len(models)
    idx = 0

    for tok, model in product(tokenizers, models):
        idx += 1
        logger.info(f"[{idx}/{total}] {tok} × {model}")

        # Extract real embeddings
        real_emb = extract_embeddings_from_files(
            real_files, tok, model, config,
            preprocessor, tokenization, emb_extractor,
        )
        if real_emb is None or real_emb.shape[0] < 10:
            logger.warning(f"  Skip: insufficient real embeddings")
            continue

        # Split-half FMD (within-real baseline)
        n = real_emb.shape[0]
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(n)
        real_a, real_b = real_emb[perm[:n // 2]], real_emb[perm[n // 2:]]

        for source_name, source_files in [("split_half", None), ("markov", markov_files), ("random", random_files)]:
            if source_name == "split_half":
                test_emb = real_b
                ref_emb = real_a
            else:
                test_emb = extract_embeddings_from_files(
                    source_files, tok, model, config,
                    preprocessor, tokenization, emb_extractor,
                )
                ref_emb = real_emb
                if test_emb is None or test_emb.shape[0] < 5:
                    continue

            try:
                nfmd_result = fmd_calc.compute_nfmd(ref_emb, test_emb)
                rows.append({
                    "tokenizer": tok,
                    "model": model,
                    "source": source_name,
                    "fmd": nfmd_result["fmd"],
                    "nfmd_trace": nfmd_result["nfmd_trace"],
                    "nfmd_norm": nfmd_result["nfmd_norm"],
                    "n_ref": ref_emb.shape[0],
                    "n_test": test_emb.shape[0],
                })
            except Exception as e:
                logger.warning(f"  FMD failed for {source_name}: {e}")

    logger.info(f"\nTotal rows: {len(rows)}")

    # ── Step 4: Save & analyze ────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "generative_eval_fmd.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    # Ranking stability: for each model, check if ranking is
    # split_half < markov < random
    logger.info("\n=== Step 4: Ranking stability analysis ===")
    ranking_results = []

    for metric in ["fmd", "nfmd_trace", "nfmd_norm"]:
        for model in models:
            model_df = df[df["model"] == model]
            means = model_df.groupby("source")[metric].mean()

            if all(s in means.index for s in ["split_half", "markov", "random"]):
                correct_order = (means["split_half"] < means["markov"] < means["random"])
                ranking_results.append({
                    "model": model,
                    "metric": metric,
                    "fmd_split_half": means["split_half"],
                    "fmd_markov": means["markov"],
                    "fmd_random": means["random"],
                    "correct_ranking": correct_order,
                })
                sig = "✓" if correct_order else "✗"
                logger.info(f"  {metric:12s} | {model:18s}: "
                           f"split={means['split_half']:.4f} < "
                           f"markov={means['markov']:.4f} < "
                           f"random={means['random']:.4f}  {sig}")

    rank_df = pd.DataFrame(ranking_results)
    rank_df.to_csv(OUTPUT_DIR / "ranking_stability.csv", index=False)

    # Summary: percentage of models with correct ranking per metric
    logger.info("\n  Ranking correctness summary:")
    for metric in ["fmd", "nfmd_trace", "nfmd_norm"]:
        subset = rank_df[rank_df["metric"] == metric]
        if len(subset) > 0:
            pct = 100 * subset["correct_ranking"].mean()
            logger.info(f"    {metric:12s}: {pct:.0f}% models have correct ranking")

    # ── Step 5: Plots ─────────────────────────────────────────────────
    logger.info("\n=== Step 5: Generating plots ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    def save_fig(fig, name):
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  ✅ {name}")

    # Plot 1: FMD by source × model
    if not df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, metric in zip(axes, ["fmd", "nfmd_trace", "nfmd_norm"]):
            sns.barplot(data=df, x="model", y=metric, hue="source",
                       ax=ax, palette="Set2", order=models)
            ax.set_title(f"{metric}")
            ax.tick_params(axis="x", rotation=30)
            ax.set_ylabel(metric)
        fig.suptitle("Generative Evaluation: Real vs Markov vs Random", fontsize=14, y=1.02)
        save_fig(fig, "generative_eval_barplot")

    # ── Step 6: Report ────────────────────────────────────────────────
    report = []
    report.append("# Generative Model Evaluation Report")
    report.append(f"\n**Validates FMD/nFMD ranking: real < Markov < random**")
    report.append(f"\n## Design")
    report.append(f"- Real: {len(real_files)} Lakh rock MIDI files")
    report.append(f"- Markov: {N_GENERATED} generated (first-order chain)")
    report.append(f"- Random: {N_GENERATED} completely random")
    report.append(f"- Variants: {len(tokenizers)} tokenizers × {len(models)} models")
    report.append(f"- Total observations: {len(rows)}")

    if not rank_df.empty:
        report.append(f"\n## Ranking Correctness (real < Markov < random)")
        report.append(f"\n| Metric | % Models Correct |")
        report.append(f"|--------|-----------------|")
        for metric in ["fmd", "nfmd_trace", "nfmd_norm"]:
            subset = rank_df[rank_df["metric"] == metric]
            pct = 100 * subset["correct_ranking"].mean() if len(subset) > 0 else 0
            report.append(f"| {metric} | {pct:.0f}% |")

    report_path = OUTPUT_DIR / "GENERATIVE_EVAL_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    logger.info(f"✅ Report: {report_path}")

    logger.info("\n" + "=" * 70)
    logger.info("GENERATIVE EVALUATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

