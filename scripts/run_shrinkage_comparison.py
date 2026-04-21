#!/usr/bin/env python3
"""Covariance estimator comparison for FMD.

Compares empirical vs Ledoit-Wolf vs Basic Shrinkage vs OAS covariance
estimation on the existing multi-genre FMD dataset.

Key question: Does shrinkage covariance change the sensitivity pattern
(η² decomposition)? Does it reduce FMD overestimation at small sample sizes?
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.manager import DatasetManager
from embeddings.extractor import EmbeddingExtractor
from metrics.fmd import FrechetMusicDistance, COVARIANCE_METHODS
from preprocessing.processor import MIDIPreprocessor
from tokenization.tokenizer import TokenizationPipeline

OUTPUT_DIR = Path("results/reports/shrinkage_comparison")
PLOTS_DIR = Path("results/plots/paper")
SUBSAMPLE_SIZE = 100
N_REPEATS = 5
SEED = 42


def load_config() -> Dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def compute_eta_squared(df: pd.DataFrame, factor: str, response: str = "fmd") -> float:
    groups = [g[response].values for _, g in df.groupby(factor)]
    if len(groups) < 2:
        return 0.0
    ss_between = sum(len(g) * (np.mean(g) - df[response].mean()) ** 2 for g in groups)
    ss_total = ((df[response] - df[response].mean()) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def main():
    logger.info("=" * 70)
    logger.info("COVARIANCE ESTIMATOR COMPARISON FOR FMD")
    logger.info("=" * 70)

    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_manager = DatasetManager(config)
    preprocessor = MIDIPreprocessor(config)
    tokenization = TokenizationPipeline(config)
    emb_extractor = EmbeddingExtractor(config)

    genres = config["lakh"]["genres"]
    genre_pairs = list(combinations(genres, 2))

    # Use subset of variants for efficiency: 2 tokenizers × 2 models × 2 preprocess
    tokenizers = ["REMI", "Octuple"]
    models = ["CLaMP-2", "MusicBERT"]
    preprocess_configs = [(False, False), (True, True)]
    cov_methods = ["empirical", "ledoit_wolf", "basic_shrinkage", "oas"]

    # ── Step 1: Extract embeddings (reuse cache) ──────────────────────
    logger.info("\n=== Step 1: Extracting embeddings ===")
    emb_cache = {}

    for genre in genres:
        for tok, model in product(tokenizers, models):
            for vel, quant in preprocess_configs:
                key = (genre, tok, model, vel, quant)
                ds_name = f"lakh_{genre}"
                midi_files = dataset_manager.list_midi_files(ds_name, processed=False, limit=120)
                if not midi_files:
                    continue

                vectors = []
                for midi_path in midi_files:
                    try:
                        midi_data = preprocessor.load_midi(midi_path)
                        if midi_data is None:
                            continue
                        if vel:
                            midi_data = preprocessor.remove_velocity(midi_data)
                        if quant:
                            midi_data = preprocessor.quantize_time(midi_data)
                        midi_data = preprocessor.filter_note_range(midi_data)
                        midi_data = preprocessor.normalize_instruments(midi_data)
                        tokenizer = tokenization.tokenizers[tok]
                        tokens = tokenizer.encode_midi_object(midi_data)
                        if not tokens:
                            continue
                        vec = emb_extractor.extract_embeddings(
                            [tokens], model, midi_data_list=[midi_data]
                        )[0]
                        vectors.append(vec)
                    except Exception:
                        continue

                if vectors:
                    emb_cache[key] = np.vstack(vectors)
                    logger.info(f"  {key}: {len(vectors)} embeddings")

    # ── Step 2: Compute FMD with different covariance methods ─────────
    logger.info("\n=== Step 2: Computing FMD with different covariance methods ===")
    rows = []
    rng = np.random.default_rng(SEED)

    for cov_method in cov_methods:
        # Create FMD calculator with this method
        cov_config = dict(config)
        cov_config["fmd_metric"] = dict(config["fmd_metric"])
        cov_config["fmd_metric"]["covariance_method"] = cov_method
        fmd_calc = FrechetMusicDistance(cov_config)

        for tok, model in product(tokenizers, models):
            for vel, quant in preprocess_configs:
                for genre_a, genre_b in genre_pairs:
                    key_a = (genre_a, tok, model, vel, quant)
                    key_b = (genre_b, tok, model, vel, quant)
                    emb_a = emb_cache.get(key_a)
                    emb_b = emb_cache.get(key_b)

                    if emb_a is None or emb_b is None:
                        continue

                    n_a, n_b = emb_a.shape[0], emb_b.shape[0]
                    sub = min(SUBSAMPLE_SIZE, n_a, n_b)
                    if sub < 10:
                        continue

                    for rep in range(N_REPEATS):
                        idx_a = rng.choice(n_a, size=sub, replace=False)
                        idx_b = rng.choice(n_b, size=sub, replace=False)

                        try:
                            result = fmd_calc.compute_nfmd(emb_a[idx_a], emb_b[idx_b])
                            rows.append({
                                "cov_method": cov_method,
                                "tokenizer": tok,
                                "model": model,
                                "remove_velocity": vel,
                                "hard_quantization": quant,
                                "pair": f"{genre_a}_vs_{genre_b}",
                                "repeat": rep,
                                "fmd": result["fmd"],
                                "nfmd_trace": result["nfmd_trace"],
                            })
                        except Exception:
                            continue

    logger.info(f"\nTotal rows: {len(rows)}")

    # ── Step 3: Analysis ──────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "shrinkage_comparison.csv", index=False)

    logger.info("\n=== Step 3: η² comparison by covariance method ===")
    df["preprocess"] = df["remove_velocity"].astype(str) + "_" + df["hard_quantization"].astype(str)

    comparison = []
    for cov_method in cov_methods:
        subset = df[df["cov_method"] == cov_method]
        if subset.empty:
            continue
        for metric in ["fmd", "nfmd_trace"]:
            for factor in ["tokenizer", "model", "preprocess"]:
                eta = compute_eta_squared(subset, factor, metric)
                comparison.append({
                    "cov_method": cov_method,
                    "metric": metric,
                    "factor": factor,
                    "eta_sq": eta,
                    "mean_fmd": subset[metric].mean(),
                    "std_fmd": subset[metric].std(),
                    "n": len(subset),
                })
                logger.info(f"  {cov_method:16s} | {metric:12s} × {factor:12s}: η²={eta:.4f}")

    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(OUTPUT_DIR / "shrinkage_eta_sq.csv", index=False)

    # ── Step 4: Plots ─────────────────────────────────────────────────
    logger.info("\n=== Step 4: Generating plots ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    if not comp_df.empty:
        # η² comparison across covariance methods
        for metric in ["fmd", "nfmd_trace"]:
            subset = comp_df[comp_df["metric"] == metric]
            if subset.empty:
                continue
            pivot = subset.pivot(index="factor", columns="cov_method", values="eta_sq")
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot.plot(kind="bar", ax=ax, colormap="Set2")
            ax.set_ylabel("η²")
            ax.set_title(f"Covariance Estimator Impact on η² ({metric})")
            ax.tick_params(axis="x", rotation=0)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"shrinkage_eta_sq_{metric}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  ✅ shrinkage_eta_sq_{metric}")

    # ── Step 5: Report ────────────────────────────────────────────────
    report = []
    report.append("# Covariance Estimator Comparison Report")
    report.append(f"\n**Impact of shrinkage estimators on FMD sensitivity patterns.**")
    report.append(f"\nMethods compared: {', '.join(cov_methods)}")
    report.append(f"\nTotal observations: {len(rows)}")

    if not comp_df.empty:
        report.append(f"\n## η² by Covariance Method (raw FMD)")
        report.append(f"\n| Method | η²(tokenizer) | η²(model) | η²(preprocess) |")
        report.append(f"|--------|---------------|-----------|----------------|")
        for method in cov_methods:
            sub = comp_df[(comp_df["cov_method"] == method) & (comp_df["metric"] == "fmd")]
            tok = sub[sub["factor"] == "tokenizer"]["eta_sq"].values
            mod = sub[sub["factor"] == "model"]["eta_sq"].values
            pre = sub[sub["factor"] == "preprocess"]["eta_sq"].values
            report.append(f"| {method} | {tok[0]:.4f if len(tok) else 'N/A'} | "
                         f"{mod[0]:.4f if len(mod) else 'N/A'} | "
                         f"{pre[0]:.4f if len(pre) else 'N/A'} |")

    report_path = OUTPUT_DIR / "SHRINKAGE_COMPARISON_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    logger.info(f"✅ Report: {report_path}")

    logger.info("\n" + "=" * 70)
    logger.info("SHRINKAGE COMPARISON COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

