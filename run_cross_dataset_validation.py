#!/usr/bin/env python3
"""Cross-dataset validation for FMD sensitivity analysis.

Validates generalizability of FMD sensitivity findings by repeating
the multi-genre analysis on independent data sources:

  Source A: Lakh + Tagtraum CD2  (original — loads existing CSV)
  Source B: Lakh + Tagtraum CD1  (cross-annotation validation)
  Source C: MidiCaps             (cross-dataset validation)

For each source, computes 32 variants × genre pairs × 10 repeats of FMD,
then compares η², pipeline rankings, and per-cell (tokenizer×model) means
across sources to assess reproducibility.

Usage:
    python run_cross_dataset_validation.py                  # all sources
    python run_cross_dataset_validation.py --source cd1     # CD1 only
    python run_cross_dataset_validation.py --source midicaps
    python run_cross_dataset_validation.py --source all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.manager import DatasetManager
from data.lakh_genre_loader import LakhGenreLoader
from data.midicaps_loader import MidiCapsGenreLoader
from embeddings.extractor import EmbeddingExtractor
from metrics.fmd import FrechetMusicDistance
from preprocessing.processor import MIDIPreprocessor
from tokenization.tokenizer import TokenizationPipeline

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
SEED = 42


def load_config() -> Dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def build_variants(config: Dict):
    """Build 32 variant tuples: (tokenizer, model, remove_vel, hard_quant)."""
    tokenizers = [t["type"] for t in config["tokenization"]["tokenizers"]]
    models = [m["name"] for m in config["embeddings"]["models"]]
    preprocess = [(False, False), (True, False), (False, True), (True, True)]
    return list(product(tokenizers, models, preprocess))


def variant_name(tok, model, vel, quant):
    return (
        f"tok={tok}|model={model}|"
        f"vel={'off' if vel else 'on'}|"
        f"quant={'on' if quant else 'off'}"
    )


# ──────────────────────────────────────────────────────────────────────
# Embedding extraction (generic — works for any dataset prefix)
# ──────────────────────────────────────────────────────────────────────

def extract_genre_embeddings(
    genre: str,
    tok_name: str,
    model_name: str,
    remove_vel: bool,
    hard_quant: bool,
    config: Dict,
    dataset_manager: DatasetManager,
    preprocessor: MIDIPreprocessor,
    tokenization: TokenizationPipeline,
    embeddings: EmbeddingExtractor,
    dataset_prefix: str = "lakh",
    max_files: int = 120,
) -> np.ndarray | None:
    """Extract embeddings for one genre + one variant. Returns (N, D) or None."""
    ds_name = f"{dataset_prefix}_{genre}"
    midi_files = dataset_manager.list_midi_files(ds_name, processed=False, limit=max_files)
    if not midi_files:
        logger.warning(f"No MIDI files found for {ds_name}")
        return None

    vectors = []
    for midi_path in midi_files:
        try:
            midi_data = preprocessor.load_midi(midi_path)
            if midi_data is None:
                continue
            if remove_vel:
                midi_data = preprocessor.remove_velocity(midi_data)
            if hard_quant:
                midi_data = preprocessor.quantize_time(midi_data)
            midi_data = preprocessor.filter_note_range(midi_data)
            midi_data = preprocessor.normalize_instruments(midi_data)

            tokenizer = tokenization.tokenizers[tok_name]
            tokens = tokenizer.encode_midi_object(midi_data)
            if not tokens:
                continue
            vec = embeddings.extract_embeddings([tokens], model_name)[0]
            vectors.append(vec)
        except Exception as exc:
            logger.debug(f"Skip {midi_path.name}: {exc}")

    return np.vstack(vectors) if vectors else None


# ──────────────────────────────────────────────────────────────────────
# FMD computation with repeated subsampling (reusable)
# ──────────────────────────────────────────────────────────────────────

def compute_fmd_repeated(
    emb_cache: Dict[Tuple, np.ndarray],
    genres: List[str],
    variants: List,
    fmd_calc: FrechetMusicDistance,
    source_name: str,
    subsample_size: int = 100,
    n_repeats: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """Compute FMD with repeated subsampling for all variant × pair combos."""
    genre_pairs = list(combinations(genres, 2))
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    total_ops = len(variants) * len(genre_pairs)
    op_idx = 0

    for tok, model, (vel, quant) in variants:
        for genre_a, genre_b in genre_pairs:
            op_idx += 1
            vname = variant_name(tok, model, vel, quant)

            key_a = (genre_a, tok, model, vel, quant)
            key_b = (genre_b, tok, model, vel, quant)

            emb_a = emb_cache.get(key_a)
            emb_b = emb_cache.get(key_b)

            if emb_a is None or emb_b is None:
                logger.warning(f"[{op_idx}/{total_ops}] Skip {vname} | {genre_a}-{genre_b}: missing embs")
                continue

            n_a, n_b = emb_a.shape[0], emb_b.shape[0]
            sub_a = min(subsample_size, n_a)
            sub_b = min(subsample_size, n_b)

            if sub_a < 5 or sub_b < 5:
                continue

            full_fmd = float(fmd_calc.compute_fmd(emb_a, emb_b))

            for rep in range(n_repeats):
                idx_a = rng.choice(n_a, size=sub_a, replace=False)
                idx_b = rng.choice(n_b, size=sub_b, replace=False)
                fmd_val = float(fmd_calc.compute_fmd(emb_a[idx_a], emb_b[idx_b]))

                rows.append({
                    "source": source_name,
                    "variant": vname,
                    "tokenizer": tok,
                    "model": model,
                    "remove_velocity": vel,
                    "hard_quantization": quant,
                    "genre_a": genre_a,
                    "genre_b": genre_b,
                    "pair": f"{genre_a}_vs_{genre_b}",
                    "repeat": rep,
                    "subsample_size": sub_a,
                    "fmd": fmd_val,
                    "full_fmd": full_fmd,
                    "n_a": n_a,
                    "n_b": n_b,
                })

            if op_idx % 20 == 0 or op_idx == total_ops:
                logger.info(
                    f"  [{source_name}][{op_idx}/{total_ops}] {vname} | "
                    f"{genre_a}-{genre_b}: full={full_fmd:.4f}"
                )

    return rows


# ──────────────────────────────────────────────────────────────────────
# Statistical analysis helpers
# ──────────────────────────────────────────────────────────────────────

def compute_eta_squared(df: pd.DataFrame, factors: List[str]) -> Dict[str, float]:
    """Compute one-way η² for each factor."""
    result = {}
    for factor in factors:
        if factor not in df.columns:
            continue
        groups = [g["fmd"].values for _, g in df.groupby(factor)]
        if len(groups) < 2:
            continue
        ss_between = sum(len(g) * (np.mean(g) - df["fmd"].mean()) ** 2 for g in groups)
        ss_total = ((df["fmd"] - df["fmd"].mean()) ** 2).sum()
        result[factor] = float(ss_between / ss_total) if ss_total > 0 else 0.0
    return result


def compute_cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean FMD per tokenizer × model cell."""
    return (
        df.groupby(["tokenizer", "model"])
        .agg(mean_fmd=("fmd", "mean"), std_fmd=("fmd", "std"), n=("fmd", "count"))
        .reset_index()
        .sort_values("mean_fmd")
    )


def compute_pipeline_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank variants by mean FMD (descending = more separability)."""
    return (
        df.groupby("variant")
        .agg(mean_fmd=("fmd", "mean"))
        .reset_index()
        .sort_values("mean_fmd", ascending=False)
        .reset_index(drop=True)
    )


def compare_rankings(rank_a: pd.DataFrame, rank_b: pd.DataFrame) -> Dict:
    """Spearman ρ between variant rankings from two sources."""
    merged = rank_a.merge(rank_b, on="variant", suffixes=("_a", "_b"))
    if len(merged) < 3:
        return {"spearman_rho": None, "p_value": None, "n_common": len(merged)}
    rho, p = sp_stats.spearmanr(merged["mean_fmd_a"], merged["mean_fmd_b"])
    return {"spearman_rho": float(rho), "p_value": float(p), "n_common": len(merged)}


# ──────────────────────────────────────────────────────────────────────
# Source runners
# ──────────────────────────────────────────────────────────────────────

def load_existing_cd2(output_dir: Path) -> Optional[pd.DataFrame]:
    """Load existing Lakh CD2 results from multi_genre_fmd.csv."""
    csv_path = Path("results/reports/lakh_multi/multi_genre_fmd.csv")
    if not csv_path.exists():
        logger.warning(f"CD2 results not found at {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df["source"] = "lakh_cd2"
    logger.info(f"Loaded existing CD2 results: {len(df)} rows")
    return df


def run_source(
    source_name: str,
    dataset_prefix: str,
    genres: List[str],
    config: Dict,
    dataset_manager: DatasetManager,
    preprocessor: MIDIPreprocessor,
    tokenization: TokenizationPipeline,
    emb_extractor: EmbeddingExtractor,
    fmd_calc: FrechetMusicDistance,
    variants: List,
    subsample_size: int,
    n_repeats: int,
    max_files: int = 120,
) -> pd.DataFrame:
    """Run full FMD analysis for one source."""
    logger.info(f"\n{'='*70}")
    logger.info(f"SOURCE: {source_name} (prefix={dataset_prefix})")
    logger.info(f"{'='*70}")

    # Step 1: Extract embeddings
    emb_cache: Dict[Tuple, np.ndarray] = {}
    total = len(genres) * len(variants)
    idx = 0
    for genre in genres:
        for tok, model, (vel, quant) in variants:
            idx += 1
            key = (genre, tok, model, vel, quant)
            if key in emb_cache:
                continue
            vname = variant_name(tok, model, vel, quant)
            logger.info(f"  [{idx}/{total}] {genre} × {vname}")

            embs = extract_genre_embeddings(
                genre, tok, model, vel, quant,
                config, dataset_manager, preprocessor, tokenization, emb_extractor,
                dataset_prefix=dataset_prefix,
                max_files=max_files,
            )
            if embs is not None:
                emb_cache[key] = embs
                logger.info(f"    → {embs.shape[0]} embeddings, dim={embs.shape[1]}")
            else:
                logger.warning(f"    → NO embeddings!")

    # Step 2: Compute FMD
    rows = compute_fmd_repeated(
        emb_cache, genres, variants, fmd_calc,
        source_name=source_name,
        subsample_size=subsample_size,
        n_repeats=n_repeats,
        seed=SEED,
    )

    logger.info(f"  [{source_name}] Total FMD rows: {len(rows)}")
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def generate_cross_validation_plots(
    all_df: pd.DataFrame,
    source_eta: Dict[str, Dict[str, float]],
    source_rankings: Dict[str, pd.DataFrame],
    source_cells: Dict[str, pd.DataFrame],
    ranking_comparisons: Dict[str, Dict],
    plots_dir: Path,
):
    """Generate all cross-validation comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    DPI = 300
    plots_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(fig, name):
        fig.tight_layout()
        for fmt in ("png", "pdf"):
            fig.savefig(plots_dir / f"{name}.{fmt}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  ✅ {name}")

    sources = sorted(all_df["source"].unique())

    # ── Plot 1: η² comparison across sources ──────────────────────────
    if source_eta:
        factors = ["tokenizer", "model", "preprocess"]
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(factors))
        width = 0.8 / max(len(sources), 1)
        colors = sns.color_palette("Set2", len(sources))

        for i, src in enumerate(sources):
            eta = source_eta.get(src, {})
            vals = [eta.get(f, 0) for f in factors]
            bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                          label=src, color=colors[i])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(factors)
        ax.set_ylabel("η²")
        ax.set_title("η² Variance Decomposition: Cross-Dataset Comparison")
        ax.legend(title="Source")
        ax.axhline(0.14, color="red", linestyle="--", alpha=0.4, label="large")
        ax.axhline(0.06, color="orange", linestyle="--", alpha=0.4, label="medium")
        save_fig(fig, "cross_eta_sq_comparison")

    # ── Plot 2: η² scatter — CD2 vs each other source ────────────────
    if "lakh_cd2" in source_eta and len(source_eta) > 1:
        cd2_eta = source_eta["lakh_cd2"]
        for src_name, src_eta_vals in source_eta.items():
            if src_name == "lakh_cd2":
                continue
            factors_common = sorted(set(cd2_eta.keys()) & set(src_eta_vals.keys()))
            if len(factors_common) < 2:
                continue
            fig, ax = plt.subplots(figsize=(7, 7))
            x_vals = [cd2_eta[f] for f in factors_common]
            y_vals = [src_eta_vals[f] for f in factors_common]
            ax.scatter(x_vals, y_vals, s=100, zorder=5)
            for f, xv, yv in zip(factors_common, x_vals, y_vals):
                ax.annotate(f, (xv, yv), textcoords="offset points",
                            xytext=(8, 8), fontsize=9)
            lim_max = max(max(x_vals), max(y_vals)) * 1.2
            ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, label="y = x")
            ax.set_xlabel("η² (Lakh CD2)")
            ax.set_ylabel(f"η² ({src_name})")
            ax.set_title(f"η² Consistency: Lakh CD2 vs {src_name}")
            ax.legend()
            save_fig(fig, f"cross_eta_scatter_{src_name}")

    # ── Plot 3: Tokenizer×Model heatmaps side by side ────────────────
    if source_cells:
        n_sources = len(source_cells)
        fig, axes = plt.subplots(1, n_sources, figsize=(6 * n_sources, 5))
        if n_sources == 1:
            axes = [axes]

        for ax, (src_name, cells_df) in zip(axes, source_cells.items()):
            pivot = cells_df.pivot_table(
                values="mean_fmd", index="tokenizer", columns="model"
            )
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                        ax=ax, linewidths=0.5, vmin=0)
            ax.set_title(f"Mean FMD: {src_name}")

        fig.suptitle("Tokenizer × Model Interaction: Cross-Dataset Comparison",
                      fontsize=14, y=1.02)
        save_fig(fig, "cross_tok_model_heatmaps")

    # ── Plot 4: FMD violin per source ─────────────────────────────────
    if len(sources) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=all_df, x="source", y="fmd", ax=ax,
                       palette="Set2", inner="quart", cut=0)
        ax.set_title("FMD Distribution by Data Source")
        ax.set_ylabel("FMD")
        save_fig(fig, "cross_fmd_violin_by_source")

    # ── Plot 5: FMD by tokenizer×model, faceted by source ────────────
    if len(sources) > 1:
        g = sns.catplot(
            data=all_df, x="tokenizer", y="fmd", hue="model",
            col="source", kind="violin", split=True,
            height=5, aspect=1.3, palette="Set1", inner="quart", cut=0,
        )
        g.figure.suptitle("FMD by Tokenizer × Model per Source", y=1.02, fontsize=14)
        save_fig(g.figure, "cross_fmd_tok_model_by_source")

    # ── Plot 6: Ranking agreement bar chart ───────────────────────────
    if ranking_comparisons:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(ranking_comparisons.keys())
        rhos = [ranking_comparisons[k].get("spearman_rho", 0) or 0 for k in labels]
        colors_bar = ["#2ecc71" if r > 0.7 else "#f39c12" if r > 0.4 else "#e74c3c" for r in rhos]
        bars = ax.bar(labels, rhos, color=colors_bar)
        for bar, v in zip(bars, rhos):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"ρ={v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Spearman ρ")
        ax.set_title("Pipeline Ranking Agreement Across Sources")
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(0.7, color="green", linestyle="--", alpha=0.3, label="strong agreement")
        ax.legend()
        save_fig(fig, "cross_ranking_agreement")

    # ── Plot 7: Grand summary 4-panel ─────────────────────────────────
    if len(sources) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # A: FMD violin by source
        sns.violinplot(data=all_df, x="source", y="fmd", ax=axes[0, 0],
                       palette="Set2", inner="quart", cut=0)
        axes[0, 0].set_title("A) FMD Distribution by Source")

        # B: η² comparison
        if source_eta:
            factors = ["tokenizer", "model", "preprocess"]
            x = np.arange(len(factors))
            width = 0.8 / max(len(sources), 1)
            colors_pal = sns.color_palette("Set2", len(sources))
            for i, src in enumerate(sources):
                eta = source_eta.get(src, {})
                vals = [eta.get(f, 0) for f in factors]
                axes[0, 1].bar(x + i * width - 0.4 + width / 2, vals, width,
                               label=src, color=colors_pal[i])
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(factors)
            axes[0, 1].set_ylabel("η²")
            axes[0, 1].set_title("B) η² Comparison")
            axes[0, 1].legend(fontsize=8)

        # C: Tok×Model heatmap (first source)
        if source_cells:
            first_src = list(source_cells.keys())[0]
            pivot = source_cells[first_src].pivot_table(
                values="mean_fmd", index="tokenizer", columns="model"
            )
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                        ax=axes[1, 0], linewidths=0.5, vmin=0)
            axes[1, 0].set_title(f"C) Tok×Model: {first_src}")

        # D: Ranking agreement
        if ranking_comparisons:
            labels_rc = list(ranking_comparisons.keys())
            rhos_rc = [ranking_comparisons[k].get("spearman_rho", 0) or 0 for k in labels_rc]
            colors_rc = ["#2ecc71" if r > 0.7 else "#f39c12" if r > 0.4 else "#e74c3c" for r in rhos_rc]
            axes[1, 1].bar(labels_rc, rhos_rc, color=colors_rc)
            axes[1, 1].set_ylabel("Spearman ρ")
            axes[1, 1].set_title("D) Ranking Agreement")
            axes[1, 1].set_ylim(-0.1, 1.1)

        save_fig(fig, "cross_summary_4panel")


# ──────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────

def generate_report(
    all_df: pd.DataFrame,
    source_eta: Dict[str, Dict[str, float]],
    source_rankings: Dict[str, pd.DataFrame],
    source_cells: Dict[str, pd.DataFrame],
    ranking_comparisons: Dict[str, Dict],
    output_dir: Path,
):
    """Generate comprehensive Markdown report."""
    report = []
    sources = sorted(all_df["source"].unique())

    report.append("# Cross-Dataset Validation Report")
    report.append(f"\n**Generalizability analysis of FMD sensitivity findings.**\n")
    report.append(f"## Design")
    report.append(f"- **Sources:** {', '.join(sources)}")
    report.append(f"- **Total FMD observations:** {len(all_df)}")
    for src in sources:
        sub = all_df[all_df["source"] == src]
        n_pairs = sub["pair"].nunique()
        report.append(f"  - {src}: {len(sub)} rows, {n_pairs} pairs")

    # η² comparison
    report.append(f"\n## η² Variance Decomposition by Source")
    report.append(f"\n| Factor | " + " | ".join(sources) + " |")
    report.append(f"|--------| " + " | ".join(["---"] * len(sources)) + " |")
    factors = ["tokenizer", "model", "preprocess"]
    for f in factors:
        vals = [f"{source_eta.get(s, {}).get(f, 0):.4f}" for s in sources]
        report.append(f"| {f} | " + " | ".join(vals) + " |")

    # Cell means comparison
    report.append(f"\n## Tokenizer × Model Cell Means by Source")
    for src_name, cells_df in source_cells.items():
        report.append(f"\n### {src_name}")
        report.append(f"\n| Tokenizer | Model | Mean FMD | Std | N |")
        report.append(f"|-----------|-------|----------|-----|---|")
        for _, row in cells_df.iterrows():
            report.append(
                f"| {row['tokenizer']} | {row['model']} | "
                f"{row['mean_fmd']:.4f} | {row['std_fmd']:.4f} | {int(row['n'])} |"
            )

    # Ranking agreement
    report.append(f"\n## Pipeline Ranking Agreement")
    if ranking_comparisons:
        report.append(f"\n| Comparison | Spearman ρ | p-value | N common |")
        report.append(f"|------------|-----------|---------|----------|")
        for comp_name, comp_vals in ranking_comparisons.items():
            rho = comp_vals.get("spearman_rho")
            p = comp_vals.get("p_value")
            n = comp_vals.get("n_common", 0)
            rho_str = f"{rho:.4f}" if rho is not None else "N/A"
            p_str = f"{p:.4e}" if p is not None else "N/A"
            report.append(f"| {comp_name} | {rho_str} | {p_str} | {n} |")
    else:
        report.append("\nInsufficient data for ranking comparison.")

    # Interpretation
    report.append(f"\n## Interpretation")
    report.append(f"\n### Generalizability Assessment\n")

    if ranking_comparisons:
        rhos = [v.get("spearman_rho", 0) or 0 for v in ranking_comparisons.values()]
        avg_rho = np.mean(rhos) if rhos else 0

        if avg_rho > 0.7:
            report.append(
                f"✅ **Strong generalizability** (avg ρ = {avg_rho:.3f}): "
                f"Pipeline rankings are highly consistent across data sources. "
                f"The sensitivity findings from Lakh CD2 replicate well."
            )
        elif avg_rho > 0.4:
            report.append(
                f"⚠️ **Moderate generalizability** (avg ρ = {avg_rho:.3f}): "
                f"Pipeline rankings show partial agreement across sources. "
                f"Main trends hold but some variation exists."
            )
        else:
            report.append(
                f"❌ **Weak generalizability** (avg ρ = {avg_rho:.3f}): "
                f"Pipeline rankings differ substantially across sources. "
                f"Findings may be dataset-specific."
            )

    # η² consistency
    report.append(f"\n### η² Consistency\n")
    for f in factors:
        vals = [source_eta.get(s, {}).get(f, 0) for s in sources]
        if len(vals) > 1:
            cv = np.std(vals) / np.mean(vals) if np.mean(vals) > 0.001 else float("inf")
            report.append(
                f"- **{f}**: range [{min(vals):.4f}, {max(vals):.4f}], "
                f"CV = {cv:.2f}"
            )

    # Conclusions
    report.append(f"\n## Conclusions\n")
    report.append(
        f"1. Cross-dataset validation tested {len(sources)} independent data sources.\n"
        f"2. Total observations: {len(all_df)}.\n"
        f"3. See plots in `results/plots/paper/cross_*.png` for visual comparison.\n"
    )

    report_path = output_dir / "CROSS_VALIDATION_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    logger.info(f"✅ Report: {report_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-dataset validation for FMD sensitivity")
    parser.add_argument(
        "--source",
        choices=["cd1", "midicaps", "all"],
        default="all",
        help="Which source(s) to run: cd1, midicaps, or all (default: all)",
    )
    parser.add_argument("--skip-cd2", action="store_true", help="Skip loading existing CD2 results")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CROSS-DATASET VALIDATION FOR FMD SENSITIVITY")
    logger.info("=" * 70)

    config = load_config()
    cv_cfg = config.get("cross_validation", {})
    subsample_size = cv_cfg.get("subsample_size", 100)
    n_repeats = cv_cfg.get("n_repeats", 10)
    output_dir = Path(cv_cfg.get("output_dir", "results/reports/cross_validation"))
    plots_dir = Path(cv_cfg.get("plots_dir", "results/plots/paper"))
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_manager = DatasetManager(config)
    preprocessor = MIDIPreprocessor(config)
    tokenization = TokenizationPipeline(config)
    emb_extractor = EmbeddingExtractor(config)
    fmd_calc = FrechetMusicDistance(config)

    genres = config["lakh"]["genres"]  # [rock, jazz, electronic, country]
    variants = build_variants(config)

    logger.info(f"Genres: {genres}")
    logger.info(f"Variants: {len(variants)}")
    logger.info(f"Source mode: {args.source}")

    all_dfs: List[pd.DataFrame] = []

    # ── Load existing CD2 results ─────────────────────────────────────
    if not args.skip_cd2:
        cd2_df = load_existing_cd2(output_dir)
        if cd2_df is not None:
            all_dfs.append(cd2_df)

    # ── Source B: Tagtraum CD1 ────────────────────────────────────────
    if args.source in ("cd1", "all"):
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING SOURCE: Lakh + Tagtraum CD1")
        logger.info("=" * 70)

        cd1_loader = LakhGenreLoader(config, tagtraum_version="cd1")
        cd1_loader.ensure_data()
        cd1_counts = cd1_loader.populate_raw_datasets()
        logger.info(f"CD1 populated: {cd1_counts}")

        cd1_df = run_source(
            source_name="lakh_cd1",
            dataset_prefix="lakh_cd1",
            genres=genres,
            config=config,
            dataset_manager=dataset_manager,
            preprocessor=preprocessor,
            tokenization=tokenization,
            emb_extractor=emb_extractor,
            fmd_calc=fmd_calc,
            variants=variants,
            subsample_size=subsample_size,
            n_repeats=n_repeats,
            max_files=cv_cfg.get("tagtraum_cd1", {}).get("max_per_genre", 500),
        )
        if not cd1_df.empty:
            all_dfs.append(cd1_df)
            cd1_df.to_csv(output_dir / "cd1_fmd.csv", index=False)
            logger.info(f"Saved CD1 results: {len(cd1_df)} rows")

    # ── Source C: MidiCaps ────────────────────────────────────────────
    if args.source in ("midicaps", "all"):
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING SOURCE: MidiCaps")
        logger.info("=" * 70)

        mc_loader = MidiCapsGenreLoader(config)
        mc_loader.ensure_data()

        # Log discovered tags for diagnostics
        tags = mc_loader.discover_genre_tags()
        logger.info(f"MidiCaps genre tags discovered: {tags}")

        mc_counts = mc_loader.populate_raw_datasets()
        logger.info(f"MidiCaps populated: {mc_counts}")

        mc_df = run_source(
            source_name="midicaps",
            dataset_prefix="midicaps",
            genres=genres,
            config=config,
            dataset_manager=dataset_manager,
            preprocessor=preprocessor,
            tokenization=tokenization,
            emb_extractor=emb_extractor,
            fmd_calc=fmd_calc,
            variants=variants,
            subsample_size=subsample_size,
            n_repeats=n_repeats,
            max_files=cv_cfg.get("midicaps", {}).get("max_per_genre", 120),
        )
        if not mc_df.empty:
            all_dfs.append(mc_df)
            mc_df.to_csv(output_dir / "midicaps_fmd.csv", index=False)
            logger.info(f"Saved MidiCaps results: {len(mc_df)} rows")

    # ── Merge and analyze ─────────────────────────────────────────────
    if not all_dfs:
        logger.error("No data collected from any source!")
        return

    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df.to_csv(output_dir / "cross_dataset_fmd.csv", index=False)
    logger.info(f"\nTotal cross-dataset observations: {len(all_df)}")
    logger.info(f"Sources: {all_df['source'].unique().tolist()}")

    # ── Statistical comparison ────────────────────────────────────────
    logger.info("\n=== Statistical Comparison ===")

    # Preprocess column for η²
    all_df["preprocess"] = (
        all_df["remove_velocity"].astype(str) + "_" + all_df["hard_quantization"].astype(str)
    )

    sources = sorted(all_df["source"].unique())
    source_eta: Dict[str, Dict[str, float]] = {}
    source_rankings: Dict[str, pd.DataFrame] = {}
    source_cells: Dict[str, pd.DataFrame] = {}

    for src in sources:
        sub = all_df[all_df["source"] == src]
        eta = compute_eta_squared(sub, ["tokenizer", "model", "preprocess"])
        source_eta[src] = eta
        source_rankings[src] = compute_pipeline_ranking(sub)
        source_cells[src] = compute_cell_means(sub)
        logger.info(f"  {src}: η² = {eta}")

    # Pairwise ranking comparisons
    ranking_comparisons: Dict[str, Dict] = {}
    for i, src_a in enumerate(sources):
        for src_b in sources[i + 1:]:
            comp_name = f"{src_a} vs {src_b}"
            comp = compare_rankings(source_rankings[src_a], source_rankings[src_b])
            ranking_comparisons[comp_name] = comp
            logger.info(f"  Ranking {comp_name}: ρ={comp.get('spearman_rho'):.4f}" if comp.get("spearman_rho") else f"  Ranking {comp_name}: insufficient overlap")

    # Save analysis JSON
    analysis_json = {
        "sources": sources,
        "total_rows": len(all_df),
        "eta_squared_by_source": source_eta,
        "ranking_comparisons": ranking_comparisons,
        "cell_means_by_source": {
            s: cells.to_dict(orient="records") for s, cells in source_cells.items()
        },
    }
    with open(output_dir / "cross_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(analysis_json, f, indent=2, default=str)

    # ── Generate plots ────────────────────────────────────────────────
    logger.info("\n=== Generating Cross-Validation Plots ===")
    generate_cross_validation_plots(
        all_df, source_eta, source_rankings, source_cells,
        ranking_comparisons, plots_dir,
    )

    # ── Generate report ───────────────────────────────────────────────
    logger.info("\n=== Generating Report ===")
    generate_report(
        all_df, source_eta, source_rankings, source_cells,
        ranking_comparisons, output_dir,
    )

    logger.info("\n" + "=" * 70)
    logger.info("CROSS-DATASET VALIDATION COMPLETE")
    logger.info(f"  Results: {output_dir}/")
    logger.info(f"  Plots:   {plots_dir}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

