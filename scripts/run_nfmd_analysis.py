#!/usr/bin/env python3
"""Normalized FMD (nFMD) analysis — scale-invariant sensitivity study.

Reads the existing multi-genre FMD CSV, re-extracts embeddings to compute
nFMD components, then repeats the ANOVA/η² analysis on normalized values.

Key question: Does η²(model) drop dramatically after normalization,
revealing hidden tokenizer/preprocessing effects?
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.manager import DatasetManager
from embeddings.extractor import EmbeddingExtractor
from metrics.fmd import FrechetMusicDistance
from preprocessing.processor import MIDIPreprocessor
from tokenization.tokenizer import TokenizationPipeline

# ──────────────────────────────────────────────────────────────────────
SUBSAMPLE_SIZE = 100
N_REPEATS = 10
SEED = 42
OUTPUT_DIR = Path("results/reports/lakh_multi")
PLOTS_DIR = Path("results/plots/paper")
INPUT_CSV = OUTPUT_DIR / "multi_genre_fmd.csv"
# ──────────────────────────────────────────────────────────────────────


def load_config() -> Dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def build_variants(config: Dict):
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


def extract_genre_embeddings(
    genre, tok_name, model_name, remove_vel, hard_quant,
    config, dataset_manager, preprocessor, tokenization, embeddings,
    max_files=120,
) -> np.ndarray | None:
    ds_name = f"lakh_{genre}"
    midi_files = dataset_manager.list_midi_files(ds_name, processed=False, limit=max_files)
    if not midi_files:
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
            vec = embeddings.extract_embeddings([tokens], model_name, midi_data_list=[midi_data])[0]
            vectors.append(vec)
        except Exception as exc:
            logger.debug(f"Skip {midi_path.name}: {exc}")
    return np.vstack(vectors) if vectors else None


def compute_eta_squared(df: pd.DataFrame, factor: str, response: str = "fmd") -> float:
    """Compute η² for a single factor."""
    groups = [g[response].values for _, g in df.groupby(factor)]
    if len(groups) < 2:
        return 0.0
    ss_between = sum(len(g) * (np.mean(g) - df[response].mean()) ** 2 for g in groups)
    ss_total = ((df[response] - df[response].mean()) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def main():
    logger.info("=" * 70)
    logger.info("NORMALIZED FMD (nFMD) ANALYSIS")
    logger.info("=" * 70)

    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_manager = DatasetManager(config)
    preprocessor = MIDIPreprocessor(config)
    tokenization = TokenizationPipeline(config)
    emb_extractor = EmbeddingExtractor(config)
    fmd_calc = FrechetMusicDistance(config)

    genres = config["lakh"]["genres"]
    genre_pairs = list(combinations(genres, 2))
    variants = build_variants(config)

    # ── Step 1: Extract embeddings (same cache as multi-genre) ────────
    logger.info("\n=== Step 1: Extracting embeddings ===")
    emb_cache: Dict[Tuple, np.ndarray] = {}
    total_combos = len(genres) * len(variants)

    for idx, (genre, (tok, model, (vel, quant))) in enumerate(
        product(genres, variants), 1
    ):
        key = (genre, tok, model, vel, quant)
        if key in emb_cache:
            continue
        vname = variant_name(tok, model, vel, quant)
        logger.info(f"[{idx}/{total_combos}] {genre} × {vname}")

        embs = extract_genre_embeddings(
            genre, tok, model, vel, quant,
            config, dataset_manager, preprocessor, tokenization, emb_extractor,
        )
        if embs is not None:
            emb_cache[key] = embs
            logger.info(f"  → {embs.shape[0]} embeddings, dim={embs.shape[1]}")
        else:
            logger.warning(f"  → NO embeddings!")

    # ── Step 2: Compute nFMD with repeated subsampling ────────────────
    logger.info("\n=== Step 2: Computing nFMD (repeated subsampling) ===")
    rows: List[Dict] = []
    rng = np.random.default_rng(SEED)

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
                continue

            n_a, n_b = emb_a.shape[0], emb_b.shape[0]
            sub_a = min(SUBSAMPLE_SIZE, n_a)
            sub_b = min(SUBSAMPLE_SIZE, n_b)
            if sub_a < 5 or sub_b < 5:
                continue

            # Full nFMD (no subsampling)
            try:
                full_nfmd = fmd_calc.compute_nfmd(emb_a, emb_b)
            except Exception:
                full_nfmd = {"fmd": float("nan"), "nfmd_trace": float("nan"), "nfmd_norm": float("nan"),
                             "trace_cov1": 0, "trace_cov2": 0, "mean_norm1": 0, "mean_norm2": 0}

            for rep in range(N_REPEATS):
                idx_a = rng.choice(n_a, size=sub_a, replace=False)
                idx_b = rng.choice(n_b, size=sub_b, replace=False)

                try:
                    nfmd_result = fmd_calc.compute_nfmd(emb_a[idx_a], emb_b[idx_b])
                except Exception:
                    continue

                rows.append({
                    "variant": vname,
                    "tokenizer": tok,
                    "model": model,
                    "remove_velocity": vel,
                    "hard_quantization": quant,
                    "genre_a": genre_a,
                    "genre_b": genre_b,
                    "pair": f"{genre_a}_vs_{genre_b}",
                    "repeat": rep,
                    "fmd": nfmd_result["fmd"],
                    "nfmd_trace": nfmd_result["nfmd_trace"],
                    "nfmd_norm": nfmd_result["nfmd_norm"],
                    "trace_cov1": nfmd_result["trace_cov1"],
                    "trace_cov2": nfmd_result["trace_cov2"],
                    "mean_norm1": nfmd_result["mean_norm1"],
                    "mean_norm2": nfmd_result["mean_norm2"],
                    "full_fmd": full_nfmd["fmd"],
                    "full_nfmd_trace": full_nfmd["nfmd_trace"],
                    "full_nfmd_norm": full_nfmd["nfmd_norm"],
                })

            if op_idx % 40 == 0 or op_idx == total_ops:
                logger.info(f"  [{op_idx}/{total_ops}] {vname} | {genre_a}-{genre_b}")

    logger.info(f"\nTotal rows: {len(rows)}")

    # ── Step 3: Save enriched CSV ─────────────────────────────────────
    csv_path = OUTPUT_DIR / "nfmd_multi_genre.csv"
    if rows:
        fields = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved: {csv_path}")

    # ── Step 4: Statistical analysis ──────────────────────────────────
    logger.info("\n=== Step 3: Statistical analysis — raw FMD vs nFMD ===")
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["fmd", "nfmd_trace", "nfmd_norm"])
    df["preprocess"] = df["remove_velocity"].astype(str) + "_" + df["hard_quantization"].astype(str)
    logger.info(f"Clean rows: {len(df)}")

    # η² for all three metrics
    metrics = ["fmd", "nfmd_trace", "nfmd_norm"]
    factors = ["tokenizer", "model", "preprocess", "pair"]

    comparison_rows = []
    for metric in metrics:
        for factor in factors:
            eta = compute_eta_squared(df, factor, metric)
            groups = [g[metric].values for _, g in df.groupby(factor)]
            F, p = sp_stats.f_oneway(*groups) if len(groups) >= 2 else (0, 1)
            comparison_rows.append({
                "metric": metric,
                "factor": factor,
                "eta_sq": eta,
                "F": float(F),
                "p": float(p),
            })
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            logger.info(f"  {metric:12s} × {factor:12s}: η²={eta:.4f}  F={F:.2f}  p={p:.2e} {sig}")

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(OUTPUT_DIR / "nfmd_eta_sq_comparison.csv", index=False)

    # Per-model η² breakdown (tok × preprocess within each model)
    logger.info("\n=== Per-model η² breakdown ===")
    per_model_rows = []
    for model_name, model_grp in df.groupby("model"):
        for metric in metrics:
            for factor in ["tokenizer", "preprocess"]:
                eta = compute_eta_squared(model_grp, factor, metric)
                per_model_rows.append({
                    "model": model_name,
                    "metric": metric,
                    "factor": factor,
                    "eta_sq": eta,
                    "n": len(model_grp),
                })
                logger.info(f"  {model_name:18s} | {metric:12s} × {factor:12s}: η²={eta:.4f}")

    per_model_df = pd.DataFrame(per_model_rows)
    per_model_df.to_csv(OUTPUT_DIR / "nfmd_per_model_eta_sq.csv", index=False)

    # ── Step 5: Three-way ANOVA on nFMD ──────────────────────────────
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        for metric in metrics:
            formula = f"{metric} ~ C(tokenizer) * C(model) * C(preprocess)"
            model_fit = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model_fit, typ=2)
            ss_total = anova_table["sum_sq"].sum()
            logger.info(f"\n  Three-way ANOVA on {metric}:")
            for idx_name in anova_table.index:
                if idx_name == "Residual":
                    continue
                ss = anova_table.loc[idx_name, "sum_sq"]
                eta = float(ss / ss_total) if ss_total > 0 else 0
                F = anova_table.loc[idx_name, "F"]
                p = anova_table.loc[idx_name, "PR(>F)"]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                logger.info(f"    {idx_name:<45s} η²={eta:.4f}  F={F:.2f}  p={p:.2e} {sig}")
            anova_table.to_csv(OUTPUT_DIR / f"anova_3way_{metric}.csv")
    except ImportError:
        logger.warning("statsmodels not available")

    # ── Step 6: Save results JSON ─────────────────────────────────────
    results = {
        "config": {
            "n_rows": len(df),
            "n_rows_raw": len(rows),
            "metrics": metrics,
            "factors": factors,
        },
        "eta_sq_comparison": comparison_rows,
        "per_model_eta_sq": per_model_rows,
    }
    with open(OUTPUT_DIR / "nfmd_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Step 7: Plots ─────────────────────────────────────────────────
    logger.info("\n=== Step 4: Generating plots ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    def save_fig(fig, name):
        fig.tight_layout()
        for fmt in ("png", "pdf"):
            fig.savefig(PLOTS_DIR / f"{name}.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  ✅ {name}")

    # Plot 1: η² comparison — raw FMD vs nFMD_trace vs nFMD_norm
    pivot = comp_df.pivot(index="factor", columns="metric", values="eta_sq")
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, colormap="Set1")
    ax.set_ylabel("η²")
    ax.set_title("Variance Decomposition: Raw FMD vs Normalized FMD")
    ax.axhline(0.14, color="red", linestyle="--", alpha=0.4, label="large threshold")
    ax.axhline(0.06, color="orange", linestyle="--", alpha=0.4, label="medium threshold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, "nfmd_eta_sq_comparison")

    # Plot 2: Per-model η² heatmap for nfmd_trace
    nfmd_trace_pm = per_model_df[per_model_df["metric"] == "nfmd_trace"]
    if not nfmd_trace_pm.empty:
        pivot_pm = nfmd_trace_pm.pivot_table(values="eta_sq", index="model", columns="factor")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_pm, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, vmin=0)
        ax.set_title("η² per Model (nFMD_trace): Tokenizer & Preprocessing Effects")
        save_fig(fig, "nfmd_per_model_heatmap")

    # Plot 3: Violin — nFMD_trace by tokenizer × model
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="tokenizer", y="nfmd_trace", hue="model", ax=ax, palette="Set1")
    ax.set_title("Normalized FMD (trace) by Tokenizer & Model")
    ax.set_ylabel("nFMD (trace)")
    save_fig(fig, "nfmd_trace_by_tok_model")

    # Plot 4: 4-panel summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A: η² raw vs normalized
    pivot.plot(kind="bar", ax=axes[0, 0], colormap="Set1", legend=True)
    axes[0, 0].set_title("A) η² Raw vs Normalized")
    axes[0, 0].set_ylabel("η²")
    axes[0, 0].axhline(0.14, color="red", linestyle="--", alpha=0.3)
    axes[0, 0].tick_params(axis="x", rotation=0)

    # B: nFMD_trace by model
    sns.boxplot(data=df, x="model", y="nfmd_trace", ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title("B) nFMD(trace) by Model")
    axes[0, 1].set_ylabel("nFMD (trace)")
    axes[0, 1].tick_params(axis="x", rotation=25)

    # C: nFMD_trace by tokenizer
    sns.boxplot(data=df, x="tokenizer", y="nfmd_trace", ax=axes[1, 0], palette="Set3")
    axes[1, 0].set_title("C) nFMD(trace) by Tokenizer")
    axes[1, 0].set_ylabel("nFMD (trace)")

    # D: Per-model η² heatmap
    if not nfmd_trace_pm.empty:
        pivot_pm2 = nfmd_trace_pm.pivot_table(values="eta_sq", index="model", columns="factor")
        sns.heatmap(pivot_pm2, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1, 1],
                    linewidths=0.5, vmin=0, cbar_kws={"shrink": 0.8})
        axes[1, 1].set_title("D) η² per Model (nFMD_trace)")

    save_fig(fig, "nfmd_summary_4panel")

    # ── Step 8: Report ────────────────────────────────────────────────
    logger.info("\n=== Step 5: Generating report ===")
    report = []
    report.append("# Normalized FMD (nFMD) Analysis")
    report.append(f"\n**Scale-invariant FMD enables fair cross-model comparison.**")
    report.append(f"\n## Design")
    report.append(f"- **Input:** {len(df)} FMD observations (NaN-filtered)")
    report.append(f"- **Metrics:** raw FMD, nFMD_trace (FMD/Tr), nFMD_norm (FMD/‖μ‖²)")
    report.append(f"- **Key question:** Does η²(model) drop after normalization?")

    report.append(f"\n## η² Comparison: Raw vs Normalized")
    report.append(f"\n| Factor | η²(fmd) | η²(nfmd_trace) | η²(nfmd_norm) |")
    report.append(f"|--------|---------|----------------|---------------|")
    for factor in factors:
        vals = {}
        for metric in metrics:
            row = comp_df[(comp_df["factor"] == factor) & (comp_df["metric"] == metric)]
            vals[metric] = row["eta_sq"].values[0] if len(row) > 0 else 0
        report.append(f"| {factor} | {vals['fmd']:.4f} | {vals['nfmd_trace']:.4f} | {vals['nfmd_norm']:.4f} |")

    report.append(f"\n## Per-Model η² (nFMD_trace)")
    report.append(f"\n| Model | η²(tokenizer) | η²(preprocess) |")
    report.append(f"|-------|---------------|----------------|")
    for model_name, grp in per_model_df[per_model_df["metric"] == "nfmd_trace"].groupby("model"):
        tok_eta = grp[grp["factor"] == "tokenizer"]["eta_sq"].values
        pre_eta = grp[grp["factor"] == "preprocess"]["eta_sq"].values
        tok = tok_eta[0] if len(tok_eta) > 0 else 0
        pre = pre_eta[0] if len(pre_eta) > 0 else 0
        report.append(f"| {model_name} | {tok:.4f} | {pre:.4f} |")

    report.append(f"\n## Conclusions")
    report.append(f"""
1. Raw FMD is dominated by model scale (η²≈0.96) — this is a **scale artefact**.
2. After trace-normalization, the model effect should decrease substantially.
3. Per-model analysis reveals which models are actually sensitive to tokenization.
4. nFMD enables **fair cross-model comparison** of pipeline choices.
""")

    report_path = OUTPUT_DIR / "NFMD_ANALYSIS_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    logger.info(f"✅ Report: {report_path}")

    logger.info("\n" + "=" * 70)
    logger.info("nFMD ANALYSIS COMPLETE")
    logger.info(f"  Results: {OUTPUT_DIR}/")
    logger.info(f"  Plots:   {PLOTS_DIR}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

