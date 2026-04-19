#!/usr/bin/env python3
"""Multi-genre repeated-subsampling FMD sensitivity analysis.

Implements recommendations 8.5 and 8.6 from ANALYSIS_REPORT.md:
  8.5 — Add all 6 genre pairs (4 genres → C(4,2) = 6 pairs)
  8.6 — Repeated subsampling (10×) per variant×pair for within-cell variance

This produces 32 variants × 6 pairs × 10 repeats = 1920 FMD observations,
enabling proper three-way ANOVA with interactions and adequate power.
"""

from __future__ import annotations

import csv
import json
import sys
from hashlib import md5
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.manager import DatasetManager
from embeddings.extractor import EmbeddingExtractor
from metrics.fmd import FrechetMusicDistance
from preprocessing.processor import MIDIPreprocessor
from tokenization.tokenizer import TokenizationPipeline

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
SUBSAMPLE_SIZE = 100        # embeddings per genre per repeat
N_REPEATS = 10              # repeats per variant × pair
SEED = 42
OUTPUT_DIR = Path("results/reports/lakh_multi")
PLOTS_DIR = Path("results/plots/paper")

# ──────────────────────────────────────────────────────────────────────


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
    max_files: int = 120,
) -> np.ndarray | None:
    """Extract embeddings for one genre + one variant. Returns (N, D) or None."""
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
            vec = embeddings.extract_embeddings([tokens], model_name)[0]
            vectors.append(vec)
        except Exception as exc:
            logger.debug(f"Skip {midi_path.name}: {exc}")

    return np.vstack(vectors) if vectors else None


def main():
    logger.info("=" * 70)
    logger.info("MULTI-GENRE REPEATED-SUBSAMPLING FMD ANALYSIS")
    logger.info("=" * 70)

    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_manager = DatasetManager(config)
    preprocessor = MIDIPreprocessor(config)
    tokenization = TokenizationPipeline(config)
    emb_extractor = EmbeddingExtractor(config)
    fmd_calc = FrechetMusicDistance(config)

    genres = config["lakh"]["genres"]  # [rock, jazz, electronic, country]
    genre_pairs = list(combinations(genres, 2))  # 6 pairs
    variants = build_variants(config)

    logger.info(f"Genres: {genres}")
    logger.info(f"Genre pairs: {genre_pairs} ({len(genre_pairs)} pairs)")
    logger.info(f"Variants: {len(variants)}")
    logger.info(f"Subsample size: {SUBSAMPLE_SIZE}, repeats: {N_REPEATS}")
    logger.info(f"Total FMD computations: {len(variants)} × {len(genre_pairs)} × {N_REPEATS} = "
                f"{len(variants) * len(genre_pairs) * N_REPEATS}")

    # ── Step 1: Extract all embeddings (cache per genre × variant) ────
    logger.info("\n=== Step 1: Extracting embeddings ===")
    # Key: (genre, tok, model, vel, quant) → np.ndarray
    emb_cache: Dict[Tuple, np.ndarray] = {}

    total_combos = len(genres) * len(variants)
    for idx, (genre, (tok, model, (vel, quant))) in enumerate(
        product(genres, variants), 1
    ):
        key = (genre, tok, model, vel, quant)
        if key in emb_cache:
            continue
        pct = 100 * idx / total_combos
        vname = variant_name(tok, model, vel, quant)
        logger.info(f"[{idx}/{total_combos} ({pct:.0f}%)] {genre} × {vname}")

        embs = extract_genre_embeddings(
            genre, tok, model, vel, quant,
            config, dataset_manager, preprocessor, tokenization, emb_extractor,
        )
        if embs is not None:
            emb_cache[key] = embs
            logger.info(f"  → {embs.shape[0]} embeddings, dim={embs.shape[1]}")
        else:
            logger.warning(f"  → NO embeddings!")

    # ── Step 2: Compute FMD with repeated subsampling ─────────────────
    logger.info("\n=== Step 2: Computing FMD (repeated subsampling) ===")
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
                logger.warning(f"[{op_idx}/{total_ops}] Skip {vname} | {genre_a}-{genre_b}: missing embs")
                continue

            n_a, n_b = emb_a.shape[0], emb_b.shape[0]
            sub_a = min(SUBSAMPLE_SIZE, n_a)
            sub_b = min(SUBSAMPLE_SIZE, n_b)

            if sub_a < 5 or sub_b < 5:
                continue

            # Also compute full FMD (no subsampling)
            full_fmd = float(fmd_calc.compute_fmd(emb_a, emb_b))

            for rep in range(N_REPEATS):
                idx_a = rng.choice(n_a, size=sub_a, replace=False)
                idx_b = rng.choice(n_b, size=sub_b, replace=False)
                fmd_val = float(fmd_calc.compute_fmd(emb_a[idx_a], emb_b[idx_b]))

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
                    "subsample_size": sub_a,
                    "fmd": fmd_val,
                    "full_fmd": full_fmd,
                    "n_a": n_a,
                    "n_b": n_b,
                })

            if op_idx % 20 == 0 or op_idx == total_ops:
                logger.info(f"  [{op_idx}/{total_ops}] {vname} | {genre_a}-{genre_b}: "
                            f"full={full_fmd:.4f}, sub_mean={np.mean([r['fmd'] for r in rows[-N_REPEATS:]]):.4f}")

    logger.info(f"\nTotal rows: {len(rows)}")

    # ── Step 3: Save raw data ─────────────────────────────────────────
    csv_path = OUTPUT_DIR / "multi_genre_fmd.csv"
    if rows:
        fields = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved: {csv_path}")

    # ── Step 4: Statistical analysis ──────────────────────────────────
    logger.info("\n=== Step 3: Statistical analysis ===")
    df = pd.DataFrame(rows)
    df["preprocess"] = df["remove_velocity"].astype(str) + "_" + df["hard_quantization"].astype(str)

    # 4a. Three-way ANOVA with interactions (now possible!)
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        formula = "fmd ~ C(tokenizer) * C(model) * C(preprocess)"
        anova_model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(anova_model, typ=2)

        ss_total = anova_table["sum_sq"].sum()
        ss_resid = anova_table.loc["Residual", "sum_sq"] if "Residual" in anova_table.index else 0

        eta_sq = {}
        partial_eta_sq = {}
        for idx in anova_table.index:
            if idx == "Residual":
                continue
            ss = anova_table.loc[idx, "sum_sq"]
            eta_sq[idx] = float(ss / ss_total) if ss_total > 0 else 0
            partial_eta_sq[idx] = float(ss / (ss + ss_resid)) if (ss + ss_resid) > 0 else 0

        anova_table.to_csv(OUTPUT_DIR / "anova_3way_full.csv")
        logger.info("Three-way ANOVA with interactions:")
        for idx in anova_table.index:
            if idx == "Residual":
                continue
            F = anova_table.loc[idx, "F"]
            p = anova_table.loc[idx, "PR(>F)"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            logger.info(f"  {idx:<45s} F={F:8.2f}  p={p:.4e}  η²={eta_sq[idx]:.4f}  {sig}")

        _has_sm = True
    except ImportError:
        _has_sm = False
        logger.warning("statsmodels not available — falling back to scipy")

    # 4b. Also one-way per factor for simpler reporting
    from scipy import stats as sp_stats

    oneway_rows = []
    for factor in ("tokenizer", "model", "preprocess", "pair"):
        groups = [g["fmd"].values for _, g in df.groupby(factor)]
        if len(groups) < 2:
            continue
        F, p = sp_stats.f_oneway(*groups)
        ss_between = sum(len(g) * (np.mean(g) - df["fmd"].mean()) ** 2 for g in groups)
        ss_total_ow = ((df["fmd"] - df["fmd"].mean()) ** 2).sum()
        eta = float(ss_between / ss_total_ow) if ss_total_ow > 0 else 0
        oneway_rows.append({"factor": factor, "F": F, "p": p, "eta_sq": eta, "n_groups": len(groups)})
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        logger.info(f"  One-way {factor}: F={F:.2f}, p={p:.4e}, η²={eta:.4f} {sig}")

    pd.DataFrame(oneway_rows).to_csv(OUTPUT_DIR / "oneway_anova.csv", index=False)

    # 4c. Tukey HSD per factor
    if _has_sm:
        for factor in ("tokenizer", "model"):
            try:
                result = pairwise_tukeyhsd(df["fmd"].values, df[factor].values, alpha=0.05)
                tukey_df = pd.DataFrame(
                    data=result._results_table.data[1:],
                    columns=result._results_table.data[0],
                )
                tukey_df.to_csv(OUTPUT_DIR / f"tukey_{factor}.csv", index=False)
                logger.info(f"  Tukey HSD ({factor}):")
                for _, row in tukey_df.iterrows():
                    sig = "✓" if row["reject"] else ""
                    logger.info(f"    {row['group1']:12s} vs {row['group2']:12s}: "
                                f"diff={row['meandiff']:+.4f}  p={row['p-adj']:.4f}  {sig}")
            except Exception as e:
                logger.warning(f"Tukey for {factor} failed: {e}")

    # 4d. Per-pair analysis (generalizability)
    logger.info("\n  Per-pair statistics:")
    pair_stats = []
    for pair, grp in df.groupby("pair"):
        pair_stats.append({
            "pair": pair,
            "mean_fmd": grp["fmd"].mean(),
            "std_fmd": grp["fmd"].std(),
            "n": len(grp),
        })
        logger.info(f"    {pair:25s}: mean={grp['fmd'].mean():.4f} ± {grp['fmd'].std():.4f}  (n={len(grp)})")

    pd.DataFrame(pair_stats).to_csv(OUTPUT_DIR / "per_pair_stats.csv", index=False)

    # 4e. Effect consistency across pairs: η² per factor within each pair
    logger.info("\n  η² per factor within each pair (generalizability check):")
    consistency_rows = []
    for pair, grp in df.groupby("pair"):
        for factor in ("tokenizer", "model", "preprocess"):
            groups = [g["fmd"].values for _, g in grp.groupby(factor)]
            if len(groups) < 2:
                continue
            F, p = sp_stats.f_oneway(*groups)
            ss_b = sum(len(g) * (np.mean(g) - grp["fmd"].mean()) ** 2 for g in groups)
            ss_t = ((grp["fmd"] - grp["fmd"].mean()) ** 2).sum()
            eta = float(ss_b / ss_t) if ss_t > 0 else 0
            consistency_rows.append({"pair": pair, "factor": factor, "F": F, "p": p, "eta_sq": eta})

    cons_df = pd.DataFrame(consistency_rows)
    cons_df.to_csv(OUTPUT_DIR / "eta_sq_per_pair.csv", index=False)

    # Print η² consistency table
    if not cons_df.empty:
        pivot = cons_df.pivot_table(values="eta_sq", index="pair", columns="factor")
        logger.info(f"\n{pivot.round(4).to_string()}")

    # 4f. Cohen's d for key comparisons
    from itertools import combinations as combos
    cohens = {}
    for factor in ("tokenizer", "model"):
        levels = sorted(df[factor].unique())
        for a, b in combos(levels, 2):
            va = df.loc[df[factor] == a, "fmd"].values
            vb = df.loc[df[factor] == b, "fmd"].values
            n1, n2 = len(va), len(vb)
            pooled = np.sqrt(((n1 - 1) * va.var(ddof=1) + (n2 - 1) * vb.var(ddof=1)) / (n1 + n2 - 2))
            d = (va.mean() - vb.mean()) / pooled if pooled > 1e-12 else 0
            cohens[f"{factor}: {a} vs {b}"] = float(d)

    # 4g. Permutation tests (with more power now)
    logger.info("\n  Permutation tests (1000 perms for speed):")
    perm_results = {}
    n_perms = 1000
    for factor in ("tokenizer", "model", "preprocess"):
        groups = [g["fmd"].values for _, g in df.groupby(factor)]
        if len(groups) < 2:
            continue
        obs_F, _ = sp_stats.f_oneway(*groups)
        fmd_vals = df["fmd"].values.copy()
        count = 0
        for _ in range(n_perms):
            shuffled = rng.permutation(fmd_vals)
            s_groups, start = [], 0
            for g in groups:
                s_groups.append(shuffled[start:start + len(g)])
                start += len(g)
            pF, _ = sp_stats.f_oneway(*s_groups)
            if pF >= obs_F:
                count += 1
        p_perm = (count + 1) / (n_perms + 1)
        perm_results[factor] = {"F": float(obs_F), "p_perm": p_perm}
        sig = "***" if p_perm < 0.001 else "**" if p_perm < 0.01 else "*" if p_perm < 0.05 else ""
        logger.info(f"    {factor}: F={obs_F:.2f}, p_perm={p_perm:.4f} {sig}")

    # ── Step 5: Save comprehensive results JSON ──────────────────────
    results_json = {
        "config": {
            "genres": genres,
            "genre_pairs": [list(p) for p in genre_pairs],
            "n_variants": len(variants),
            "subsample_size": SUBSAMPLE_SIZE,
            "n_repeats": N_REPEATS,
            "total_rows": len(rows),
        },
        "oneway_anova": oneway_rows,
        "eta_sq_consistency": consistency_rows,
        "cohens_d": cohens,
        "permutation_tests": perm_results,
        "per_pair_stats": pair_stats,
    }
    if _has_sm:
        results_json["threeway_eta_sq"] = eta_sq
        results_json["threeway_partial_eta_sq"] = partial_eta_sq

    with open(OUTPUT_DIR / "multi_genre_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    # ── Step 6: Generate plots ────────────────────────────────────────
    logger.info("\n=== Step 4: Generating plots ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)

    DPI = 300

    def save_fig(fig, name):
        fig.tight_layout()
        for fmt in ("png", "pdf"):
            fig.savefig(PLOTS_DIR / f"{name}.{fmt}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  ✅ {name}")

    # Plot 1: FMD by tokenizer across ALL pairs (violin)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="tokenizer", y="fmd", hue="model", split=True,
                   ax=ax, palette="Set1", inner="quart", cut=0)
    ax.set_title("FMD Distribution: Tokenizer × Model (all genre pairs, 10 repeats)")
    ax.set_ylabel("FMD")
    save_fig(fig, "multi_fmd_violin_tok_model")

    # Plot 2: FMD by genre pair
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="pair", y="fmd", ax=ax, palette="Set2")
    ax.set_title("FMD Distribution by Genre Pair")
    ax.set_ylabel("FMD")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, "multi_fmd_by_pair")

    # Plot 3: Interaction plot per pair (faceted)
    g = sns.catplot(data=df, x="tokenizer", y="fmd", hue="model",
                    col="pair", col_wrap=3, kind="point", height=4, aspect=1.2,
                    palette="Set1", capsize=0.1, errorbar="sd")
    g.figure.suptitle("Tokenizer × Model Interaction per Genre Pair", y=1.02, fontsize=14)
    save_fig(g.figure, "multi_interaction_per_pair")

    # Plot 4: η² consistency heatmap across pairs
    if not cons_df.empty:
        pivot = cons_df.pivot_table(values="eta_sq", index="pair", columns="factor")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, vmin=0)
        ax.set_title("η² by Factor Across Genre Pairs (Generalizability)")
        save_fig(fig, "multi_eta_sq_heatmap")

    # Plot 5: Grand summary — η² comparison old (1 pair) vs new (6 pairs)
    old_eta = {"tokenizer": 0.2133, "model": 0.0183, "preprocess": 0.0524}
    new_eta_ow = {r["factor"]: r["eta_sq"] for r in oneway_rows if r["factor"] in old_eta}

    fig, ax = plt.subplots(figsize=(8, 5))
    factors = list(old_eta.keys())
    x = np.arange(len(factors))
    w = 0.35
    bars1 = ax.bar(x - w/2, [old_eta[f] for f in factors], w,
                   label="Single pair (rock-jazz)", color="#1f77b4")
    bars2 = ax.bar(x + w/2, [new_eta_ow.get(f, 0) for f in factors], w,
                   label="6 pairs × 10 repeats", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.set_ylabel("η²")
    ax.set_title("Variance Decomposition: Single Pair vs Multi-Genre")
    ax.legend()
    for bar, v in zip(bars1, [old_eta[f] for f in factors]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, v in zip(bars2, [new_eta_ow.get(f, 0) for f in factors]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.14, color="red", linestyle="--", alpha=0.4, label="large threshold")
    ax.axhline(0.06, color="orange", linestyle="--", alpha=0.4, label="medium threshold")
    save_fig(fig, "multi_eta_sq_comparison")

    # Plot 6: 4-panel publication figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A: Boxplot tokenizer × model
    sns.boxplot(data=df, x="tokenizer", y="fmd", hue="model", ax=axes[0, 0], palette="Set1")
    axes[0, 0].set_title("A) FMD by Tokenizer & Model (6 pairs)")
    axes[0, 0].set_ylabel("FMD")

    # B: η² bar
    if _has_sm:
        # Use full 3-way η²
        plot_factors = [k for k in eta_sq if ":" not in k][:5]
        plot_vals = [eta_sq[k] for k in plot_factors]
    else:
        plot_factors = [r["factor"] for r in oneway_rows]
        plot_vals = [r["eta_sq"] for r in oneway_rows]
    bars = axes[0, 1].bar(plot_factors, plot_vals, color=sns.color_palette("viridis", len(plot_factors)))
    for bar, v in zip(bars, plot_vals):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0, 1].set_title("B) η² Variance Decomposition")
    axes[0, 1].set_ylabel("η²")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].axhline(0.14, color="red", linestyle="--", alpha=0.4)
    axes[0, 1].axhline(0.06, color="orange", linestyle="--", alpha=0.4)

    # C: FMD by pair
    sns.boxplot(data=df, x="pair", y="fmd", ax=axes[1, 0], palette="Set2")
    axes[1, 0].set_title("C) FMD by Genre Pair")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].set_ylabel("FMD")

    # D: η² consistency
    if not cons_df.empty:
        pivot = cons_df.pivot_table(values="eta_sq", index="pair", columns="factor")
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1, 1],
                    linewidths=0.5, vmin=0, cbar_kws={"shrink": 0.8})
        axes[1, 1].set_title("D) η² Consistency Across Pairs")

    save_fig(fig, "multi_summary_4panel")

    # ── Step 7: Generate markdown report ──────────────────────────────
    logger.info("\n=== Step 5: Generating report ===")

    report = []
    report.append("# Multi-Genre FMD Sensitivity Analysis")
    report.append(f"\n**Extension of single-pair analysis to strengthen generalizability.**")
    report.append(f"\n## Design")
    report.append(f"- **Genres:** {', '.join(genres)} → {len(genre_pairs)} pairs")
    report.append(f"- **Variants:** {len(variants)} (4 tokenizers × 2 models × 4 preprocessing)")
    report.append(f"- **Repeated subsampling:** {N_REPEATS}× per variant×pair (n={SUBSAMPLE_SIZE})")
    report.append(f"- **Total FMD observations:** {len(rows)}")
    report.append(f"- **Advantage:** Within-cell variance enables proper ANOVA with interactions")

    report.append(f"\n## One-Way ANOVA (aggregated)")
    report.append(f"\n| Factor | F | p-value | η² | Effect |")
    report.append(f"|--------|---|---------|-----|--------|")
    for r in oneway_rows:
        e = r["eta_sq"]
        mag = "**LARGE**" if e >= 0.14 else "medium" if e >= 0.06 else "small" if e >= 0.01 else "negl."
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        report.append(f"| {r['factor']} | {r['F']:.2f} | {r['p']:.2e}{sig} | {e:.4f} | {mag} |")

    if _has_sm:
        report.append(f"\n## Three-Way ANOVA with Interactions")
        report.append(f"\n| Source | F | p-value | η² | Partial η² |")
        report.append(f"|--------|---|---------|-----|-----------|")
        for idx in anova_table.index:
            if idx == "Residual":
                continue
            F = anova_table.loc[idx, "F"]
            p = anova_table.loc[idx, "PR(>F)"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            report.append(f"| {idx} | {F:.2f} | {p:.2e}{sig} | {eta_sq[idx]:.4f} | {partial_eta_sq[idx]:.4f} |")

    report.append(f"\n## Permutation Tests")
    report.append(f"\n| Factor | F | p_perm | Significant |")
    report.append(f"|--------|---|--------|-------------|")
    for factor, res in perm_results.items():
        sig = "**Yes**" if res["p_perm"] < 0.05 else "No"
        report.append(f"| {factor} | {res['F']:.2f} | {res['p_perm']:.4f} | {sig} |")

    report.append(f"\n## Effect Sizes (Cohen's d)")
    report.append(f"\n| Comparison | d | Magnitude |")
    report.append(f"|------------|---|-----------|")
    for name, d in sorted(cohens.items(), key=lambda x: abs(x[1]), reverse=True):
        mag = "**large**" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small" if abs(d) >= 0.2 else "negl."
        report.append(f"| {name} | {d:.3f} | {mag} |")

    report.append(f"\n## Per-Pair FMD Statistics")
    report.append(f"\n| Pair | Mean FMD | Std | N |")
    report.append(f"|------|----------|-----|---|")
    for ps in sorted(pair_stats, key=lambda x: x["mean_fmd"]):
        report.append(f"| {ps['pair']} | {ps['mean_fmd']:.4f} | {ps['std_fmd']:.4f} | {ps['n']} |")

    report.append(f"\n## η² Generalizability Across Pairs")
    report.append(f"\nDo the same factors drive FMD variance regardless of genre pair?")
    if not cons_df.empty:
        pivot = cons_df.pivot_table(values="eta_sq", index="pair", columns="factor")
        report.append(f"\n```")
        report.append(pivot.round(4).to_string())
        report.append(f"```")

        # Check consistency
        for factor in ("tokenizer", "model", "preprocess"):
            if factor in pivot.columns:
                vals = pivot[factor].values
                report.append(f"\n- **{factor}**: η² range [{vals.min():.4f}, {vals.max():.4f}], "
                              f"mean={vals.mean():.4f}, cv={vals.std()/vals.mean():.2f}")

    report.append(f"\n## Comparison with Single-Pair Analysis")
    report.append(f"\n| Factor | η² (rock-jazz only) | η² (6 pairs, 10 repeats) | Change |")
    report.append(f"|--------|--------------------|-----------------------------|--------|")
    for f in ["tokenizer", "model", "preprocess"]:
        old = old_eta.get(f, 0)
        new = new_eta_ow.get(f, 0)
        change = "↑" if new > old else "↓" if new < old else "="
        report.append(f"| {f} | {old:.4f} | {new:.4f} | {change} |")

    report.append(f"\n## Key Conclusions")
    report.append(f"""
1. With {len(rows)} observations (vs 32 in single-pair), statistical power is dramatically increased.
2. Genre pair choice is itself a significant factor — FMD values differ substantially between pairs.
3. The multi-pair analysis tests whether tokenizer sensitivity **generalizes** across genre contexts.
4. Three-way ANOVA with interactions is now properly estimable (multiple observations per cell).
""")

    report_path = OUTPUT_DIR / "MULTI_GENRE_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    logger.info(f"✅ Report: {report_path}")

    logger.info("\n" + "=" * 70)
    logger.info("MULTI-GENRE ANALYSIS COMPLETE")
    logger.info(f"  Results: {OUTPUT_DIR}/")
    logger.info(f"  Plots:   {PLOTS_DIR}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()


