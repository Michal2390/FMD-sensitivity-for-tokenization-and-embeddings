"""Publication-quality plots for Lakh MIDI validation results.

Generates:
- Boxplot FMD per tokenizer (panels per model)
- Heatmap η² variance decomposition
- t-SNE scatter (colour = genre, shape = tokenizer, panels = model)
- Grouped bar chart intra- vs inter-genre cosine similarity
- Scatter FMD vs token entropy with regression line
- ANOVA summary table as figure
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


_DPI = 300
_FORMATS = ("png", "pdf")


def _save(fig: plt.Figure, base_path: Path) -> str:
    """Save figure in PNG and PDF at publication DPI."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for fmt in _FORMATS:
        out = base_path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(base_path.with_suffix(".png"))


def _parse_variant(variant_str: str) -> Dict[str, str]:
    """Parse 'tok=REMI|model=CLaMP-1|vel=on|quant=off' into dict."""
    parts = {}
    for segment in variant_str.split("|"):
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k] = v
    return parts


def generate_lakh_plots(config: Dict) -> Dict[str, str]:
    """Generate all Lakh validation plots from saved artefacts.

    Reads CSVs and JSONs from ``results/reports/lakh/`` and writes figures
    to ``results/plots/paper/``.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)

    lakh_cfg = config.get("lakh", {})
    report_dir = Path(lakh_cfg.get("output_dir", "results/reports/lakh"))
    plots_dir = Path(lakh_cfg.get("plots_dir", "results/plots/paper"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. Boxplot FMD per tokenizer (faceted by model)
    # ------------------------------------------------------------------
    pairwise_csv = report_dir / "lakh_pairwise_fmd.csv"
    if pairwise_csv.exists():
        df = pd.read_csv(pairwise_csv)
        df = df[df["valid"] == True]  # noqa: E712
        if not df.empty:
            fig, axes = plt.subplots(1, df["model"].nunique(), figsize=(6 * df["model"].nunique(), 5), sharey=True)
            if df["model"].nunique() == 1:
                axes = [axes]
            for ax, (model, grp) in zip(axes, df.groupby("model")):
                sns.boxplot(data=grp, x="tokenizer", y="fmd", ax=ax, palette="Set2")
                ax.set_title(f"Model: {model}")
                ax.set_xlabel("Tokenizer")
                ax.set_ylabel("FMD (rock vs classical)")
            outputs["lakh_fmd_boxplot"] = _save(fig, plots_dir / "lakh_fmd_boxplot_by_tokenizer")

            # Also a single combined boxplot with hue=model
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            sns.boxplot(data=df, x="tokenizer", y="fmd", hue="model", ax=ax2, palette="Set1")
            ax2.set_title("FMD Sensitivity: Tokenizer × Model (Lakh rock vs classical)")
            ax2.set_xlabel("Tokenizer")
            ax2.set_ylabel("FMD")
            outputs["lakh_fmd_boxplot_combined"] = _save(fig2, plots_dir / "lakh_fmd_boxplot_combined")

    # ------------------------------------------------------------------
    # 2. η² variance decomposition heatmap
    # ------------------------------------------------------------------
    sens_json = report_dir / "sensitivity_results.json"
    if sens_json.exists():
        with open(sens_json, "r") as fh:
            sens = json.load(fh)

        eta_sq = sens.get("eta_squared")
        if eta_sq:
            factors = list(eta_sq.keys())
            values = [eta_sq[f] for f in factors]
            fig, ax = plt.subplots(figsize=(max(6, len(factors) * 1.5), 4))
            colours = sns.color_palette("viridis", len(factors))
            bars = ax.bar(factors, values, color=colours)
            ax.set_ylabel("η² (proportion of variance)")
            ax.set_title("Variance Decomposition (η²) – FMD Sensitivity Factors")
            ax.set_ylim(0, max(values) * 1.3 if values else 1.0)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            ax.tick_params(axis="x", rotation=30)
            outputs["lakh_eta_squared"] = _save(fig, plots_dir / "lakh_eta_squared")

        # Cohen's d
        cohens = sens.get("cohens_d", {})
        if cohens:
            cd_df = pd.DataFrame([{"comparison": k, "cohens_d": v} for k, v in cohens.items()])
            cd_df = cd_df.sort_values("cohens_d", key=abs, ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, max(4, len(cd_df) * 0.35)))
            sns.barplot(data=cd_df, y="comparison", x="cohens_d", ax=ax, palette="coolwarm")
            ax.set_title("Effect Sizes (Cohen's d)")
            ax.set_xlabel("Cohen's d")
            outputs["lakh_cohens_d"] = _save(fig, plots_dir / "lakh_cohens_d")

    # ------------------------------------------------------------------
    # 3. Cosine similarity: intra vs inter
    # ------------------------------------------------------------------
    cosine_csv = report_dir / "cosine_similarity.csv"
    if cosine_csv.exists():
        cos_df = pd.read_csv(cosine_csv)
        if not cos_df.empty:
            # Parse variant to get tokenizer/model
            parsed = cos_df["variant"].apply(_parse_variant).apply(pd.Series)
            cos_df = pd.concat([cos_df, parsed], axis=1)

            melted = cos_df.melt(
                id_vars=["variant", "tok", "model"],
                value_vars=["intra_a", "intra_b", "inter"],
                var_name="similarity_type",
                value_name="cosine_sim",
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=melted, x="tok", y="cosine_sim", hue="similarity_type", ax=ax, palette="Set2")
            ax.set_title("Intra- vs Inter-Genre Cosine Similarity")
            ax.set_xlabel("Tokenizer")
            ax.set_ylabel("Mean Cosine Similarity")
            ax.legend(title="Type")
            outputs["lakh_cosine_similarity"] = _save(fig, plots_dir / "lakh_cosine_similarity")

    # ------------------------------------------------------------------
    # 4. t-SNE scatter
    # ------------------------------------------------------------------
    tsne_csv = report_dir / "tsne_projections.csv"
    if tsne_csv.exists():
        tsne_df = pd.read_csv(tsne_csv)
        if not tsne_df.empty:
            parsed = tsne_df["variant"].apply(_parse_variant).apply(pd.Series)
            tsne_df = pd.concat([tsne_df, parsed], axis=1)

            # Select a few representative variants for clarity
            unique_variants = tsne_df["variant"].unique()
            # Pick up to 4 variants (one per tokenizer, first model)
            sample_variants = []
            seen_tok = set()
            for v in unique_variants:
                p = _parse_variant(v)
                if p.get("tok") not in seen_tok and p.get("vel") == "on" and p.get("quant") == "off":
                    sample_variants.append(v)
                    seen_tok.add(p.get("tok"))
                if len(sample_variants) >= 4:
                    break

            if sample_variants:
                sub = tsne_df[tsne_df["variant"].isin(sample_variants)]
                g = sns.FacetGrid(sub, col="tok", col_wrap=2, height=4, sharex=False, sharey=False)
                g.map_dataframe(sns.scatterplot, x="tsne1", y="tsne2", hue="genre",
                                style="genre", palette="Set1", alpha=0.7, s=30)
                g.add_legend()
                g.figure.suptitle("t-SNE: Rock vs Classical Embeddings by Tokenizer", y=1.02)
                outputs["lakh_tsne"] = _save(g.figure, plots_dir / "lakh_tsne_by_tokenizer")

    # ------------------------------------------------------------------
    # 5. Token statistics
    # ------------------------------------------------------------------
    tok_csv = report_dir / "token_statistics.csv"
    if tok_csv.exists():
        tok_df = pd.read_csv(tok_csv)
        if not tok_df.empty:
            parsed = tok_df["variant"].apply(_parse_variant).apply(pd.Series)
            tok_df = pd.concat([tok_df, parsed], axis=1)

            for metric in ("length_mean", "entropy_mean"):
                if metric in tok_df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(data=tok_df, x="tok", y=metric, hue="genre", ax=ax, palette="Set1")
                    nice = metric.replace("_mean", "").title()
                    ax.set_title(f"Token {nice} by Tokenizer and Genre")
                    ax.set_xlabel("Tokenizer")
                    ax.set_ylabel(f"Mean {nice}")
                    outputs[f"lakh_token_{metric}"] = _save(fig, plots_dir / f"lakh_token_{metric}")

    # ------------------------------------------------------------------
    # 6. FMD vs token entropy scatter
    # ------------------------------------------------------------------
    corr_json = report_dir / "fmd_token_correlations.json"
    if corr_json.exists() and pairwise_csv.exists() and tok_csv.exists():
        try:
            fmd_df = pd.read_csv(pairwise_csv)
            fmd_df = fmd_df[fmd_df["valid"] == True]  # noqa
            tok_df = pd.read_csv(tok_csv)

            # Merge on variant
            if "entropy_mean" in tok_df.columns:
                # Average entropy across genres per variant
                tok_agg = tok_df.groupby("variant")["entropy_mean"].mean().reset_index()
                merged = fmd_df.merge(tok_agg, on="variant", how="inner")
                if not merged.empty:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sns.regplot(data=merged, x="entropy_mean", y="fmd", ax=ax,
                                scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
                    ax.set_title("FMD vs Token Entropy")
                    ax.set_xlabel("Mean Token Entropy")
                    ax.set_ylabel("FMD (rock vs classical)")
                    outputs["lakh_fmd_vs_entropy"] = _save(fig, plots_dir / "lakh_fmd_vs_entropy")
        except Exception as exc:
            logger.warning(f"FMD-entropy scatter failed: {exc}")

    # ------------------------------------------------------------------
    # 7. ANOVA table as figure
    # ------------------------------------------------------------------
    anova_csv = report_dir / "anova_table.csv"
    if anova_csv.exists():
        try:
            anova_df = pd.read_csv(anova_csv, index_col=0)
            if not anova_df.empty:
                fig, ax = plt.subplots(figsize=(10, max(2, len(anova_df) * 0.5 + 1)))
                ax.axis("off")
                display_df = anova_df.round(4).reset_index()
                table = ax.table(
                    cellText=display_df.values,
                    colLabels=display_df.columns,
                    cellLoc="center",
                    loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.2, 1.4)
                ax.set_title("Three-Way ANOVA Table", fontsize=12, pad=20)
                outputs["lakh_anova_table"] = _save(fig, plots_dir / "lakh_anova_table")
        except Exception as exc:
            logger.warning(f"ANOVA table figure failed: {exc}")

    # ------------------------------------------------------------------
    if not outputs:
        logger.warning("No Lakh plots generated – missing report files in " + str(report_dir))
    else:
        logger.info(f"Generated {len(outputs)} Lakh validation plots in {plots_dir}")

    return outputs

